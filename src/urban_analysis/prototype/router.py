import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# プロトタイプの経路探索用パラメータ
KDE_BANDWIDTH_M = 500
KDE_ALPHA = 10.0
OSM_XML_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'osm_hakodate', 'Hakodate.osm.xml')
CACHED_GRAPH_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'osm', 'hakodate_walk.graphml')

class RouteGenerator:
    def __init__(self, bandwidth=KDE_BANDWIDTH_M, alpha=KDE_ALPHA):
        self.bandwidth = bandwidth
        self.alpha = alpha
        
        # XMLから一度だけ生成し、以降はgraphmlから高速読み込み
        if os.path.exists(CACHED_GRAPH_PATH):
            print("キャッシュされたネットワークグラフを読み込み中...")
            self.G = ox.load_graphml(CACHED_GRAPH_PATH)
        else:
            print("OSM XMLファイルからネットワークグラフを構築中（初回のみ数分かかります）...")
            self.G = ox.graph_from_xml(OSM_XML_PATH, simplify=True)
            os.makedirs(os.path.dirname(CACHED_GRAPH_PATH), exist_ok=True)
            ox.save_graphml(self.G, CACHED_GRAPH_PATH)
            print("グラフのキャッシュを保存しました。")

    def optimized_network_kernel_density(self, subgraph, event_nodes):
        """指定されたサブグラフ上でNKDEを計算する"""
        node_densities = pd.Series(0.0, index=list(subgraph.nodes()))

        print(f"KDE計算: {len(event_nodes)}地点からの影響を道路に伝播中...")
        for event_node in tqdm(event_nodes, desc="NKDE Propagation", leave=False):
            if not subgraph.has_node(event_node):
                continue
                
            try:
                reachable_nodes_dist = nx.single_source_dijkstra_path_length(
                    subgraph, source=event_node, cutoff=self.bandwidth, weight='length'
                )
                for node, dist in reachable_nodes_dist.items():
                    kernel_val = (1 - (dist / self.bandwidth)**2)**2
                    node_densities[node] += kernel_val
            except Exception:
                pass

        edge_densities = {}
        for u, v, data in subgraph.edges(data=True):
            edge_len = float(data.get('length', 1.0))
            avg_node_density = (node_densities.get(u, 0) + node_densities.get(v, 0)) / 2
            edge_densities[(u, v, 0)] = avg_node_density / edge_len if edge_len > 0 else 0

        return edge_densities

    def _solve_tsp_held_karp(self, cost_matrix):
        """動的計画法 (Held-Karp) によるTSP厳密解の算出"""
        n = len(cost_matrix)
        # dp[(mask, last_node)] = (min_cost, parent_node)
        dp = {}

        # 初期状態: 起点(0)から各ノード(i)へのコスト
        for i in range(1, n):
            dp[(1 << i | 1, i)] = (cost_matrix[0][i], 0)

        # 部分集合のサイズを2からn-1まで増やす (起点0は常に含める)
        import itertools
        for size in range(2, n):
            for subset in itertools.combinations(range(1, n), size):
                mask = 1
                for node in subset:
                    mask |= (1 << node)
                
                for next_node in range(1, n):
                    if not (mask & (1 << next_node)):
                        continue
                    
                    prev_mask = mask ^ (1 << next_node)
                    res = []
                    for prev_node in range(1, n):
                        if prev_node == next_node or not (prev_mask & (1 << prev_node)):
                            continue
                        if (prev_mask, prev_node) in dp:
                            res.append((dp[(prev_mask, prev_node)][0] + cost_matrix[prev_node][next_node], prev_node))
                    
                    if res:
                        dp[(mask, next_node)] = min(res)

        # 全ノードを巡回した状態 (Hamiltonian Path の終点を探す)
        full_mask = (1 << n) - 1
        res = []
        for i in range(1, n):
            if (full_mask, i) in dp:
                res.append((dp[(full_mask, i)][0], i))
        
        if not res:
            return list(range(n)) # フォールバック

        best_res = min(res)
        min_total_cost = best_res[0]
        last_node = best_res[1]
        
        # 経路の復元
        path = []
        curr_mask = full_mask
        curr_node = last_node
        while curr_node != 0:
            path.append(curr_node)
            prev_node = dp[(curr_mask, curr_node)][1]
            curr_mask ^= (1 << curr_node)
            curr_node = prev_node
        path.append(0)
        return path[::-1]

    def generate_route(self, target_poi, recommended_df, all_cluster_points_df):
        if recommended_df.empty:
            return None
            
        start_coord = (target_poi['lat'], target_poi['lng'])
        target_cluster = target_poi['cluster']
        
        # 起点 + 推薦POIのリスト
        poi_coords = [start_coord] + [(r['lat'], r['lng']) for _, r in recommended_df.iterrows()]
        n_pois = len(poi_coords)
        
        # 1. 範囲を絞ったサブグラフの抽出
        print("\n経路周辺のサブグラフを抽出中...")
        try:
            lats, lngs = [p[0] for p in poi_coords], [p[1] for p in poi_coords]
            padding = 0.01 
            # OSMnx 2.x: bboxは(left, bottom, right, top) = (min_lng, min_lat, max_lng, max_lat)
            bbox = (min(lngs)-padding, min(lats)-padding, max(lngs)+padding, max(lats)+padding)
            subgraph = ox.truncate.truncate_graph_bbox(self.G, bbox)
        except Exception as e:
            print(f"サブグラフ抽出エラー: {e}")
            subgraph = self.G

        # 2. KDE計算用のイベントポイント（同じ景観クラスタの地点）を用意
        event_nodes = []
        if all_cluster_points_df is not None:
            cluster_events = all_cluster_points_df[all_cluster_points_df['cluster'] == target_cluster]
            if not cluster_events.empty:
                try:
                    nodes = ox.distance.nearest_nodes(subgraph, X=cluster_events['lng'].tolist(), Y=cluster_events['lat'].tolist())
                    event_nodes = list(set(nodes))
                except Exception:
                    pass
            
        # 3. Network KDE の計算とエッジコスト設定
        print("道路ネットワークに景観評価(NKDE)を反映中...")
        if event_nodes:
            edge_densities = self.optimized_network_kernel_density(subgraph, event_nodes)
        else:
            edge_densities = {}
            
        for u, v, k, data in subgraph.edges(data=True, keys=True):
            length = float(data.get('length', 1.0))
            density = edge_densities.get((u, v, 0), 0.0)
            data['kde_cost'] = length / (1.0 + self.alpha * density)

        # 4. 各POI間の景観コスト行列を事前計算
        print(f"10個のPOI間の景観コスト行列を生成中 ({n_pois * (n_pois-1)} ペア)...")
        poi_nodes = []
        for p in poi_coords:
            poi_nodes.append(ox.distance.nearest_nodes(subgraph, X=p[1], Y=p[0]))
            
        cost_matrix = np.zeros((n_pois, n_pois))
        for i in range(n_pois):
            lengths = nx.single_source_dijkstra_path_length(subgraph, poi_nodes[i], weight='kde_cost')
            for j in range(n_pois):
                if i == j:
                    cost_matrix[i][j] = 0
                else:
                    cost_matrix[i][j] = lengths.get(poi_nodes[j], 1e9)

        # 5. TSPによる最適順序の決定
        print("TSP最適化を実行中...")
        best_order = self._solve_tsp_held_karp(cost_matrix)
        
        # 6. 最終的な経路ジオメトリの構築
        print("最終ルートを生成中...")
        route_geometry = []
        total_length = 0.0
        
        for idx in range(len(best_order) - 1):
            u_idx, v_idx = best_order[idx], best_order[idx+1]
            try:
                path = nx.shortest_path(subgraph, poi_nodes[u_idx], poi_nodes[v_idx], weight='kde_cost')
                for j in range(len(path)-1):
                    u, v = path[j], path[j+1]
                    edge_data = min(subgraph[u][v].values(), key=lambda d: d.get('kde_cost', float('inf')))
                    total_length += float(edge_data.get('length', 1.0))
                    if 'geometry' in edge_data:
                        coords = list(edge_data['geometry'].coords)
                        route_geometry.extend([(lat, lng) for lng, lat in coords])
                    else:
                        route_geometry.extend([(subgraph.nodes[u]['y'], subgraph.nodes[u]['x']), (subgraph.nodes[v]['y'], subgraph.nodes[v]['x'])])
            except nx.NetworkXNoPath:
                continue
                
        print(f"-> 拡張ルート算出完了 (総物理距離: 約{total_length:.0f}m)")
        return route_geometry, best_order
