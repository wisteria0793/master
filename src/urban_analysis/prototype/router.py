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

    def _solve_tsptw_dfs(self, cost_matrix, time_matrix, time_windows, start_time_min, stay_duration_min):
        """
        DFSとパレート支配に基づく枝刈りを用いた時間制約付き経路探索。
        開いていない（待ち時間が長い、または閉店後）POIはスキップし、
        「最も多くのPOIを訪問でき、かつ景観コストが最小」となるルートを探索する。
        """
        n = len(cost_matrix)
        best_visited_count = -1
        best_cost = float('inf')
        best_path = None

        # [current_node, mask, current_cost, current_time, path]
        stack = [(0, 1, 0.0, start_time_min, [0])]
        memo = {}
        
        while stack:
            curr_node, mask, curr_cost, curr_time, path = stack.pop()
            
            # パレート支配による枝刈り (同じ訪問ノード集合・現在地において、コストも時間も劣るなら探索打ち切り)
            state = (mask, curr_node)
            is_dominated = False
            if state in memo:
                for prev_cost, prev_time in memo[state]:
                    if prev_cost <= curr_cost and prev_time <= curr_time:
                        is_dominated = True
                        break
                if is_dominated:
                    continue
                # 現在のパスが過去のパスを支配している場合は更新
                memo[state] = [(c, t) for c, t in memo[state] if not (curr_cost <= c and curr_time <= t)]
                memo[state].append((curr_cost, curr_time))
            else:
                memo[state] = [(curr_cost, curr_time)]
                
            visited_count = bin(mask).count('1')
            
            # 帰還コストを加算してベストルートを更新するかチェック
            ret_cost = cost_matrix[curr_node][0]
            final_cost = curr_cost + ret_cost
            
            # 訪問数が多い、または訪問数が同じでコストが低い場合に更新
            if visited_count > best_visited_count or (visited_count == best_visited_count and final_cost < best_cost):
                best_visited_count = visited_count
                best_cost = final_cost
                best_path = path + [0]
            
            # 次のノードへの遷移
            for next_node in range(1, n):
                if not (mask & (1 << next_node)):
                    move_cost = cost_matrix[curr_node][next_node]
                    move_time = time_matrix[curr_node][next_node]
                    
                    arrival_time = curr_time + move_time
                    open_t, close_t = time_windows[next_node]
                    
                    # 閉店時間を過ぎていたらこの経路は無効（スキップ）
                    if arrival_time > close_t:
                        continue
                        
                    # 開店前なら待機するが、待ち時間が長すぎる場合は立ち寄らない
                    wait_time = max(0, open_t - arrival_time)
                    if wait_time > 30: # 30分以上待つならスキップ
                        continue
                        
                    departure_time = arrival_time + wait_time + stay_duration_min
                    new_cost = curr_cost + move_cost
                    new_mask = mask | (1 << next_node)
                    
                    stack.append((next_node, new_mask, new_cost, departure_time, path + [next_node]))
                    
        return best_path

    def generate_route(self, target_poi, recommended_df, street_df, start_time_min=600, stay_duration_min=30):
        """
        起点POIから推薦POI群を巡る景観重視・時間枠考慮のルートを生成する
        start_time_min: 出発時刻 (分) 例: 10:00 -> 600
        stay_duration_min: 各POIの滞在時間 (分)
        """
        if recommended_df.empty:
            return None
            
        start_coord = (target_poi['lat'], target_poi['lng'])
        
        # 1. 起点POIの「景観クラスタ」を取得（phase3_recommenderで計算済みの値を使用）
        if 'ls_cluster' in target_poi:
            target_landscape_cluster = target_poi['ls_cluster']
        else:
            # フォールバック (万が一キーがない場合)
            from scipy.spatial import KDTree
            street_coords = street_df[['lat', 'lng']].values
            tree = KDTree(street_coords)
            _, idx = tree.query([start_coord[0], start_coord[1]])
            target_landscape_cluster = street_df.iloc[idx]['cluster']
            
        print(f"経路探索用 景観クラスタ: {target_landscape_cluster} を適用します。")

        # 起点 + 推薦POIのリストと時間枠の準備
        poi_coords = [start_coord] + [(r['lat'], r['lng']) for _, r in recommended_df.iterrows()]
        
        # 時間枠の取得 (open_time, close_time)
        target_tw = (target_poi.get('open_time', 0), target_poi.get('close_time', 1440))
        time_windows = [target_tw] + [(r.get('open_time', 0), r.get('close_time', 1440)) for _, r in recommended_df.iterrows()]
        
        n_pois = len(poi_coords)
        
        # 2. 範囲を絞ったサブグラフの抽出
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

        # 3. KDE計算用のイベントポイント（同じ「景観クラスタ」の地点）を用意
        event_nodes = []
        if street_df is not None:
            cluster_events = street_df[street_df['cluster'] == target_landscape_cluster]
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

        # 4. 各POI間の景観コスト行列と移動時間行列を事前計算
        print(f"{n_pois}個のPOI間の景観コスト行列・移動時間行列を生成中 ({n_pois * (n_pois-1)} ペア)...")
        poi_nodes = []
        for p in poi_coords:
            poi_nodes.append(ox.distance.nearest_nodes(subgraph, X=p[1], Y=p[0]))
            
        cost_matrix = np.zeros((n_pois, n_pois))
        time_matrix = np.zeros((n_pois, n_pois))
        WALKING_SPEED_M_PER_MIN = 72.0 # 約1.2m/s
        
        for i in range(n_pois):
            lengths_cost = nx.single_source_dijkstra_path_length(subgraph, poi_nodes[i], weight='kde_cost')
            lengths_dist = nx.single_source_dijkstra_path_length(subgraph, poi_nodes[i], weight='length')
            for j in range(n_pois):
                if i == j:
                    cost_matrix[i][j] = 0
                    time_matrix[i][j] = 0
                else:
                    cost_matrix[i][j] = lengths_cost.get(poi_nodes[j], 1e9)
                    dist = lengths_dist.get(poi_nodes[j], 1e9)
                    time_matrix[i][j] = dist / WALKING_SPEED_M_PER_MIN

        # 5. TSPTWによる最適順序の決定
        print(f"TSPTW最適化を実行中... (出発時刻: {start_time_min//60:02d}:{start_time_min%60:02d}, 滞在時間: {stay_duration_min}分)")
        best_order = self._solve_tsptw_dfs(cost_matrix, time_matrix, time_windows, start_time_min, stay_duration_min)
        
        # スケジュール（到着予定時刻）の計算
        schedules = []
        current_time = start_time_min
        for idx in range(len(best_order)):
            if idx == 0:
                schedules.append({'arrival': current_time, 'wait': 0, 'departure': current_time})
            else:
                u_idx = best_order[idx-1]
                v_idx = best_order[idx]
                move_time = time_matrix[u_idx][v_idx]
                arrival = current_time + move_time
                open_t, _ = time_windows[v_idx]
                wait = max(0, open_t - arrival)
                departure = arrival + wait + (stay_duration_min if v_idx != 0 else 0)
                
                schedules.append({
                    'arrival': arrival,
                    'wait': wait,
                    'departure': departure
                })
                current_time = departure
        
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
        return route_geometry, best_order, schedules
