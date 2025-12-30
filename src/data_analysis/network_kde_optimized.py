# -*- coding: utf-8 -*-
"""
このスクリプトは、特定の景観クラスタについて、
道路ネットワーク上でのカーネル密度推定（Network KDE）を実行し、
そのクラスタが集中しているエリア（ホットスポット）を可視化します。
【高速化版】

処理手順：
1. GNNエンベディングと座標データを読み込み、再度クラスタリングを実行して各地点の所属クラスタを決定します。
2. OSMnxで道路ネットワークグラフを読み込みます。
3. 分析対象とするクラスタ（例: 緑地エリア）に属する地点を特定します。
4. 各地点から影響を受けるノードの密度を先に計算し、その後エッジの密度を算出する高速な手法で計算します。
5. 密度スコアに応じて道路を色分けしたインタラクティブな地図を生成し、保存します。
"""

import pandas as pd
import numpy as np
import os
import json
import folium
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from shapely.geometry import LineString
from tqdm import tqdm

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
N_CLUSTERS = 20
EMBEDDING_DIM = 64

# --- Network KDE パラメータ ---
TARGET_CLUSTER_ID = 2  # 分析対象のクラスタID (例: 2は緑地エリアと仮定)
BANDWIDTH = 500       # 密度計算の範囲 (メートル)
THRESHOLD_DENSITY = 0.05 # これ未満の密度は表示しない

# --- パス設定 ---
EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', f'embeddings_dim{EMBEDDING_DIM}.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OSM_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'osm_hakodate', 'Hakodate.osm.xml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'docs', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_clustered_data():
    """GNNエンベディングを読み込み、クラスタリングして座標と結合する"""
    print("データ読み込みとクラスタリングを実行中...")
    embedding_df = pd.read_csv(EMBEDDING_PATH)
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item}
    
    embedding_df['latitude'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    embedding_df['longitude'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    embedding_df.dropna(subset=['latitude', 'longitude'], inplace=True)

    feature_cols = embedding_df.columns.drop(['point_id', 'latitude', 'longitude'])
    features = embedding_df[feature_cols].values
    features_scaled = StandardScaler().fit_transform(features)

    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    embedding_df['cluster'] = clusters - 1
    
    return embedding_df

def optimized_network_kernel_density(G, event_points, bandwidth):
    """ネットワークカーネル密度を高速に計算する"""
    
    # 1. グラフの全ノードについて、密度を0で初期化
    node_densities = pd.Series(0.0, index=list(G.nodes()))

    print("各イベントポイントからの影響を計算中...")
    # 2. 各イベントポイントについてループ
    for _, event in tqdm(event_points.iterrows(), total=len(event_points), desc="KDE Calculation"):
        event_node = event['osm_node']
        
        if not G.has_node(event_node):
            continue

        # イベントノードからバンド幅内の全ノードへの最短距離を一度に計算
        reachable_nodes_dist = nx.single_source_dijkstra_path_length(
            G, source=event_node, cutoff=bandwidth, weight='length'
        )
        
        # 影響下のノードにカーネル値を加算
        for node, dist in reachable_nodes_dist.items():
            kernel_val = (1 - (dist / bandwidth)**2)**2
            node_densities[node] += kernel_val

    print("エッジ密度を計算中...")
    # 3. エッジの密度を、両端ノードの密度の平均として計算
    edge_densities = {}
    for u, v, data in G.edges(data=True):
        edge_len = data.get('length', 1)
        avg_node_density = (node_densities.get(u, 0) + node_densities.get(v, 0)) / 2
        edge_densities[(u, v, 0)] = avg_node_density / edge_len if edge_len > 0 else 0

    return edge_densities

def main():
    """メインの処理フロー"""
    clustered_df = get_clustered_data()
    
    print(f"分析対象クラスタ: {TARGET_CLUSTER_ID}")
    event_points = clustered_df[clustered_df['cluster'] == TARGET_CLUSTER_ID]
    print(f"対象クラスタの地点数: {len(event_points)}")
    if len(event_points) == 0:
        print("対象クラスタの地点が見つからないため、処理を終了します。")
        return

    print("OSM道路網を読み込み中...")
    G = ox.graph_from_xml(OSM_PATH, simplify=True)
    
    # 地点とOSMノードを紐付け
    event_coords = list(zip(event_points['latitude'], event_points['longitude']))
    event_osm_nodes = ox.nearest_nodes(G, [c[1] for c in event_coords], [c[0] for c in event_coords])
    event_points = event_points.copy()
    event_points.loc[:, 'osm_node'] = event_osm_nodes

    print("Network KDEの計算を開始します...")
    edge_densities = optimized_network_kernel_density(G, event_points, BANDWIDTH)
    
    density_series = pd.Series(edge_densities)
    
    print("KDE計算完了。結果を可視化中...")
    
    # ベースマップを作成
    m = folium.Map(location=(clustered_df['latitude'].mean(), clustered_df['longitude'].mean()), 
                   zoom_start=14, tiles='OpenStreetMap')

    # 密度の正規化とカラーマッピングの準備
    norm = None
    if not density_series.empty and density_series.max() > 0:
        # 閾値より大きい密度を持つエッジのみを抽出
        pos_densities = density_series[density_series >= THRESHOLD_DENSITY]
        if not pos_densities.empty:
            from matplotlib.colors import LogNorm
            # LogNorm（対数正規化）を使用して、値の偏りが大きいデータでも色の違いを表現
            # vminを閾値以上にする
            norm = LogNorm(vmin=pos_densities.min(), vmax=pos_densities.max())
            cmap = plt.get_cmap('plasma')
            print(f"密度の範囲 (min, max) for coloring: ({pos_densities.min():.4f}, {pos_densities.max():.4f}) (閾値: {THRESHOLD_DENSITY})")
        else:
            norm = None
            print(f"密度が計算されましたが、閾値 {THRESHOLD_DENSITY} を超える有意な値はありませんでした。")
    else:
        norm = None
        print("密度が計算されませんでした。")

    # エッジを手動で描画
    if norm: # 有意な密度が計算された場合のみホットスポットを描画
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_key = (u, v, key) if key is not None else (u, v, 0)
            density = density_series.get(edge_key, 0)
            
            # 密度が閾値以上のエッジのみを描画
            if density >= THRESHOLD_DENSITY:
                color_rgba = cmap(norm(density))
                color_hex = '#%02x%02x%02x' % (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
                weight = 5
                opacity = 0.8
                
                if 'geometry' in data:
                    edge_geom = data['geometry']
                else:
                    point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
                    point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
                    edge_geom = LineString([point_u, point_v])
                
                points = list(edge_geom.coords)
                folium.PolyLine(
                    locations=[(lat, lon) for lon, lat in points],
                    color=color_hex,
                    weight=weight,
                    opacity=opacity,
                    tooltip=f"Density: {density:.4f}"
                ).add_to(m)

    # 地図を保存
    map_output_path = os.path.join(OUTPUT_DIR, f'network_kde_optimized_cluster_{TARGET_CLUSTER_ID}_bw{BANDWIDTH}.html')
    m.save(map_output_path)

    print(f"Network KDEの可視化マップを保存しました: {map_output_path}")
    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()
