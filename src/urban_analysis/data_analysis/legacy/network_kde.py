# -*- coding: utf-8 -*-
"""
このスクリプトは、特定の景観クラスタについて、
道路ネットワーク上でのカーネル密度推定（Network KDE）を実行し、
そのクラスタが集中しているエリア（ホットスポット）を可視化します。

処理手順：
1. GNNエンベディングと座標データを読み込み、再度クラスタリングを実行して各地点の所属クラスタを決定します。
2. OSMnxで道路ネットワークグラフを読み込みます。
3. 分析対象とするクラスタ（例: 緑地エリア）に属する地点を特定します。
4. 道路網の各エッジ（道路セグメント）上で密度を計算します。
   - 各道路セグメントから、対象クラスタの地点までのネットワーク距離を計算。
   - カーネル関数を用いて距離を重み付けし、密度スコアを算出。
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

def network_kernel_density(G, event_points, bandwidth):
    """ネットワークカーネル密度を計算する"""
    densities = {}
    
    # 道路網の各エッジ（u,v,key）に対してループ
    for u, v, key, data in tqdm(G.edges(keys=True, data=True), desc="Calculating Network KDE", unit="road"):
        edge_len = data.get('length', 1) # lengthがない場合は1とする

        # エッジのジオメトリを取得または作成
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            # ノード座標からジオメトリを再構築
            point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
            point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
            edge_geom = LineString([point_u, point_v])

        # エッジの中間点を評価点とする
        eval_point_geom = edge_geom.interpolate(0.5, normalized=True)
        eval_node, _ = ox.nearest_nodes(G, eval_point_geom.x, eval_point_geom.y, return_dist=True)

        total_density = 0
        
        # 各イベントポイントからの影響を計算
        for _, event in event_points.iterrows():
            try:
                # 評価点とイベントポイント間のネットワーク距離を計算
                path_len = nx.shortest_path_length(G, source=eval_node, target=event['osm_node'], weight='length')
                
                # 距離がバンド幅内の場合のみ密度を計算
                if path_len <= bandwidth:
                    # Quarticカーネル関数
                    kernel_val = (1 - (path_len / bandwidth)**2)**2
                    total_density += kernel_val
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        # エッジの長さで密度を調整
        densities[(u, v, key)] = total_density / edge_len if edge_len > 0 else 0

    return densities

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
    # simplify=Trueにすることで、不要なノードを削減し、エッジにgeometry情報が付与されやすくなる
    G = ox.graph_from_xml(OSM_PATH, simplify=True)
    
    # 地点とOSMノードを紐付け
    event_coords = list(zip(event_points['latitude'], event_points['longitude']))
    event_osm_nodes = ox.nearest_nodes(G, [c[1] for c in event_coords], [c[0] for c in event_coords])
    event_points = event_points.copy()
    event_points.loc[:, 'osm_node'] = event_osm_nodes

    print("Network KDEの計算を開始します（時間がかかる場合があります）...")
    edge_densities = network_kernel_density(G, event_points, BANDWIDTH)
    
    # 密度スコアをPandas Seriesに変換
    density_series = pd.Series(edge_densities)
    
    print("KDE計算完了。結果を可視化中...")
    
    # 密度を正規化して色を決定
    if not density_series.empty and density_series.sum() > 0:
        norm_densities = MinMaxScaler().fit_transform(density_series.values.reshape(-1, 1)).flatten()
        cmap = plt.get_cmap('hot_r') # 'hot_r'は値が低いと黒、高いと赤・黄になる
        colors = [cmap(d) for d in norm_densities]
        edge_colors = {edge: color for edge, color in zip(density_series.index, colors)}
    else:
        print("密度が計算されなかったため、地図はグレースケールで表示されます。")
        edge_colors = {}

    # ベースマップを作成
    m = folium.Map(location=(clustered_df['latitude'].mean(), clustered_df['longitude'].mean()), 
                   zoom_start=14, tiles='cartodbpositron')

    # エッジを手動で描画
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_key = (u, v, key)
        
        # 色と太さを設定
        color_hex = '#555555' # デフォルト色
        weight = 1
        opacity = 0.5
        
        if edge_key in edge_colors:
            color_rgba = edge_colors[edge_key]
            color_hex = '#%02x%02x%02x' % (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
            weight = 4
            opacity = 0.8

        # ジオメトリを取得または作成
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
            opacity=opacity
        ).add_to(m)

    # 地図を保存
    map_output_path = os.path.join(OUTPUT_DIR, f'network_kde_cluster_{TARGET_CLUSTER_ID}_bw{BANDWIDTH}.html')
    m.save(map_output_path)

    print(f"Network KDEの可視化マップを保存しました: {map_output_path}")
    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()
