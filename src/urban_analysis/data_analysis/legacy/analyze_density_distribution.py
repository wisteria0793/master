# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import osmnx as ox
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 設定 (network_kde_optimized.py と合わせる) ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
N_CLUSTERS = 19
EMBEDDING_DIM = 64
TARGET_CLUSTER_ID = 9
BANDWIDTH = 300

EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', 'embeddings_dim64_streetclip_mean.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OSM_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'osm_hakodate', 'Hakodate.osm.xml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'docs', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_clustered_data():
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
    node_densities = pd.Series(0.0, index=list(G.nodes()))
    print("各イベントポイントからの影響を計算中...")
    for _, event in tqdm(event_points.iterrows(), total=len(event_points), desc="KDE Calculation"):
        event_node = event['osm_node']
        if not G.has_node(event_node): continue
        reachable_nodes_dist = nx.single_source_dijkstra_path_length(G, source=event_node, cutoff=bandwidth, weight='length')
        for node, dist in reachable_nodes_dist.items():
            kernel_val = (1 - (dist / bandwidth)**2)**2
            node_densities[node] += kernel_val
    print("エッジ密度を計算中...")
    edge_densities = {}
    for u, v, data in G.edges(data=True):
        edge_len = data.get('length', 1)
        avg_node_density = (node_densities.get(u, 0) + node_densities.get(v, 0)) / 2
        edge_densities[(u, v, 0)] = avg_node_density / edge_len if edge_len > 0 else 0
    return edge_densities

def main():
    clustered_df = get_clustered_data()
    event_points = clustered_df[clustered_df['cluster'] == TARGET_CLUSTER_ID]
    print(f"対象クラスタの地点数: {len(event_points)}")
    
    print("OSM道路網を読み込み中...")
    G = ox.graph_from_xml(OSM_PATH, simplify=True)
    event_coords = list(zip(event_points['latitude'], event_points['longitude']))
    event_osm_nodes = ox.nearest_nodes(G, [c[1] for c in event_coords], [c[0] for c in event_coords])
    event_points = event_points.copy()
    event_points.loc[:, 'osm_node'] = event_osm_nodes

    print("Network KDEの計算を開始します...")
    edge_densities = optimized_network_kernel_density(G, event_points, BANDWIDTH)
    density_values = np.array(list(edge_densities.values()))
    
    # 0以外の密度値を抽出
    non_zero_densities = density_values[density_values > 0]
    
    print(f"\n--- 密度統計 (全エッジ数: {len(density_values)}) ---")
    print(f"密度 > 0 のエッジ数: {len(non_zero_densities)} ({len(non_zero_densities)/len(density_values)*100:.2f}%)")
    if len(non_zero_densities) > 0:
        print(f"最小値: {non_zero_densities.min():.6f}")
        print(f"最大値: {non_zero_densities.max():.6f}")
        print(f"平均値: {non_zero_densities.mean():.6f}")
        print(f"中央値: {np.median(non_zero_densities):.6f}")
        for p in [50, 75, 90, 95, 99]:
            print(f"{p}パーセンタイル: {np.percentile(non_zero_densities, p):.6f}")

    # 可視化
    if len(non_zero_densities) > 0:
        plt.figure(figsize=(12, 6))
        
        # ヒストグラム
        plt.subplot(1, 2, 1)
        sns.histplot(non_zero_densities, kde=True, log_scale=True)
        plt.title(f'Density Distribution (Log Scale)\nCluster {TARGET_CLUSTER_ID}, BW {BANDWIDTH}')
        plt.xlabel('Density (log scale)')
        plt.ylabel('Frequency')

        # 累積分布関数 (CDF)
        plt.subplot(1, 2, 2)
        sns.ecdfplot(non_zero_densities)
        plt.title(f'Cumulative Distribution (CDF)\nCluster {TARGET_CLUSTER_ID}, BW {BANDWIDTH}')
        plt.xlabel('Density')
        plt.ylabel('Proportion')
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f'density_distribution_cluster{TARGET_CLUSTER_ID}_bw{BANDWIDTH}.png')
        plt.savefig(plot_path)
        print(f"\n分布図を保存しました: {plot_path}")
    else:
        print("有意な密度値が計算されなかったため、可視化をスキップします。")

if __name__ == '__main__':
    main()
