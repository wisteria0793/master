# -*- coding: utf-8 -*-
"""
GNN+Combinedエンベディング（StreetCLIP + Seg-GNN）を可視化するスクリプト。
"""

import pandas as pd
import numpy as np
import os
import json
import folium
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
N_CLUSTERS = 20
EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', 'embeddings_dim64_combined_mean.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'docs', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("データを読み込み中...")
    try:
        embedding_df = pd.read_csv(EMBEDDING_PATH)
    except FileNotFoundError:
        print(f"エラー: {EMBEDDING_PATH}")
        return None
        
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item and item["api_location"]}
    
    embedding_df['latitude'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    embedding_df['longitude'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    embedding_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    return embedding_df

def main():
    df = load_data()
    if df is None: return

    print("Combinedエンベディングの階層的クラスタリングを実行中...")
    
    feature_cols = df.columns.drop(['point_id', 'latitude', 'longitude'])
    features = df[feature_cols].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    linked = linkage(features_scaled, method='ward')

    plt.figure(figsize=(20, 10))
    dendrogram(
        linked,
        orientation='top',
        labels=df['point_id'].tolist(),
        distance_sort='descending',
        show_leaf_counts=True,
        truncate_mode='lastp',
        p=100
    )
    plt.title(f'Hierarchical Clustering (Combined: StreetCLIP + Seg-GNN, {N_CLUSTERS} Clusters)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'combined_gnn_dendrogram_{N_CLUSTERS}.png'))
    plt.close()

    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    df['cluster'] = clusters - 1
    
    print(df['cluster'].value_counts().sort_index())

    print("地図を生成中...")
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=14)
    colors = plt.cm.get_cmap('tab20', N_CLUSTERS)

    for idx, row in df.iterrows():
        color_hex = '#%02x%02x%02x' % tuple(int(x*255) for x in colors(row['cluster'])[:3])
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.8,
            tooltip=f"Cluster: {row['cluster']}"
        ).add_to(m)

    m.save(os.path.join(OUTPUT_DIR, f'combined_gnn_cluster_map_{N_CLUSTERS}.html'))
    print("完了。")

if __name__ == '__main__':
    main()
