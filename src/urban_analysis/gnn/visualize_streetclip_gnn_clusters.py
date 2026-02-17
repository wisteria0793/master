# -*- coding: utf-8 -*-
"""
GNN+StreetCLIPエンベディング（平均済み）を可視化するスクリプト。
"""

import pandas as pd
import numpy as np
import os
import json
import folium
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
N_CLUSTERS = 19

# --- パス設定 ---
# 集約済み（平均）エンベディングを使用
EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', 'embeddings_dim64_streetclip_mean.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'docs', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """GNNエンベディングと座標データを読み込んでマージする"""
    print("データを読み込み中...")
    try:
        embedding_df = pd.read_csv(EMBEDDING_PATH)
    except FileNotFoundError:
        print(f"エラー: エンベディングファイルが見つかりません: {EMBEDDING_PATH}")
        return None
        
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item and item["api_location"]}
    
    embedding_df['latitude'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    embedding_df['longitude'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    embedding_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    print(f"{len(embedding_df)} 地点分のエンベディングと座標を読み込みました。")
    return embedding_df

def main():
    """メインの処理フロー"""
    df = load_data()
    if df is None:
        return

    # --- 1. 階層的クラスタリング ---
    print("GNN+StreetCLIPエンベディングの階層的クラスタリングを実行中...")
    
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
    plt.title(f'Hierarchical Clustering (StreetCLIP + GNN, {N_CLUSTERS} Clusters)')
    plt.xlabel('Point ID')
    plt.ylabel('Distance (Ward)')
    plt.tight_layout()
    dendrogram_output_path = os.path.join(OUTPUT_DIR, f'streetclip_gnn_dendrogram_{N_CLUSTERS}.png')
    plt.savefig(dendrogram_output_path)
    print(f"デンドログラムを保存しました: {dendrogram_output_path}")
    plt.close()

    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    df['cluster'] = clusters - 1
    
    print("クラスタリング結果:")
    print(df['cluster'].value_counts().sort_index())

    # --- 2. 地図上への可視化 ---
    print("インタラクティブな地図を生成中...")
    
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=14)

    colors = plt.cm.get_cmap('tab20', N_CLUSTERS)

    for idx, row in df.iterrows():
        cluster_id = row['cluster']
        color_rgba = colors(cluster_id)
        color_hex = '#%02x%02x%02x' % (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
            
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.8,
            tooltip=f"Point ID: {row['point_id']}<br>Cluster: {cluster_id}"
        ).add_to(m)

    map_output_path = os.path.join(OUTPUT_DIR, f'streetclip_gnn_cluster_map_{N_CLUSTERS}.html')
    m.save(map_output_path)
    print(f"クラスタ地図を保存しました: {map_output_path}")
    
    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()
