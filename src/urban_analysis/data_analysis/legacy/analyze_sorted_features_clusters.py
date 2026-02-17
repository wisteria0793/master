# -*- coding: utf-8 -*-
"""
ソート済み連結特徴量（Sorted View Concatenation）に基づく景観クラスタリングと地図化。
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

def get_coordinates_df(metadata_path):
    """pano_metadata.jsonから座標データを取得"""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        coords_data = [
            {
                "point_id": item["pano_id"],
                "latitude": item["api_location"][0],
                "longitude": item["api_location"][1]
            }
            for item in metadata if "api_location" in item and item["api_location"]
        ]
        return pd.DataFrame(coords_data)
    except Exception as e:
        print(f"メタデータの読み込みエラー: {e}")
        return None

def main():
    # --- 設定 ---
    N_CLUSTERS = 12  # クラスタ数
    # ---

    base_dir = '/Users/atsuyakatougi/Desktop/master'
    input_features_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'location_features_sorted.csv')
    pano_metadata_path = os.path.join(base_dir, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
    results_output_dir = os.path.join(base_dir, 'docs', 'results')
    os.makedirs(results_output_dir, exist_ok=True)

    # 1. データ読み込み
    print(f"特徴量を読み込み中: {input_features_path}...")
    df = pd.read_csv(input_features_path)
    
    # 特徴量列の選択（point_id と num_images 以外）
    feature_cols = [c for c in df.columns if c not in ['point_id', 'num_images']]
    X = df[feature_cols].values

    # 標準化（クラスタリングの精度向上のため推奨）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 階層的クラスタリング
    print(f"階層的クラスタリングを実行中 (Ward法, {N_CLUSTERS}クラスタ)...")
    linked = linkage(X_scaled, method='ward')

    # デンドログラム保存
    plt.figure(figsize=(15, 8))
    dendrogram(linked, truncate_mode='lastp', p=40, show_leaf_counts=True, no_labels=True)
    plt.title('Hierarchical Clustering Dendrogram (Sorted Features)')
    plt.savefig(os.path.join(results_output_dir, 'sorted_features_dendrogram.png'))
    plt.close()

    # クラスタ割り当て
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    df['cluster'] = clusters

    # 3. 特徴分析（プロファイル保存）
    # 元の比率に近い平均値を計算するため、標準化前のデータを使用
    cluster_profiles = df.groupby('cluster')[feature_cols].mean()
    cluster_profiles.to_csv(os.path.join(results_output_dir, 'sorted_features_cluster_profiles.csv'))

    # 4. 地図プロット
    print("地図を作成中...")
    coords_df = get_coordinates_df(pano_metadata_path)
    if coords_df is not None:
        merged_df = pd.merge(df, coords_df, on='point_id', how='inner')
        
        if not merged_df.empty:
            map_center = [merged_df['latitude'].mean(), merged_df['longitude'].mean()]
            m = folium.Map(location=map_center, zoom_start=14)
            
            # カラーパレット
            colors = sns.color_palette('hls', n_colors=N_CLUSTERS).as_hex()
            
            for _, row in merged_df.iterrows():
                c_idx = int(row['cluster']) - 1
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    color=colors[c_idx],
                    fill=True,
                    fill_color=colors[c_idx],
                    fill_opacity=0.7,
                    popup=f"ID: {row['point_id']}<br>Cluster: {row['cluster']}"
                ).add_to(m)
            
            map_path = os.path.join(results_output_dir, 'sorted_features_cluster_map.html')
            m.save(map_path)
            print(f"地図を保存しました: {map_path}")
    
    print("全工程完了。")

if __name__ == '__main__':
    main()
