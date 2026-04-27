# -*- coding: utf-8 -*-
"""
単体GNN結果の可視化スクリプト
GNNで空間的平滑化を行った「景観」と「機能」を個別にマップ化する。
"""

import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_RES_BASE = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results_individual'
GNN_IN_BASE = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs_individual'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results'

def create_smoothed_map(name, k, title, output_path):
    print(f"[{name}] 平滑化マップを生成中...")
    embeddings = np.load(GNN_RES_BASE / name / 'embeddings.npy')
    df_nodes = pd.read_csv(GNN_IN_BASE / name / 'nodes.csv')
    
    # GNN埋め込みに対する最終クラスタリング
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    df_nodes['smoothed_cluster'] = labels
    
    hakodate_center = [41.768, 140.729]
    m = folium.Map(location=hakodate_center, zoom_start=14, tiles='cartodbpositron')
    
    cmap = plt.cm.get_cmap('tab20', k)
    def get_hex_color(c):
        rgba = cmap(c)
        return '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    
    for _, row in df_nodes.iterrows():
        cluster_id = int(row['smoothed_cluster'])
        color = get_hex_color(cluster_id)
        
        radius = 7 if row['type'] == 'poi' else 3
        
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=radius,
            popup=f"Smoothed Cluster: {cluster_id}",
            color='white',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=1.0
        ).add_to(m)
        
    m.save(str(output_path))
    print(f"保存完了: {output_path}")

def main():
    # 1. 景観平滑化マップ (K=12)
    create_smoothed_map('landscape', 12, "Landscape GNN Smoothed", OUTPUT_DIR / 'smoothed_landscape_map_k12.html')
    
    # 2. 機能平滑化マップ (K=8)
    create_smoothed_map('function', 8, "Function GNN Smoothed", OUTPUT_DIR / 'smoothed_function_map_k8.html')

if __name__ == "__main__":
    main()
