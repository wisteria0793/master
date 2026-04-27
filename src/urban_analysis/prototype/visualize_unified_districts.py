# -*- coding: utf-8 -*-
"""
最終統合地区可視化スクリプト
GNNで生成された統合埋め込みベクトルをもとに、
最終的な都市地区をクラスタリング・可視化する。
"""

import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from pathlib import Path
import os

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_RESULTS_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results'
GNN_INPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs'
OUTPUT_MAP = PROJECT_ROOT / 'docs' / 'results' / 'unified_urban_districts_map.html'

def main():
    print("データをロード中...")
    embeddings = np.load(GNN_RESULTS_DIR / 'unified_district_embeddings.npy')
    nodes_info = pd.read_csv(GNN_INPUT_DIR / 'nodes_info.csv')
    
    # 1. 最終クラスタリング (統合地区の抽出)
    # 統合された16次元ベクトルを、最終的に12の地区に分類
    K = 12
    print(f"統合地区のクラスタリングを実行中 (K={K})...")
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    unified_labels = kmeans.fit_predict(embeddings)
    nodes_info['unified_cluster'] = unified_labels
    
    # 2. 地図の生成
    hakodate_center = [41.768, 140.729]
    m = folium.Map(location=hakodate_center, zoom_start=14, tiles='cartodbpositron')
    
    # カラーパレットの生成（tab20 を使用して、既存の可視化と統一）
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab20', K)
    def get_hex_color(k):
        rgba = cmap(k)
        return '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    # 3. 景観情報が近隣に存在しないPOIの除外（フィルタリング）
    # prepare_gnn_graph_data で作成した edge_index を使用して近接ノードを確認
    edge_index = np.load(GNN_INPUT_DIR / 'edge_index.npy')
    from collections import defaultdict
    neighbors = defaultdict(list)
    for u, v in edge_index.T:
        neighbors[u].append(v)
        
    print("孤立したPOIを特定し、フィルタリング中...")
    filtered_nodes_idx = []
    for i, row in nodes_info.iterrows():
        if row['type'] == 'poi':
            neighbor_types = [nodes_info.iloc[nb]['type'] for nb in neighbors[i]]
            if 'ls' in neighbor_types:
                filtered_nodes_idx.append(i)
        else:
            filtered_nodes_idx.append(i)
            
    df_plot = nodes_info.loc[filtered_nodes_idx].copy()
    
    print("地図上にプロット中...")
    # 景観地点を先に描画し、POIを上に重ねる（視認性向上）
    for type_name in ['ls', 'poi']:
        df_subset = df_plot[df_plot['type'] == type_name]
        for _, row in df_subset.iterrows():
            cluster_id = int(row['unified_cluster'])
            color_hex = get_hex_color(cluster_id)
            
            radius = 7 if row['type'] == 'poi' else 3
            
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=radius,
                popup=f"ID: {row['id']}",
                color='white',
                weight=1,
                fill=True,
                fill_color=color_hex,
                fill_opacity=1.0 # 完全に不透明で視認性重視
            ).add_to(m)
        
    # 保存
    os.makedirs(OUTPUT_MAP.parent, exist_ok=True)
    m.save(str(OUTPUT_MAP))
    
    # クラスタごとの統計（フィルタリング後のデータを使用）
    stats = df_plot.groupby(['unified_cluster', 'type']).size().unstack(fill_value=0)
    print("\n--- 統合地区ごとの構成統計 (景観なしPOI除外後) ---")
    print(stats)
    
    print(f"\n統合地区マップを保存しました: {OUTPUT_MAP}")

if __name__ == "__main__":
    main()
