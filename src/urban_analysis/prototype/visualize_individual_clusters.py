# -*- coding: utf-8 -*-
"""
単独クラスタリング結果の可視化スクリプト
統合前の「景観のみ」「機能のみ」の結果を地図上にプロットし、比較用マップを作成する。
"""

import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from pathlib import Path
import os

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
LS_CSV = PROJECT_ROOT / 'data' / 'processed' / 'refined_landscape_clusters_k12.csv'
POI_CSV = PROJECT_ROOT / 'data' / 'processed' / 'refined_poi_clusters_k8.csv'
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs' # 座標等の情報用
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results'

def create_map(df, k, title, output_path):
    print(f"地図生成中: {title}...")
    hakodate_center = [41.768, 140.729]
    m = folium.Map(location=hakodate_center, zoom_start=14, tiles='cartodbpositron')
    
    cmap = plt.cm.get_cmap('tab20', k)
    def get_hex_color(c):
        rgba = cmap(c)
        return '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    for _, row in df.iterrows():
        cluster_id = int(row['cluster'])
        color = get_hex_color(cluster_id)
        
        radius = 5 if 'name' in df.columns else 3
        
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=radius,
            popup=f"ID: {row.get('id', row.get('point_id'))}<br>Cluster: {cluster_id}",
            color='white',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=1.0
        ).add_to(m)
    
    m.save(str(output_path))
    print(f"保存完了: {output_path}")

def main():
    # 座標情報の読み込み (nodes_info.csv から取得するのが最も確実)
    nodes_info = pd.read_csv(INPUT_DIR / 'nodes_info.csv')
    
    # 1. 景観単独マップ
    df_ls = nodes_info[nodes_info['type'] == 'ls'].copy()
    create_map(df_ls, 12, "Landscape Only (K=12)", OUTPUT_DIR / 'individual_landscape_map_k12.html')
    
    # 2. 機能単独マップ
    df_poi = nodes_info[nodes_info['type'] == 'poi'].copy()
    create_map(df_poi, 8, "Function Only (K=8)", OUTPUT_DIR / 'individual_function_map_k8.html')

if __name__ == "__main__":
    main()
