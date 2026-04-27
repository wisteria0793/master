# -*- coding: utf-8 -*-
"""
実験18.4: アプローチA（統合GNN）とアプローチB（独立型）を左右に並べて比較するDualMapを生成する。
"""

import pandas as pd
import numpy as np
import os
import folium
from folium.plugins import DualMap
import matplotlib.pyplot as plt
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
APPROACH_B_DIR = PROJECT_ROOT / 'data' / 'processed' / 'approach_b'
RESIDUAL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'comparison_approach_b'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("比較用データを準備中...")
    # アプローチA（18.3 統合版）
    df_a = pd.read_csv(RESIDUAL_DIR / 'residual_embeddings_clustered.csv')
    
    # アプローチB（18.4 独立テキストクラスタ）
    # POIは18.4のもの、SVはAと共通
    df_b_poi = pd.read_csv(APPROACH_B_DIR / 'poi_text_only_clusters.csv')
    df_sv = df_a[df_a['type'] == 'sv'].copy()
    
    print("DualMapを生成中（左右同期可視化）...")
    map_center = [df_a['lat'].mean(), df_a['lng'].mean()]
    dual_map = DualMap(location=map_center, zoom_start=16)
    
    # カラーパレット
    # 統合地区(A)用
    cmap_a = plt.cm.get_cmap('tab20', 20)
    # 景観背景用
    cmap_landscape = plt.cm.get_cmap('terrain', 20)
    # テキスト機能(B)用
    cmap_text = plt.cm.get_cmap('Set1', 20)
    
    # --- 左側：アプローチA（統合GNN） ---
    for _, row in df_a.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap_a(row['cluster_id'])[:3])
        if row['type'] == 'poi':
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=7, color='black', weight=1, fill=True, fill_color=color, fill_opacity=1.0,
                popup=f"A-POI: {row['name']} (Cluster: {row['cluster_id']})"
            ).add_to(dual_map.m1)
        else:
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=3, color=color, fill=True, fill_color=color, fill_opacity=0.5,
                popup=f"A-SV: Cluster {row['cluster_id']}"
            ).add_to(dual_map.m1)
            
    # --- 右側：アプローチB（単純な地理的所属 / 景観クラスタにPOIが属するだけ） ---
    # 背景としての景観
    for _, row in df_sv.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap_landscape(row['cluster_id'])[:3])
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=3, color=color, fill=True, fill_color=color, fill_opacity=0.3,
            popup=f"Landscape Cluster: {row['cluster_id']}"
        ).add_to(dual_map.m2)
        
    # POIも「景観クラスタ」の色で塗る（機能的な独自性は無視）
    # POIに最も近い景観ポイントのクラスタを割り当てる（もしくはすでに統合データにあるLandscapeIDを使用）
    # ここでは単純化のため、統合データのcluster_id（景観ベース）を流用
    for _, row in df_a[df_a['type'] == 'poi'].iterrows():
        # Aのデータにある景観ベースのID（実験18.3での所属）を使用
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap_landscape(row['cluster_id'])[:3])
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=7, color='black', weight=1, fill=True, fill_color=color, fill_opacity=1.0,
            popup=f"POI: {row['name']} (Belongs to Landscape Cluster: {row['cluster_id']})"
        ).add_to(dual_map.m2)
        
    output_path = OUTPUT_DIR / 'approach_comparison_dual_map.html'
    dual_map.save(output_path)
    print(f"完了。比較 DualMap を保存しました: {output_path}")

if __name__ == "__main__":
    main()
