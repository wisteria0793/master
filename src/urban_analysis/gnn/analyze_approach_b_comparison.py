# -*- coding: utf-8 -*-
"""
実験18.4: アプローチB（独立型）の可視化。
景観クラスタ（面）の上に、機能クラスタ（点）を重ね合わせ、アプローチA（統合型）との違いを視覚化する。
"""

import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
APPROACH_B_DIR = PROJECT_ROOT / 'data' / 'processed' / 'approach_b'
RESIDUAL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'comparison_approach_b'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS_FUNCTION = 20
N_CLUSTERS_LANDSCAPE = 20

def main():
    print("データを読み込み中...")
    # 機能クラスタ（アプローチB）
    df_poi = pd.read_csv(APPROACH_B_DIR / 'poi_text_only_clusters.csv')
    # 景観クラスタ（実験18.3のSVデータを利用）
    df_unified = pd.read_csv(RESIDUAL_DIR / 'residual_embeddings_clustered.csv')
    df_sv = df_unified[df_unified['type'] == 'sv'].copy()
    
    print("アプローチB（単純重ね合わせ）のマップを生成中...")
    map_center = [df_poi['lat'].mean(), df_poi['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=15)
    
    # カラーパレット（景観用と機能用で分ける）
    cmap_landscape = plt.cm.get_cmap('terrain', N_CLUSTERS_LANDSCAPE)
    cmap_function = plt.cm.get_cmap('Set1', N_CLUSTERS_FUNCTION) # 機能ははっきりした色
    
    # 1. 景観（背景）をプロット
    for _, row in df_sv.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap_landscape(row['cluster_id'])[:3])
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,
            popup=f"Landscape Cluster: {row['cluster_id']}"
        ).add_to(m)
        
    # 2. 機能（POI）をプロット - 境界線を黒にして強調
    for _, row in df_poi.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap_function(row['text_cluster_id'])[:3])
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=8,
            color='black',
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=f"POI: {row['name']} (Function: {row['text_cluster_id']})"
        ).add_to(m)
        
    map_path = OUTPUT_DIR / 'approach_b_overlay_map.html'
    m.save(map_path)
    print(f"完了。比較用マップを保存しました: {map_path}")
    
    # 簡易レポート生成
    print("\n【アプローチBの構造的特徴】")
    print("- 景観（背景）: GNNに基づく空間的連続性のある面。")
    print("- 機能（点）: 景観とは無関係に、説明文の類似性だけで決まった『点』。")
    print("※ これを統合型（18.3）と比較することで、『景観と機能の共生関係』を導き出した価値が明確になります。")

if __name__ == "__main__":
    main()
