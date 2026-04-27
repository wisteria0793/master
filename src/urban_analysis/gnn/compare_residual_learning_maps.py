# -*- coding: utf-8 -*-
"""
実験18.11: Proposed (Pre-GNN) と Baseline (Raw-Clustered) の統合学習結果を地図で比較する。
"""

import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import folium
import matplotlib.pyplot as plt
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
BASE_DIR = PROJECT_ROOT / 'data' / 'processed'
BASELINE_DIR = BASE_DIR / 'gnn_unified_residual_baseline'
PROPOSED_DIR = BASE_DIR / 'gnn_unified_residual_proposed'
OUTPUT_DIR = PROJECT_ROOT / 'docs/results/residual_learning_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 20

def generate_comparison_map(csv_path, output_name, title):
    print(f"地図生成中: {title}...")
    df = pd.read_csv(csv_path)
    
    # 埋め込み次元（dim_0〜dim_63）を抽出してクラスタリング
    feat_cols = [f'dim_{i}' for i in range(64)]
    z = df[feat_cols].values
    
    linked = linkage(z, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust') - 1
    df['cluster_id'] = clusters
    
    # 地図プロット
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=15)
    cmap = plt.get_cmap('tab20', N_CLUSTERS)
    
    for _, row in df.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap(row['cluster_id'])[:3])
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=8 if row['type'] == 'poi' else 4,
            color='black' if row['type'] == 'poi' else color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=1.0 if row['type'] == 'poi' else 0.5,
            popup=f"{row['type'].upper()}: {row['name']} (C: {row['cluster_id']})"
        ).add_to(m)
    
    m.save(OUTPUT_DIR / output_name)
    print(f"保存完了: {output_name}")

def main():
    # Baseline の地図化
    generate_comparison_map(
        BASELINE_DIR / 'residual_embeddings_baseline.csv',
        'map_18_3_baseline_raw_seeds.html',
        "Baseline (Raw Text Clusters)"
    )
    
    # Proposed の地図化
    generate_comparison_map(
        PROPOSED_DIR / 'residual_embeddings_proposed.csv',
        'map_18_3_proposed_pre_gnn.html',
        "Proposed (Pre-smoothed GNN Embeddings)"
    )
    
    print(f"\n比較地図が正常に出力されました: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
