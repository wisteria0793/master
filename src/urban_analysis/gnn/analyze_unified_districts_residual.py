# -*- coding: utf-8 -*-
"""
実験18.3: 景観保存型（Residual）統合埋め込みをクラスタリングし、可視化する。
"""

import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
RESIDUAL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'unified_gnn_residual'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 20

def main():
    print("景観保存型（Residual）埋め込みを読み込み中...")
    df = pd.read_csv(RESIDUAL_DIR / 'residual_embeddings.csv')
    
    feature_cols = [c for c in df.columns if c.startswith('dim_')]
    features = df[feature_cols].values
    
    print(f"高解像度GNNデータに基づき {N_CLUSTERS} クラスタに分割中...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    df['cluster_id'] = clusters - 1
    
    # 統計
    print("\n【景観保存型 - クラスタ組成分析】")
    summary = df.groupby(['cluster_id', 'type']).size().unstack(fill_value=0)
    print(summary)
    
    # クラスタごとに最大のクラスタをチェック
    max_cluster_size = summary.sum(axis=1).max()
    print(f"\n最大クラスタサイズ: {max_cluster_size} (全 {len(df)} 地点中)")
    
    # 地図の生成
    print("\n地図（Residual GAT版）を生成中...")
    map_center = [df['lat'].mean(), df['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=15)
    
    cmap = plt.cm.get_cmap('tab20', N_CLUSTERS)
    
    for _, row in df.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap(row['cluster_id'])[:3])
        if row['type'] == 'poi':
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=8,
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                popup=f"POI: {row['name']} (Cluster: {row['cluster_id']})"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"SV: {row['id']} (Cluster: {row['cluster_id']})"
            ).add_to(m)
            
    map_path = OUTPUT_DIR / f'unified_district_map_residual_k{N_CLUSTERS}.html'
    m.save(map_path)
    print(f"\n完了。景観保存型マップを保存しました: {map_path}")
    
    # 成果物の保存
    df.to_csv(RESIDUAL_DIR / 'residual_embeddings_clustered.csv', index=False)

    print("\n【各クラスタの代表的なPOI】")
    for cid in range(min(N_CLUSTERS, len(summary))):
        cluster_pois = df[(df['cluster_id'] == cid) & (df['type'] == 'poi')]['name'].tolist()
        poi_str = ", ".join(cluster_pois[:8]) + ("..." if len(cluster_pois) > 8 else "")
        print(f"Cluster {cid:02d}: POIs: {poi_str}")

if __name__ == "__main__":
    main()
