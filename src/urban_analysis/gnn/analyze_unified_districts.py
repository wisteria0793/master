# -*- coding: utf-8 -*-
"""
実験18: 統一埋め込みベクトル（景観+機能）をクラスタリングし、都市の「統合地区」として可視化する。
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
GNN_UNIFIED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'unified_gnn'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 20

def main():
    print("統一エンベディングを読み込み中...")
    df = pd.read_csv(GNN_UNIFIED_DIR / 'unified_embeddings.csv')
    
    # クラスタリング用の特徴量抽出 (dim_0 ~ dim_63)
    feature_cols = [c for c in df.columns if c.startswith('dim_')]
    features = df[feature_cols].values
    
    # 階層的クラスタリング (Ward法)
    print(f"{N_CLUSTERS} クラスタに分割中...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    df['cluster_id'] = clusters - 1
    
    # クラスタリング結果を保存
    clustered_csv = GNN_UNIFIED_DIR / 'unified_embeddings_clustered.csv'
    df.to_csv(clustered_csv, index=False)
    print(f"クラスタリング結果を保存しました: {clustered_csv}")
    
    # クラスタ統計
    print("\n【クラスタ組成分析】")
    summary = df.groupby(['cluster_id', 'type']).size().unstack(fill_value=0)
    print(summary)
    
    # 地図の生成
    print("\n地図を生成中...")
    map_center = [df['lat'].mean(), df['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=14)
    
    # カラーマップ設定
    cmap = plt.cm.get_cmap('tab20', N_CLUSTERS)
    
    for _, row in df.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap(row['cluster_id'])[:3])
        
        if row['type'] == 'poi':
            # POIはマーカーで表示
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=7,
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                popup=f"POI: {row['name']} (Cluster: {row['cluster_id']})"
            ).add_to(m)
        else:
            # 道路ポイント(SV)は小さな円で表示
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"SV: {row['id']} (Cluster: {row['cluster_id']})"
            ).add_to(m)
            
    map_path = OUTPUT_DIR / f'unified_district_map_k{N_CLUSTERS}.html'
    m.save(map_path)
    print(f"\n完了。統合地区マップを保存しました: {map_path}")
    
    # クラスタごとの代表的な名前（POI）を表示
    print("\n【各クラスタの傾向（含まれる主なPOI）】")
    for cid in range(N_CLUSTERS):
        cluster_pois = df[(df['cluster_id'] == cid) & (df['type'] == 'poi')]['name'].tolist()
        poi_str = ", ".join(cluster_pois[:5]) + ("..." if len(cluster_pois) > 5 else "")
        sv_count = len(df[(df['cluster_id'] == cid) & (df['type'] == 'sv')])
        print(f"Cluster {cid:02d}: SV={sv_count:4d} nodes, POIs: {poi_str}")

if __name__ == "__main__":
    main()
