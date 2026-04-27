# -*- coding: utf-8 -*-
"""
実験18.6: アプローチC - テキスト情報のみを用いたGNN埋め込みの分析と可視化。
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
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'text_only_gnn'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'text_only_gnn'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 20

def main():
    print("テキストGNN埋め込みを読み込み中...")
    df = pd.read_csv(INPUT_DIR / 'text_only_gnn_embeddings.csv')
    
    feature_cols = [c for c in df.columns if c.startswith('dim_')]
    features = df[feature_cols].values
    
    print(f"テキストGNNデータに基づき {N_CLUSTERS} クラスタに分割中...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    df['cluster_id'] = clusters - 1
    
    # 統計
    print("\n【テキストGNN - クラスタ分布分析】")
    cluster_counts = df.groupby('cluster_id').size()
    print(cluster_counts)
    
    # 最大クラスタサイズの確認
    max_cluster_size = cluster_counts.max()
    print(f"\n最大クラスタサイズ: {max_cluster_size} (全 {len(df)} 地点中)")
    
    # 地図の生成
    print("\n地図（テキストGNN版 / アプローチC）を生成中...")
    map_center = [df['lat'].mean(), df['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=15)
    
    cmap = plt.cm.get_cmap('tab20', N_CLUSTERS)
    
    for _, row in df.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap(row['cluster_id'])[:3])
        if row['type'] == 'poi':
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=8, color='black', weight=1, fill=True, fill_color=color, fill_opacity=1.0,
                popup=f"A-POI: {row['name']} (Cluster: {row['cluster_id']})"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.5,
                popup=f"A-SV: Cluster {row['cluster_id']}"
            ).add_to(m)
            
    map_path = OUTPUT_DIR / 'text_only_gnn_map.html'
    m.save(map_path)
    print(f"\n完了。テキストGNNマップを保存しました: {map_path}")
    
    # 成果物の保存
    df.to_csv(INPUT_DIR / 'text_only_gnn_embeddings_clustered.csv', index=False)

    print("\n【上位クラスタの代表POI】")
    for cid in range(min(10, len(cluster_counts))):
        cluster_pois = df[(df['cluster_id'] == cid) & (df['type'] == 'poi')]['name'].tolist()
        poi_str = ", ".join(cluster_pois[:5]) + ("..." if len(cluster_pois) > 5 else "")
        print(f"Cluster {cid:02d}: {poi_str}")

if __name__ == "__main__":
    main()
