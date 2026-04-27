# -*- coding: utf-8 -*-
"""
段階的統合（後期統合）検証スクリプト
個別GNNで平滑化されたベクトルを後から結合し、同時統合（現在のGNN）との結果の違いを確認する。
"""

import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_INDIV_RES = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results_individual'
GNN_INDIV_IN = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs_individual'
OUTPUT_MAP = PROJECT_ROOT / 'docs' / 'results' / 'late_integration_urban_districts_map.html'

def main():
    print("個別GNNの結果をロード中心...")
    
    # 景観データ (3501 nodes)
    z_ls = np.load(GNN_INDIV_RES / 'landscape' / 'embeddings.npy')
    df_ls = pd.read_csv(GNN_INDIV_IN / 'landscape' / 'nodes.csv')
    df_ls['emb_idx'] = range(len(df_ls))
    
    # 機能データ (374 nodes)
    z_poi = np.load(GNN_INDIV_RES / 'function' / 'embeddings.npy')
    df_poi = pd.read_csv(GNN_INDIV_IN / 'function' / 'nodes.csv')
    df_poi['emb_idx'] = range(len(df_poi))
    
    # 空間連結のためのKDTree
    tree_ls = KDTree(df_ls[['lat', 'lng']].values * np.array([111000, 82000]))
    tree_poi = KDTree(df_poi[['lat', 'lng']].values * np.array([111000, 82000]))
    
    print("空間情報の相互補完（段階的結合）を実行中...")
    
    # POIノードに対する景観情報の付与 (16 + 16 = 32dims)
    poi_combined_features = []
    for i, row in df_poi.iterrows():
        # 自地点の平滑化機能ベクトル
        self_feat = z_poi[i]
        # 周囲150mの景観ベクトルの平均
        indices = tree_ls.query_ball_point([row['lat'] * 111000, row['lng'] * 82000], r=150)
        if indices:
            neighbor_ls_feat = np.mean(z_ls[indices], axis=0)
        else:
            neighbor_ls_feat = np.zeros(16)
        poi_combined_features.append(np.concatenate([self_feat, neighbor_ls_feat]))
    
    # 景観ノードに対する機能情報の付与 (16 + 16 = 32dims)
    ls_combined_features = []
    for i, row in df_ls.iterrows():
        # 自地点の平滑化景観ベクトル
        self_feat = z_ls[i]
        # 周囲150mの機能ベクトルの平均
        indices = tree_poi.query_ball_point([row['lat'] * 111000, row['lng'] * 82000], r=150)
        if indices:
            neighbor_poi_feat = np.mean(z_poi[indices], axis=0)
        else:
            neighbor_poi_feat = np.zeros(16)
        ls_combined_features.append(np.concatenate([neighbor_poi_feat, self_feat]))

    # 全地点の統合
    all_features = np.vstack([poi_combined_features, ls_combined_features])
    all_nodes = pd.concat([df_poi, df_ls]).reset_index(drop=True)
    
    print(f"最終クラスタリング中 (K=12, Features=32dims)...")
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_features)
    all_nodes['late_cluster'] = labels
    
    # 地図生成
    print("段階的統合マップを生成中...")
    m = folium.Map(location=[41.768, 140.729], zoom_start=14, tiles='cartodbpositron')
    cmap = plt.cm.get_cmap('tab20', 12)
    
    for _, row in all_nodes.iterrows():
        color = '#%02x%02x%02x' % tuple((np.array(cmap(int(row['late_cluster']))[:3])*255).astype(int))
        radius = 7 if row['type'] == 'poi' else 3
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=radius,
            color='white',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=1.0
        ).add_to(m)
        
    m.save(str(OUTPUT_MAP))
    print(f"保存完了: {OUTPUT_MAP}")

if __name__ == "__main__":
    main()
