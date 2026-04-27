# -*- coding: utf-8 -*-
"""
実験18.5: アプローチB（純粋景観モデル）の単独マップ生成。
GNNなしの景観クラスタに基づき、各POIを最寄りのクラスタに割り当てた「純粋な見た目ベース」の地図を作成する。
"""

import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
APPROACH_B_DIR = PROJECT_ROOT / 'data' / 'processed' / 'approach_b'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'comparison_approach_b'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 20

def main():
    print("純粋景観データを読み込み中...")
    df_sv = pd.read_csv(APPROACH_B_DIR / 'sv_landscape_only_clusters.csv')
    
    # POI情報の取得（位置情報のため）
    # nodes_metadata.json からPOIのみ抽出
    import json
    with open(PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered' / 'nodes_metadata.json', 'r') as f:
        nodes_meta = json.load(f)
    poi_data = [n for n in nodes_meta if n['type'] == 'poi']
    df_poi = pd.DataFrame(poi_data)
    
    print("POIを最寄りの景観クラスタへ割り当て中 (Spatial Join)...")
    sv_coords = df_sv[['lat', 'lng']].values
    poi_coords = df_poi[['lat', 'lng']].values
    
    # KDTreeで最寄り検索
    # 簡易メートル変換
    sv_m = sv_coords * np.array([111000, 82000])
    poi_m = poi_coords * np.array([111000, 82000])
    
    tree = KDTree(sv_m)
    _, nearest_indices = tree.query(poi_m)
    
    # 最寄りのSVのクラスタIDを引き継ぐ
    df_poi['landscape_cluster_id'] = df_sv.iloc[nearest_indices]['landscape_cluster_id'].values
    
    print("アプローチB（純粋景観ベース）の単独マップを生成中...")
    map_center = [df_sv['lat'].mean(), df_sv['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=15)
    
    cmap = plt.cm.get_cmap('terrain', N_CLUSTERS)
    
    # 景観ポイントをプロット
    for _, row in df_sv.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap(row['landscape_cluster_id'])[:3])
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.3,
            popup=f"Landscape Cluster: {row['landscape_cluster_id']}"
        ).add_to(m)
        
    # POIのプロットを削除（ご指示により、景観のみを表示）
    
    map_path = OUTPUT_DIR / 'landscape_only_baseline_map.html'
    m.save(map_path)
    print(f"\n完了。純粋景観ベース地図を保存しました: {map_path}")
    
    # 比較用データとして保存
    df_poi.to_csv(APPROACH_B_DIR / 'poi_landscape_assignment.csv', index=False)

if __name__ == "__main__":
    main()
