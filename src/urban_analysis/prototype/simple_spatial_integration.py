# -*- coding: utf-8 -*-
"""
フェーズ1：シンプル統合スクリプト
POI地点に対して、近傍の景観クラスター情報を紐付ける。
"""

import os
import json
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
POI_CLUSTER_FILE = PROJECT_ROOT / 'data' / 'processed' / 'gnn_embeddings' / 'poi_gnn_clusters.csv'
LANDSCAPE_CLUSTER_FILE = PROJECT_ROOT / 'data' / 'processed' / 'segmentation_results_50m' / 'jsd_cluster_assignments_20.csv'
METADATA_FILE = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered' / 'nodes_metadata.json'
OUTPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'integrated_poi_landscape_clusters.csv'

def main():
    print("データを読み込み中...")
    
    # 1. クラスターデータの読み込み
    if not POI_CLUSTER_FILE.exists():
        print(f"Error: {POI_CLUSTER_FILE} not found.")
        return
    df_poi = pd.read_csv(POI_CLUSTER_FILE)
    
    if not LANDSCAPE_CLUSTER_FILE.exists():
        print(f"Error: {LANDSCAPE_CLUSTER_FILE} not found.")
        return
    df_ls = pd.read_csv(LANDSCAPE_CLUSTER_FILE)
    
    # 2. メタデータ（座標情報）の読み込み
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        nodes_meta = json.load(f)
    
    # メタデータから座標マッピングを作成
    # SVノードの ID は metadata では "-0-hL..."
    # 景観クラスターCSVの location_id は "pano_-0-hL..."
    coord_map = {n['id']: (n['lat'], n['lng']) for n in nodes_meta}
    
    print("景観地点の座標を紐付け中...")
    ls_coords = []
    ls_clusters = []
    for _, row in df_ls.iterrows():
        loc_id = row['location_id'].replace('pano_', '')
        if loc_id in coord_map:
            ls_coords.append(coord_map[loc_id])
            ls_clusters.append(row['cluster'])
            
    ls_coords = np.array(ls_coords)
    ls_clusters = np.array(ls_clusters)
    
    print(f"有効な景観地点数: {len(ls_coords)}")

    # 3. 空間近傍探索 (KDTree)
    # 緯度経度をメートル近似に変換 (函館周辺: 1deg lat ~ 111km, 1deg lng ~ 82km)
    lat_to_m = 111000
    lng_to_m = 82000
    
    ls_coords_m = ls_coords * np.array([lat_to_m, lng_to_m])
    tree = KDTree(ls_coords_m)
    
    # 各POIに対して近傍の景観を検索
    poi_coords = df_poi[['lat', 'lng']].values
    poi_coords_m = poi_coords * np.array([lat_to_m, lng_to_m])
    
    # 半径 150m 以内の景観地点を検索
    RADIUS = 150 
    indices = tree.query_ball_point(poi_coords_m, r=RADIUS)
    
    integrated_results = []
    for i, idx_list in enumerate(indices):
        poi_info = df_poi.iloc[i].to_dict()
        
        if len(idx_list) > 0:
            # 近傍にある景観クラスターの最頻値を採用
            neighbor_clusters = ls_clusters[idx_list]
            dominant_ls_cluster = pd.Series(neighbor_clusters).mode()[0]
            ls_count = len(idx_list)
        else:
            dominant_ls_cluster = -1 # 近傍に景観データなし
            ls_count = 0
            
        poi_info['dominant_landscape_cluster'] = dominant_ls_cluster
        poi_info['neighbor_landscape_count'] = ls_count
        integrated_results.append(poi_info)
        
    # 4. 結果の保存
    df_integrated = pd.DataFrame(integrated_results)
    df_integrated.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"統合データを保存しました: {OUTPUT_FILE}")
    print(f"分析完了: {len(df_integrated)} POI")

if __name__ == "__main__":
    main()
