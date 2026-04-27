# -*- coding: utf-8 -*-
"""
GNN用グラフデータ準備スクリプト
1. POIと景観クラスターをマージ
2. 景観地点の集約（4方向→1地点）
3. One-hot特徴量の生成
4. 空間近接グラフ（隣接行列）の構築
"""

import pandas as pd
import numpy as np
import json
import torch
from scipy.spatial import KDTree
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
REFINED_LS_CSV = PROJECT_ROOT / 'data' / 'processed' / 'refined_landscape_clusters_k12.csv'
REFINED_POI_CSV = PROJECT_ROOT / 'data' / 'processed' / 'refined_poi_clusters_k8.csv'
METADATA_FILE = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered' / 'nodes_metadata.json'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs'

def main():
    print("データをロード中...")
    df_ls = pd.read_csv(REFINED_LS_CSV)
    df_poi = pd.read_csv(REFINED_POI_CSV)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # SVの座標マッピング
    coord_map = {n['id']: (n['lat'], n['lng']) for n in meta if n['type'] == 'sv'}

    # 1. 景観地点の集約 (4方向を1つの地点IDにまとめる)
    print("景観地点の集約中...")
    ls_aggregated = df_ls.groupby('point_id')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
    
    # 座標情報の付与
    ls_final = []
    for _, row in ls_aggregated.iterrows():
        pid = row['point_id']
        if pid in coord_map:
            lat, lng = coord_map[pid]
            ls_final.append({'id': pid, 'lat': lat, 'lng': lng, 'cluster': row['cluster'], 'type': 'ls'})
    df_ls_final = pd.DataFrame(ls_final)
    print(f"集約後の景観地点数: {len(df_ls_final)}")

    # 2. 全ノードの統合
    df_poi['type'] = 'poi'
    # POIのIDを識別子にする
    all_nodes = pd.concat([
        df_poi[['id', 'lat', 'lng', 'cluster', 'type']],
        df_ls_final[['id', 'lat', 'lng', 'cluster', 'type']]
    ]).reset_index(drop=True)
    
    # 座標が NaN の地点を完全に除外
    all_nodes = all_nodes.dropna(subset=['lat', 'lng'])
    
    node_count = len(all_nodes)
    print(f"総ノード数: {node_count}")

    # 3. 特徴量生成 (One-hot)
    # POI: 8dims, LS: 12dims -> Total 20dims
    features = np.zeros((node_count, 8 + 12), dtype=np.float32)
    for i, row in all_nodes.iterrows():
        c = int(row['cluster'])
        if row['type'] == 'poi':
            features[i, c] = 1.0  # 前半8個がPOI
        else:
            features[i, 8 + c] = 1.0 # 後半12個がLS

    # 4. 隣接グラフの構築 (KDTree)
    print("隣接グラフ構築中...")
    coords = all_nodes[['lat', 'lng']].values
    # メートル近似
    coords_m = coords * np.array([111000, 82000])
    tree = KDTree(coords_m)
    
    # 半径150mで接続
    RADIUS = 150
    adj_list = tree.query_ball_tree(tree, r=RADIUS)
    
    # 疎行列形式でのエッジリスト作成
    edge_index = []
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            if i != j:
                edge_index.append([i, j])
    
    edge_index = np.array(edge_index).T # (2, E)
    
    # 5. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_nodes.to_csv(OUTPUT_DIR / 'nodes_info.csv', index=False)
    np.save(OUTPUT_DIR / 'features.npy', features)
    np.save(OUTPUT_DIR / 'edge_index.npy', edge_index)
    
    print(f"GNN用データを保存しました: {OUTPUT_DIR}")
    print(f"エッジ数: {edge_index.shape[1]}")

if __name__ == "__main__":
    main()
