# -*- coding: utf-8 -*-
"""
単体GNN用データ準備スクリプト
1. 景観のみのグラフ構築 (Radius 150m)
2. POIのみのグラフ構築 (Radius 300m)
"""

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
NODES_INFO = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs' / 'nodes_info.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs_individual'

def prepare_gnn_set(df_nodes, name, in_dim, radius):
    print(f"[{name}] グラフデータを準備中...")
    node_count = len(df_nodes)
    
    # 1. 特徴量生成 (One-hot)
    features = np.zeros((node_count, in_dim), dtype=np.float32)
    for i, row in enumerate(df_nodes.itertuples()):
        c = int(row.cluster)
        if c < in_dim:
            features[i, c] = 1.0

    # 2. 隣接グラフ構築
    coords = df_nodes[['lat', 'lng']].values
    coords_m = coords * np.array([111000, 82000])
    tree = KDTree(coords_m)
    adj_list = tree.query_ball_tree(tree, r=radius)
    
    edge_index = []
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            if i != j:
                edge_index.append([i, j])
    
    edge_index = np.array(edge_index).T
    
    # 3. 保存
    save_path = OUTPUT_DIR / name
    save_path.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(save_path / 'nodes.csv', index=False)
    np.save(save_path / 'features.npy', features)
    np.save(save_path / 'edge_index.npy', edge_index)
    print(f"  保存先: {save_path} (エッジ数: {edge_index.shape[1] if edge_index.size > 0 else 0})")

def main():
    nodes_info = pd.read_csv(NODES_INFO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 景観単体 (K=12, r=150)
    df_ls = nodes_info[nodes_info['type'] == 'ls'].copy()
    prepare_gnn_set(df_ls, 'landscape', 12, 150)
    
    # POI単体 (K=8, r=300) ※地点が疎らなため半径を広げる
    df_poi = nodes_info[nodes_info['type'] == 'poi'].copy()
    prepare_gnn_set(df_poi, 'function', 8, 300)

if __name__ == "__main__":
    main()
