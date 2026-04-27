# -*- coding: utf-8 -*-
"""
実験18.9: 比較用シード（初期クラスタID）の生成。
1. POI機能クラスタ (A: GNNあり, B: GNNなし)
2. 景観クラスタ (StreetCLIP GNN済み)
これらを後の統合プロセスに引き渡す。
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
RESIDUAL_A_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual' # 手法のベース(A)
SEED_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'dual_seeds'
os.makedirs(SEED_OUTPUT_DIR, exist_ok=True)

N_FUNCTIONAL_CLUSTERS = 10
N_LANDSCAPE_CLUSTERS = 10

def main():
    # 1. データの読み込み
    print("データを読み込み中...")
    poi_feats = np.load(INPUT_DIR / 'poi_features.npy') # 799 dims (768 text + 31 time)
    # sv_feats = np.load(INPUT_DIR / 'sv_features.npy') # 今回は直接使わず埋め込みからIDを振る
    with open(INPUT_DIR / 'nodes_metadata.json', 'r', encoding='utf-8') as f:
        nodes_meta = json.load(f)
    
    poi_meta = [n for n in nodes_meta if n['type'] == 'poi']
    sv_meta = [n for n in nodes_meta if n['type'] == 'sv']
    
    # --- シードB：POIクラスタ（GNNなし） ---
    # POIの生データ（テキスト+時間）をそのままクラスタリング
    print("シードB（GNNなしPOIクラスタ）を作成中...")
    scaler_b = StandardScaler()
    poi_feats_scaled = scaler_b.fit_transform(poi_feats)
    linked_b = linkage(poi_feats_scaled, method='ward')
    poi_clusters_b = fcluster(linked_b, N_FUNCTIONAL_CLUSTERS, criterion='maxclust') - 1
    
    # --- シードA：POIクラスタ（GNNあり） ---
    # GNN学習後の埋め込みを用いてクラスタリング（空間的に平滑化された機能）
    print("シードA（GNNありPOIクラスタ）を作成中...")
    df_a_all = pd.read_csv(RESIDUAL_A_DIR / 'residual_embeddings.csv')
    df_a_poi = df_a_all[df_a_all['type'] == 'poi'].copy()
    poi_embs_a = df_a_poi[[c for c in df_a_poi.columns if c.startswith('dim_')]].values
    
    linked_a = linkage(poi_embs_a, method='ward')
    poi_clusters_a = fcluster(linked_a, N_FUNCTIONAL_CLUSTERS, criterion='maxclust') - 1

    # --- 景観シード：SVクラスタ（GNNあり） ---
    # 景観側は提案手法の基盤であるため「GNN済み」に固定
    print("景観シード（GNN済み景観クラスタ）を作成中...")
    df_a_sv = df_a_all[df_a_all['type'] == 'sv'].copy()
    sv_embs = df_a_sv[[c for c in df_a_sv.columns if c.startswith('dim_')]].values
    
    linked_l = linkage(sv_embs, method='ward')
    sv_clusters = fcluster(linked_l, N_LANDSCAPE_CLUSTERS, criterion='maxclust') - 1
    
    # --- 保存 ---
    df_poi_seeds = pd.DataFrame(poi_meta)
    df_poi_seeds['seed_a_id'] = poi_clusters_a
    df_poi_seeds['seed_b_id'] = poi_clusters_b
    df_poi_seeds.to_csv(SEED_OUTPUT_DIR / 'poi_seeds.csv', index=False)
    
    df_sv_seeds = pd.DataFrame(sv_meta)
    df_sv_seeds['landscape_seed_id'] = sv_clusters
    df_sv_seeds.to_csv(SEED_OUTPUT_DIR / 'sv_seeds.csv', index=False)
    
    print(f"シード作成完了: {SEED_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
