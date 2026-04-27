# -*- coding: utf-8 -*-
"""
実験18.11 Baseline: POI紹介文ベクトルに単なるクラスタリング（GNNなし）を適用し、シードを作成する。
これを学習ベースの統合GNN（18.3）の入力として使用する。
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_baseline_18_3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 10

def main():
    print("POI生データ（紹介文+時間）を読み込み中...")
    poi_feats = np.load(INPUT_DIR / 'poi_features.npy') # 799 dims
    
    # 標準化
    scaler = StandardScaler()
    poi_feats_scaled = scaler.fit_transform(poi_feats)
    
    # 単純なクラスタリング (GNNなし)
    print(f"GNNを介さず、直接 {N_CLUSTERS} クラスタに分類中 (Baseline)...")
    linked = linkage(poi_feats_scaled, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust') - 1
    
    # One-hot形式で保存（GNN入力用）
    n_nodes = len(poi_feats)
    x_seeds = np.zeros((n_nodes, N_CLUSTERS), dtype=np.float32)
    for i, c in enumerate(clusters):
        x_seeds[i, c] = 1.0
        
    np.save(OUTPUT_DIR / 'raw_poi_seeds.npy', x_seeds)
    print(f"Baseline用シードを保存しました: {OUTPUT_DIR / 'raw_poi_seeds.npy'}")

if __name__ == "__main__":
    main()
