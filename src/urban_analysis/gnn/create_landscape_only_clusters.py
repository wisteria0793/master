# -*- coding: utf-8 -*-
"""
実験18.5: アプローチB修正 - 純粋な景観（StreetCLIP）のみを使用したクラスタリング。
GNNによる情報の伝播を行わず、視覚的類似性のみで地区を分類する（Baseline）。
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
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'approach_b'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 20

def main():
    print("景観（SV）特徴量データを読み込み中...")
    sv_feats = np.load(INPUT_DIR / 'sv_features.npy')
    with open(INPUT_DIR / 'nodes_metadata.json', 'r', encoding='utf-8') as f:
        nodes_meta = json.load(f)
    
    # SVノードのみを抽出
    sv_meta = [n for n in nodes_meta if n['type'] == 'sv']
    print(f"対象SVポイント数: {len(sv_meta)}")
    
    print(f"純粋景観（StreetCLIP）のみのクラスタリングを実行中 (k={N_CLUSTERS})...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(sv_feats)
    
    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    
    # 結果の保存
    df_results = pd.DataFrame(sv_meta)
    df_results['landscape_cluster_id'] = clusters - 1
    
    output_path = OUTPUT_DIR / 'sv_landscape_only_clusters.csv'
    df_results.to_csv(output_path, index=False)
    print(f"純粋景観クラスタリング結果を保存しました: {output_path}")

if __name__ == "__main__":
    main()
