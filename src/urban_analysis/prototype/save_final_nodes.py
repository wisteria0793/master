# -*- coding: utf-8 -*-
"""
最終ノードデータの保存スクリプト
GNNの埋め込みと統合地区ラベルを既存のノード情報に結合し、保存する。
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_RESULTS = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results' / 'unified_district_embeddings.npy'
NODES_INFO = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs' / 'nodes_info.csv'
OUTPUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'final_unified_nodes.csv'

def main():
    print("データをロード中...")
    embeddings = np.load(GNN_RESULTS)
    nodes_info = pd.read_csv(NODES_INFO)
    
    # K=12 で最終クラスタリングを実行
    print("最終クラスタリングを実行中 (K=12)...")
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # ラベルを付与
    nodes_info['unified_cluster'] = labels
    
    # 保存
    nodes_info.to_csv(OUTPUT_CSV, index=False)
    print(f"最集データを保存しました: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
