# -*- coding: utf-8 -*-
"""
GNNで学習したCombinedエンベディング（StreetCLIP + Seg-GNN）を、
地点ごとに平均して集約するスクリプト。
"""

import pandas as pd
import os
import numpy as np

BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', 'embeddings_dim64_combined_directional.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', 'embeddings_dim64_combined_mean.csv')

def main():
    print(f"読み込み中: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("エラー: 入力ファイルが見つかりません。")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # point_id, direction 以外を平均
    feature_cols = [c for c in df.columns if c not in ['point_id', 'direction']]
    
    print("地点ごとに特徴量を平均化しています...")
    df_agg = df.groupby('point_id')[feature_cols].mean().reset_index()
    
    df_agg.to_csv(OUTPUT_FILE, index=False)
    print(f"保存完了: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
