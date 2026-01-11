# -*- coding: utf-8 -*-
"""
方向別（Front, Right, Back, Left）のGNNエンベディングを、
地点（point_id）ごとに平均して1つのベクトルに集約するスクリプト。
"""

import pandas as pd
import os
import numpy as np

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
EMBEDDING_DIM = 64
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', f'embeddings_dim64_directional.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', f'embeddings_dim64_mean.csv')

def main():
    print(f"読み込み中: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("エラー: 入力ファイルが見つかりません。")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"元データ形状: {df.shape}")
    
    # 特徴量カラムを特定（point_id, direction 以外）
    feature_cols = [c for c in df.columns if c not in ['point_id', 'direction']]
    
    # point_id ごとに平均を計算
    print("地点ごとに特徴量を平均化（集約）しています...")
    df_agg = df.groupby('point_id')[feature_cols].mean().reset_index()
    
    print(f"集約後データ形状: {df_agg.shape}")
    
    # 保存
    df_agg.to_csv(OUTPUT_FILE, index=False)
    print(f"保存完了: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
