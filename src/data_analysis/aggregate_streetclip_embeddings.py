# -*- coding: utf-8 -*-
"""
StreetCLIPによって生成された方向別（Front, Right, Back, Left）の画像エンベディングを、
地点（point_id）ごとに平均して1つのベクトルに集約するスクリプト。
"""

import pandas as pd
import os
import numpy as np

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'streetclip_embeddings', 'streetclip_features.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'streetclip_embeddings', 'streetclip_features_mean.csv')

def main():
    print(f"読み込み中: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print("エラー: 入力ファイルが見つかりません。")
        return

    # データ量が多い場合はメモリに注意が必要ですが、1.6万行x768列程度なら通常は問題ありません
    df = pd.read_csv(INPUT_FILE)
    print(f"元データ形状: {df.shape}")
    
    # 特徴量カラムを特定 ('feat_' で始まる列)
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    print(f"特徴量の次元数: {len(feature_cols)}")
    
    # point_id ごとに平均を計算
    print("地点ごとに特徴量を平均化（集約）しています...")
    # 数値列のみを指定して平均をとる（エラー回避のため）
    df_agg = df.groupby('point_id')[feature_cols].mean().reset_index()
    
    print(f"集約後データ形状: {df_agg.shape}")
    
    # 保存
    df_agg.to_csv(OUTPUT_FILE, index=False)
    print(f"保存完了: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
