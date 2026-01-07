import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import re
import os
from tqdm import tqdm

def main():
    # ファイルパス設定
    input_file_path = 'data/processed/segmentation_results_50m/segmentation_ratios.csv'
    output_file_path = 'data/processed/segmentation_results_50m/directional_jsd_scores_from_mean.csv'

    print(f"データを読み込み中: {input_file_path}...")
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_file_path}")
        return

    # ファイル名から point_id と方向を抽出する関数
    def extract_info(filename):
        # マッチパターン: pano_ID_hHEADING.jpg
        match = re.search(r'pano_([a-zA-Z0-9_-]+)_h(\d+)\.jpg', filename)
        if match:
            return match.group(1), match.group(2)
        return None, None

    print("point ID と heading を抽出中...")
    df[['point_id', 'heading']] = df['filename'].apply(lambda x: pd.Series(extract_info(x)))

    # 情報抽出に失敗した行を削除
    df.dropna(subset=['point_id', 'heading'], inplace=True)

    # heading を方向名にマッピング
    direction_map = {
        '0': 'front',
        '90': 'right',
        '180': 'back',
        '270': 'left'
    }
    df['direction'] = df['heading'].map(direction_map)

    # 計算に不要なカラムを削除
    meta_cols = ['filename', 'heading', 'point_id', 'direction']
    numeric_cols = [c for c in df.columns if c not in meta_cols and c != 'location_id'] 
    
    # 数値カラムが実際に数値であることを確認
    print("確率分布を正規化中...")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # 行の合計が100または1か確認し、合計で割って正規化する（JSD計算のため）
    row_sums = df[numeric_cols].sum(axis=1)
    valid_rows = row_sums > 0
    df = df[valid_rows].copy()
    df[numeric_cols] = df.loc[valid_rows, numeric_cols].div(df.loc[valid_rows, numeric_cols].sum(axis=1), axis=0)

    # point_id でグループ化
    grouped = df.groupby('point_id')

    results = []

    print("各地点における平均ベクトルからのJSDを計算中...")
    for point_id, group in tqdm(grouped):
        # 最大4方向のデータを想定
        vectors = {}
        valid_vectors = []
        
        for _, row in group.iterrows():
            if pd.notna(row['direction']):
                vec = row[numeric_cols].values.astype(float)
                vectors[row['direction']] = vec
                valid_vectors.append(vec)

        available_directions = list(vectors.keys())
        
        if len(available_directions) < 2:
            continue

        # 地点の平均確率分布ベクトルを計算
        mean_vector = np.mean(valid_vectors, axis=0)

        # 各方向のベクトルと平均ベクトルとのJSDを計算
        dir_scores = {}
        for d in available_directions:
            # 平均ベクトルとのJSDを計算
            score = jensenshannon(vectors[d], mean_vector)
            dir_scores[d] = score

        # 最も特徴的な方向（平均からのJSDが最大）を見つける
        if dir_scores:
            most_distinct_dir = max(dir_scores, key=dir_scores.get)
            most_distinct_val = dir_scores[most_distinct_dir]
        else:
            most_distinct_dir = None
            most_distinct_val = 0.0

        # 結果行を作成
        row_data = {
            'point_id': point_id,
            'most_distinct_direction': most_distinct_dir,
            'most_distinct_jsd_score': most_distinct_val,
            'num_directions': len(available_directions)
        }
        
        # 個別のスコアを追加 (平均ベクトルとの乖離度)
        for d in ['front', 'right', 'back', 'left']:
            row_data[f'jsd_from_mean_{d}'] = dir_scores.get(d, None)

        results.append(row_data)

    # 結果からデータフレームを作成
    results_df = pd.DataFrame(results)

    # CSVファイルに保存
    print(f"結果を保存中: {output_file_path}...")
    results_df.to_csv(output_file_path, index=False)
    print("完了しました。")

if __name__ == '__main__':
    main()
