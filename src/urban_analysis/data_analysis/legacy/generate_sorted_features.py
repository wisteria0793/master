import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

def main():
    # ファイルパス設定
    input_file_path = 'data/processed/segmentation_results_50m/segmentation_ratios.csv'
    output_file_path = 'data/processed/segmentation_results_50m/location_features_sorted.csv'

    print(f"データを読み込み中: {input_file_path}...")
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_file_path}")
        return

    # ファイル名から point_id を抽出
    def extract_point_id(filename):
        # マッチパターン: pano_ID_hHEADING.jpg
        match = re.search(r'pano_([a-zA-Z0-9_-]+)_h(\d+)\.jpg', filename)
        if match:
            return match.group(1)
        return None

    print("point ID を抽出中...")
    df['point_id'] = df['filename'].apply(extract_point_id)
    df.dropna(subset=['point_id'], inplace=True)

    # 数値データのカラムを特定
    meta_cols = ['filename', 'heading', 'point_id', 'direction', 'location_id']
    numeric_cols = [c for c in df.columns if c not in meta_cols]
    
    # 数値変換と正規化
    print("確率分布を正規化中...")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # 行ごとの正規化（合計を1にする）
    row_sums = df[numeric_cols].sum(axis=1)
    valid_rows = row_sums > 0
    df = df[valid_rows].copy()
    df[numeric_cols] = df.loc[valid_rows, numeric_cols].div(df.loc[valid_rows, numeric_cols].sum(axis=1), axis=0)

    # --- 各ラベルごとに方向の値をソートして連結する ---
    print("各ラベルごとに方向の値を降順ソートして連結中...")
    
    results = []
    
    # point_id ごとにグループ化
    for point_id, group in tqdm(df.groupby('point_id')):
        row_data = {'point_id': point_id}
        
        for col in numeric_cols:
            # 該当ラベルの全方向の値を取得
            values = group[col].values
            # 降順にソート (大きい順)
            # これにより「最も多い方向の値」「二番目に多い方向の値」...という順序になり、回転不変性が得られる
            sorted_values = np.sort(values)[::-1]
            
            # 常に長さ4のベクトルにする（欠損対応: 4枚に満たない場合は0埋め）
            padded_values = np.zeros(4)
            length = min(len(sorted_values), 4)
            padded_values[:length] = sorted_values[:length]
            
            # 特徴量として展開 (例: building_rank1, building_rank2, building_rank3, building_rank4)
            for i in range(4):
                row_data[f'{col}_rank{i+1}'] = padded_values[i]
        
        # サンプルとして方向数も記録
        row_data['num_images'] = len(group)
        results.append(row_data)

    result_df = pd.DataFrame(results)

    # --- 保存 ---
    print(f"特徴量を保存中: {output_file_path}...")
    result_df.to_csv(output_file_path, index=False)
    
    print("完了しました。")
    print(f"生成された特徴量の数: {len(result_df.columns) - 2} (point_id, num_images除く)")
    print("\n生成されたデータの先頭:")
    print(result_df.head())

if __name__ == '__main__':
    main()
