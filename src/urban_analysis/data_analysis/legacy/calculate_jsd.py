# -*- coding: utf-8 -*-
"""
このスクリプトは、セマンティックセグメンテーションの結果から得られたピクセルごとのカテゴリ比率（確率分布）を処理し、
地点ごとの平均確率分布間のイェンセン・シャノン・ダイバージェンス（JSD）を計算します。

主な処理手順：
1.  CSVファイルからセグメンテーション比率データを読み込みます。
2.  Google Street View のデータのように、同一地点から複数（例: 4方向）の画像が取得されている場合、
    これらの画像のセグメンテーション比率を地点ごとに平均化し、単一の確率分布を作成します。
3.  地点ごとの平均化された確率分布間のJSD行列を計算し、分布間の類似度（または相違度）を数値化します。
4.  計算されたJSD行列をCSVファイルとして保存します。

これにより、各地点の景観特性の類似性を定量的に分析することができます。
"""

# 必要なライブラリをインポート
import pandas as pd  # データフレーム操作のため
import numpy as np   # 数値計算のため
from scipy.spatial.distance import jensenshannon  # イェンセン・シャノン・ダイバージェンス計算のため
from itertools import combinations  # 組み合わせを生成するため
import os            # ファイルパス操作のため
from tqdm import tqdm  # 処理の進捗表示のため

def calculate_jsd_matrix(df):
    """
    与えられたデータフレーム（各行が確率分布）に対して、イェンセン・シャノン・ダイバージェンス（JSD）行列を計算します。

    Args:
        df (pd.DataFrame): 各行が確率分布を表すデータフレーム。

    Returns:
        pd.DataFrame: JSD値の行列。
    """
    image_names = df.index  # データフレームのインデックス（画像名または地点ID）を取得
    # JSD結果を格納するための空のデータフレームを初期化
    jsd_matrix = pd.DataFrame(np.zeros((len(image_names), len(image_names))), index=image_names, columns=image_names)
    
    # tqdmを使用して進捗バーを表示しながら、すべての画像ペアの組み合わせを反復処理
    pair_combinations = list(combinations(image_names, 2)) # 全ての異なるペアの組み合わせをリスト化
    for image1, image2 in tqdm(pair_combinations, desc="JSDを計算中"): # 進捗バー付きでペアをイテレート
        dist1 = df.loc[image1].values  # 最初の画像の分布を取得
        dist2 = df.loc[image2].values  # 2番目の画像の分布を取得
        
        # 確率分布に負の値が含まれていないことを確認
        if (dist1 < 0).any() or (dist2 < 0).any():
            raise ValueError("確率分布に負の値を含めることはできません。")

        jsd = jensenshannon(dist1, dist2)  # 2つの分布間のJSDを計算
        jsd_matrix.loc[image1, image2] = jsd  # 結果をJSD行列に格納
        jsd_matrix.loc[image2, image1] = jsd  # 対称行列なので逆も同じ値を格納
        
    return jsd_matrix  # 完成したJSD行列を返す

def main():
    """
    メイン関数：データを読み込み、JSDを計算し、結果を保存します。
    """
    # ファイルパスを定義
    base_dir = '/Users/atsuyakatougi/Desktop/master' # プロジェクトのベースディレクトリ
    # 入力CSVファイルのパス (セグメンテーション比率が含まれる)
    input_csv_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'segmentation_ratios.csv')
    # 出力CSVファイルのパス (平均化されたデータに基づくJSD行列)
    output_csv_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'jsd_matrix_averaged.csv')

    # セグメンテーション比率データを読み込み
    print(f"データを読み込み中: {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"エラー: 入力ファイルが見つかりません: {input_csv_path}")
        return

    # データの前処理
    print("データを前処理中...")
    # 'filename'カラムが存在するか確認
    if 'filename' not in df.columns:
        print("エラー: 'filename'カラムが見つかりません。")
        return
        
    df.set_index('filename', inplace=True)  # 'filename'カラムをインデックスに設定
    df.fillna(0, inplace=True) # 欠損値(NaN)を0で埋める
    
    # 比率（パーセンテージ）を確率に変換 (100で割る)
    numeric_cols = df.select_dtypes(include=np.number).columns # 数値型のカラムを選択
    df[numeric_cols] = df[numeric_cols] / 100.0 # 100で割って確率に変換

    # --- 地点に基づく平均化処理 ---
    print("各地点の分布を平均化中...")
    # ファイル名インデックスから地点IDを抽出 (例: 'pano_zfF1NCp80jVWlZbe7ie5Fg_h270.jpg' から 'pano_zfF1NCp80jVWlZbe7ie5Fg' を抽出)
    # '_'で分割し、最後の2つの要素（_hXXXと.jpg）を除いた部分を結合して地点IDとする
    df['location_id'] = df.index.str.rsplit('_', n=2).str[0] + '_' + df.index.str.rsplit('_', n=2).str[1]
    
    # location_idでグループ化し、各地点の平均分布を計算
    averaged_df = df.groupby('location_id')[numeric_cols].mean()

    # --- 平均化データに対するJSD計算 ---
    # JSD行列を計算
    print("平均化されたデータに基づいてイェンセン・シャノン・ダイバージェンス行列を計算中...")
    jsd_matrix = calculate_jsd_matrix(averaged_df)

    # 結果を保存
    print(f"JSD行列を保存中: {output_csv_path}...")
    jsd_matrix.to_csv(output_csv_path)  # 計算結果をCSVファイルとして保存

    print("処理が完了しました。")

if __name__ == '__main__':
    main() # スクリプトが直接実行された場合にmain関数を呼び出す