# -*- coding: utf-8 -*-
"""
階層的クラスタリングの実行結果を読み込み、各クラスタに属する施設の
詳細情報（カテゴリなど）を付与して、Excelファイルにまとめるスクリプト。

■ 処理概要
1. `hierarchical_clustering.py` が出力したクラスタリング結果のCSVファイルを読み込む。
2. 全施設の情報が格納されたJSONファイルを読み込み、カテゴリなどの詳細情報を抽出する。
3. 上記2つの情報を、施設の「名前」をキーとして結合（マージ）する。
4. 最終的な結果を、クラスタごとにシートを分けてExcelファイルに出力する。

■ 使い方
コマンドラインで、集計したいクラスタ数を `--num_clusters` 引数で指定して実行する。
例: python src/data_analysis/summarize_hierarchical_facilities.py --num_clusters 17
"""

import pandas as pd
import json
import argparse
from pathlib import Path

# --- 定数定義 ---
# 入力・出力のベースとなるディレクトリやファイル名を定義
BASE_INPUT_DIR = Path("data/hierarchical_clustering")
FACILITIES_JSON_PATH = Path("data/processed/poi/filtered_facilities.json")
BASE_OUTPUT_DIR = Path("data/hierarchical_clustering")


def load_cluster_results(num_clusters: int) -> pd.DataFrame:
    """
    指定されたクラスタ数の階層的クラスタリング結果CSVを読み込む。
    """
    # ファイルパスを動的に構築
    file_path = BASE_INPUT_DIR / f'locations_with_clusters_hc_{num_clusters}_with_address.csv'
    print(f"クラスタリング結果を読み込み中: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"クラスタリング結果ファイルが見つかりません: {file_path}\n`hierarchical_clustering.py` を --num_clusters {num_clusters} で実行したか確認してください。")
        
    return pd.read_csv(file_path)


def load_facility_details() -> pd.DataFrame:
    """
    全施設情報のJSONファイルを読み込み、必要な詳細情報（カテゴリ等）を抽出する。
    """
    print(f"施設詳細情報を読み込み中: {FACILITIES_JSON_PATH}")
    
    if not FACILITIES_JSON_PATH.exists():
        raise FileNotFoundError(f"施設情報ファイルが見つかりません: {FACILITIES_JSON_PATH}")

    with open(FACILITIES_JSON_PATH, 'r', encoding='utf-8') as f:
        facilities_data = json.load(f)
    
    # 必要な情報だけを抽出してリストに格納
    processed_facilities = []
    for facility in facilities_data:
        # 'categories' と 'google_places_data' の 'types' をカンマ区切りの文字列に変換
        categories_str = ', '.join(facility.get('categories', []))
        google_types_str = ', '.join(facility.get('google_places_data', {}).get('details', {}).get('types', []))
        
        processed_facilities.append({
            'name': facility.get('name'),
            'categories_str': categories_str,
            'google_types_str': google_types_str
        })
        
    return pd.DataFrame(processed_facilities)


def main(num_clusters: int):
    """
    メイン処理を実行する関数。
    """
    try:
        # --- 1. 各種データの読み込み ---
        df_hierarchical = load_cluster_results(num_clusters)
        df_facilities_details = load_facility_details()
        
        # --- 2. データの結合 ---
        # 施設の 'name' をキーとして、クラスタリング結果と施設詳細情報を結合する
        print("データを結合中...")
        merged_df = pd.merge(df_hierarchical, df_facilities_details, on='name', how='left')
        
        # --- 3. Excelファイルへの書き込み ---
        # 出力ファイルパスを動的に構築
        output_path = BASE_OUTPUT_DIR / f'hierarchical_clustered_facilities_cl_{num_clusters}.xlsx'
        BASE_OUTPUT_DIR.mkdir(exist_ok=True) # 出力ディレクトリがなければ作成

        print(f"Excelファイルを作成中: {output_path}")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 'cluster'列の値ごとにグループ化し、ループ処理
            for cluster_id, group_df in merged_df.groupby('cluster'):
                # Excelのシート名として使用（シート名は31文字以内という制約に対応）
                sheet_name = f'Cluster_{cluster_id}'
                
                # 出力する列を選択・並び替え
                output_df = group_df[['name', 'address', 'categories_str', 'google_types_str']]
                
                # グループのデータフレームを、指定したシート名でExcelシートに書き出す
                output_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  - Cluster {cluster_id} のデータをシート '{sheet_name}' に書き込みました。")
                
        print(f"\n処理が正常に完了しました。Excelファイルが作成されました: {output_path}")
        
    except FileNotFoundError as e:
        print(f"\nエラー: ファイルが見つかりません。\n{e}")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    # --- コマンドライン引数の設定 ---
    parser = argparse.ArgumentParser(
        description="階層的クラスタリングの結果を、施設詳細情報と合わせてExcelにまとめるスクリプト。"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        required=True, # この引数は必須とする
        help="集計したいクラスタリング結果のクラスタ数 (例: 17)。"
    )
    args = parser.parse_args()

    # メイン関数を実行
    main(args.num_clusters)