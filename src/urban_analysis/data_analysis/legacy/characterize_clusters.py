# -*- coding: utf-8 -*-
"""
このスクリプトは、JSD行列から得られたクラスタリング結果を基に、
各クラスタのセマンティックセグメンテーション比率の平均を計算し、視覚化します。
これにより、各景観エリア（クラスタ）がどのような視覚的特徴を持つかを定量的に把握できます。

主な処理手順：
1.  クラスタ割り当てファイル（jsd_cluster_assignments.csv）を読み込みます。
2.  平均化されたセグメンテーション比率ファイル（jsd_matrix_averaged.csv）を読み込みます。
3.  これらの情報を結合し、クラスタごとにセグメンテーション比率の平均を算出します。
4.  各クラスタの平均セグメンテーション比率を棒グラフで可視化し、画像ファイルとして保存します。
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = '/Users/atsuyakatougi/Desktop/master'
    
    # 入力ファイルのパス
    cluster_assignments_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'jsd_cluster_assignments.csv')
    jsd_averaged_data_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'jsd_matrix_averaged.csv')

    # 出力ディレクトリの定義と作成
    output_dir = os.path.join(base_dir, 'docs', 'results', 'cluster_profiles')
    os.makedirs(output_dir, exist_ok=True)

    print(f"クラスタ割り当てを読み込み中: {cluster_assignments_path}...")
    try:
        clusters_df = pd.read_csv(cluster_assignments_path, index_col=0)
    except FileNotFoundError:
        print(f"エラー: クラスタ割り当てファイルが見つかりません: {cluster_assignments_path}")
        return

    print(f"平均化されたセグメンテーション比率を読み込み中: {jsd_averaged_data_path}...")
    try:
        # jsd_matrix_averaged.csv の行インデックスは location_id そのものなので、そのまま使う
        averaged_ratios_df = pd.read_csv(jsd_averaged_data_path, index_col=0)
    except FileNotFoundError:
        print(f"エラー: 平均化されたセグメンテーション比率ファイルが見つかりません: {jsd_averaged_data_path}")
        return
        
    # jsd_matrix_averaged.csv は JSD matrix 自身なので、これを平均化されたセグメンテーション比率として使うのは間違い
    # jsd_matrix_averaged.csv はJSD行列であり、地点ごとのセグメンテーション比率ではない。
    # 正しい元データは jsd_matrix_averaged.csv の計算に使われた averaged_df。
    # ここで `calculate_jsd.py` の `averaged_df` にアクセスできないため、元の `segmentation_ratios.csv` から再生成する。

    print("元のセグメンテーション比率データを再読み込みし、再平均化中...")
    original_segmentation_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'segmentation_ratios.csv')
    try:
        df_original = pd.read_csv(original_segmentation_path)
    except FileNotFoundError:
        print(f"エラー: 元のセグメンテーション比率ファイルが見つかりません: {original_segmentation_path}")
        return

    df_original.set_index('filename', inplace=True)
    df_original.fillna(0, inplace=True)
    numeric_cols = df_original.select_dtypes(include=np.number).columns
    df_original[numeric_cols] = df_original[numeric_cols] / 100.0

    df_original['location_id'] = df_original.index.str.rsplit('_', n=2).str[0] + '_' + df_original.index.str.rsplit('_', n=2).str[1]
    averaged_ratios_df = df_original.groupby('location_id')[numeric_cols].mean()
    
    # クラスタ情報と平均化された比率情報を結合
    # clusters_df のインデックスが location_id になっていることを確認
    merged_df = averaged_ratios_df.merge(clusters_df, left_index=True, right_index=True, how='inner')

    # クラスタごとの平均セグメンテーション比率を計算
    cluster_profiles = merged_df.groupby('cluster')[numeric_cols].mean()

    print("クラスタごとのプロファイルを計算しました。")
    print(cluster_profiles)
    
    # クラスタごとのプロファイルテーブルをCSVとして保存
    cluster_profiles_table_path = os.path.join(output_dir, 'cluster_profiles_table.csv')
    cluster_profiles.to_csv(cluster_profiles_table_path)
    print(f"クラスタプロファイルのテーブルを保存しました: {cluster_profiles_table_path}")

    # 各クラスタのプロファイルを棒グラフで可視化
    print("クラスタプロファイルの棒グラフを生成中...")
    num_clusters = len(cluster_profiles)
    fig, axes = plt.subplots(num_clusters, 1, figsize=(12, 6 * num_clusters), sharex=True)
    if num_clusters == 1: # 1クラスタの場合、axesは一次元配列ではないため調整
        axes = [axes]

    for i, cluster_id in enumerate(cluster_profiles.index):
        ax = axes[i]
        profile = cluster_profiles.loc[cluster_id]
        profile = profile.sort_values(ascending=False) # 比率が大きい順にソート

        sns.barplot(x=profile.index, y=profile.values, ax=ax, palette='viridis')
        ax.set_title(f'Cluster {cluster_id} Profile (Average Segmentation Ratios)')
        ax.set_ylabel('Average Ratio')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right') # ラベルを回転

    plt.tight_layout()
    plot_output_path = os.path.join(output_dir, 'cluster_profiles_bar_charts.png')
    plt.savefig(plot_output_path)
    print(f"クラスタプロファイルの棒グラフを保存しました: {plot_output_path}")
    plt.close()

    print("処理が完了しました。")

if __name__ == '__main__':
    main()
