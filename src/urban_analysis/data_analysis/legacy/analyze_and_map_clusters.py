
# -*- coding: utf-8 -*-
"""
このスクリプトは、JSD行列に基づく一連の景観分析ワークフローを実行します。
クラスタリング、特性評価、地理空間マッピングを単一のスクリプトで処理します。

主な処理手順：
1.  JSD行列CSVファイルを読み込みます。
2.  指定されたクラスタ数に基づき、階層的クラスタリングを実行します。
3.  MDS（多次元尺度構成法）を用いて、クラスタリング結果を2次元に可視化します。
4.  各クラスタの平均的な景観構成（プロファイル）を算出し、テーブルとグラフで保存します。
5.  各地点のクラスタ所属情報を、インタラクティブな地図上に色分けしてプロットします。
6.  生成されたすべての成果物（クラスタ割り当て、プロット、地図）にクラスタ数を含めたファイル名を付け、保存します。
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

def get_coordinates_df(metadata_path):
    """pano_metadata.jsonから座標のデータフレームを読み込む"""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        coords_data = [
            {
                "pano_id": item["pano_id"],
                "latitude": item["api_location"][0],
                "longitude": item["api_location"][1]
            }
            for item in metadata if "api_location" in item and item["api_location"]
        ]
        return pd.DataFrame(coords_data)
    except FileNotFoundError:
        print(f"エラー: メタデータファイルが見つかりません: {metadata_path}")
        return None
    except Exception as e:
        print(f"メタデータの読み込み中にエラーが発生しました: {e}")
        return None

def main():
    # --- 設定 ---
    # ここでクラスタ数を指定します
    N_CLUSTERS = 20
    # ---

    base_dir = '/Users/atsuyakatougi/Desktop/master'
    
    # 入力パス
    input_jsd_matrix_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'jsd_matrix_averaged.csv')
    original_segmentation_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'segmentation_ratios.csv')
    pano_metadata_path = os.path.join(base_dir, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
    
    # 出力ディレクトリ
    results_output_dir = os.path.join(base_dir, 'docs', 'results')
    profiles_output_dir = os.path.join(results_output_dir, 'cluster_profiles')
    data_output_dir = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m')
    os.makedirs(results_output_dir, exist_ok=True)
    os.makedirs(profiles_output_dir, exist_ok=True)

    # --- 1. データ読み込み ---
    print(f"JSD行列を読み込み中: {input_jsd_matrix_path}...")
    try:
        jsd_matrix_df = pd.read_csv(input_jsd_matrix_path, index_col=0)
    except FileNotFoundError:
        print(f"エラー: JSD行列ファイルが見つかりません: {input_jsd_matrix_path}")
        return

    jsd_matrix = jsd_matrix_df.values
    np.fill_diagonal(jsd_matrix, 0)

    # --- 2. 階層的クラスタリングとデンドログラム生成 ---
    print(f"{N_CLUSTERS}個のクラスタで階層的クラスタリングを実行中 (average法)...")
    
    # JSD行列が正方行列なので、linkageが要求する冗長形式(condensed distance matrix)に変換
    distance_matrix_condensed = squareform(jsd_matrix)

    # 階層的クラスタリングを実行
    linked = linkage(distance_matrix_condensed, method='average')

    # デンドログラムを生成して保存
    plt.figure(figsize=(20, 10))
    dendrogram(
        linked,
        orientation='top',
        labels=jsd_matrix_df.index,
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title(f'JSD Hierarchical Clustering Dendrogram (Average Linkage, {N_CLUSTERS} Clusters)')
    plt.xlabel('Location ID')
    plt.ylabel('Distance (JSD)')
    plt.xticks(rotation=90, fontsize=4)
    plt.tight_layout()
    dendrogram_output_path = os.path.join(results_output_dir, f'jsd_dendrogram_{N_CLUSTERS}.png')
    plt.savefig(dendrogram_output_path)
    print(f"デンドログラムを保存しました: {dendrogram_output_path}")
    plt.close()

    # 指定したクラスタ数でクラスタを切り出す
    # fclusterはクラスタ番号が1から始まるため、後で-1して0始まりに合わせる
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    clusters = clusters - 1  # 0から始まるように調整
    
    jsd_matrix_df['cluster'] = clusters
    
    print("クラスタリング結果:")
    print(jsd_matrix_df['cluster'].value_counts().sort_index())

    # クラスタ割り当てをCSVファイルに保存
    cluster_assignments_output_path = os.path.join(data_output_dir, f'jsd_cluster_assignments_{N_CLUSTERS}.csv')
    jsd_matrix_df[['cluster']].to_csv(cluster_assignments_output_path)
    print(f"クラスタ割り当てを保存しました: {cluster_assignments_output_path}")

    # --- 3. MDSによる可視化 ---
    print("MDSを実行中...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10) # n_initを追加して安定化
    mds_coords = mds.fit_transform(jsd_matrix)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(mds_coords[:, 0], mds_coords[:, 1], c=clusters, cmap=plt.get_cmap('viridis', N_CLUSTERS), s=50, alpha=0.7)
    plt.title(f'JSDに基づく景観エリアのMDSプロット ({N_CLUSTERS}クラスタ)')
    plt.xlabel('MDS次元 1')
    plt.ylabel('MDS次元 2')
    plt.colorbar(scatter, ticks=range(N_CLUSTERS), label='クラスタID')
    plt.grid(True)
    plt.tight_layout()
    mds_output_path = os.path.join(results_output_dir, f'jsd_mds_plot_{N_CLUSTERS}.png')
    plt.savefig(mds_output_path)
    print(f"MDSプロットを保存しました: {mds_output_path}")
    plt.close()

    # --- 4. クラスタの特性評価 ---
    print("クラスタのプロファイルを計算中...")
    df_original = pd.read_csv(original_segmentation_path)
    df_original.set_index('filename', inplace=True)
    df_original.fillna(0, inplace=True)
    numeric_cols = df_original.select_dtypes(include=np.number).columns
    df_original[numeric_cols] = df_original[numeric_cols] / 100.0
    
    # location_idの形式をクラスタリング時と合わせる
    df_original['location_id'] = df_original.index
    #df_original['location_id'] = df_original.index.str.rsplit('_', n=1).str[0] # To match averaged pano IDs
    averaged_ratios_df = df_original.groupby('location_id')[numeric_cols].mean()

    # クラスタ情報と比率情報を結合
    merged_df_profiles = averaged_ratios_df.merge(jsd_matrix_df[['cluster']], left_index=True, right_index=True, how='inner')
    cluster_profiles = merged_df_profiles.groupby('cluster')[numeric_cols].mean()

    print("クラスタごとのプロファイル:")
    print(cluster_profiles)
    
    # プロファイルテーブルをCSVとして保存
    profiles_table_path = os.path.join(profiles_output_dir, f'cluster_profiles_table_{N_CLUSTERS}.csv')
    cluster_profiles.to_csv(profiles_table_path)
    print(f"クラスタプロファイルのテーブルを保存しました: {profiles_table_path}")

    # プロファイルを棒グラフで可視化
    num_clusters = len(cluster_profiles)
    for i, cluster_id in enumerate(cluster_profiles.index):
        plt.figure(figsize=(15, 7))
        profile = cluster_profiles.loc[cluster_id]
        profile = profile[profile > 0.01].sort_values(ascending=False) # 1%以上の要素のみ表示

        sns.barplot(x=profile.index, y=profile.values, palette='viridis')
        plt.title(f'Cluster {cluster_id} Profile (Average Segmentation Ratios > 1%)')
        plt.ylabel('Average Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_output_path = os.path.join(profiles_output_dir, f'cluster_{cluster_id}_profile.png')
        plt.savefig(plot_output_path)
        print(f"クラスタ{cluster_id}のプロファイルグラフを保存しました: {plot_output_path}")
        plt.close()

    # --- 5. 地図上へのマッピング ---
    print("座標データを読み込み、地図を作成中...")
    coords_df = get_coordinates_df(pano_metadata_path)
    if coords_df is None:
        return
        
    # 'location_id'からpano_idを抽出
    plot_df = jsd_matrix_df.copy()
    plot_df.reset_index(inplace=True)
    plot_df['pano_id'] = plot_df['location_id'].str.replace('^pano_', '', regex=True).str.split('_h').str[0]

    merged_map_df = pd.merge(plot_df, coords_df, on='pano_id')

    if merged_map_df.empty:
        print("エラー: 地図作成のためのデータマージに失敗しました。")
        return
        
    map_center = [merged_map_df['latitude'].mean(), merged_map_df['longitude'].mean()]
    cluster_map = folium.Map(location=map_center, zoom_start=13)

    colors = sns.color_palette('viridis', n_colors=N_CLUSTERS).as_hex()
    cluster_colors = {i: colors[i] for i in range(N_CLUSTERS)}

    for _, row in merged_map_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=cluster_colors[row['cluster']],
            fill=True,
            fill_color=cluster_colors[row['cluster']],
            fill_opacity=0.8,
            popup=f"Location: {row['location_id']}<br>Cluster: {row['cluster']}"
        ).add_to(cluster_map)

    map_output_path = os.path.join(results_output_dir, f'cluster_map_{N_CLUSTERS}.html')
    cluster_map.save(map_output_path)
    print(f"クラスタの地理的分布マップを保存しました: {map_output_path}")

    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()
