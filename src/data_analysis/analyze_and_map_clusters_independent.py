# -*- coding: utf-8 -*-
"""
このスクリプトは、景観のセグメンテーション比率に基づき、各ビュー（四方の画像）を
独立したデータポイントとして扱って分析ワークフローを実行します。
クラスタリング、特性評価、地理空間マッピングを単一のスクリプトで処理します。

主な処理手順：
1.  各ビューのセグメンテーション比率データ（separate_view_vectors.csv）を読み込みます。
2.  各ビューの特徴ベクトルから距離行列を計算します。
3.  指定されたクラスタ数に基づき、階層的クラスタリングを実行します。
4.  MDS（多次元尺度構成法）を用いて、クラスタリング結果を2次元に可視化します。
5.  各クラスタの平均的な景観構成（プロファイル）を算出します。
6.  各ビューのクラスタ所属情報を、インタラクティブな地図上に色分けしてプロットします。
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
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
    N_CLUSTERS = 20
    # ---

    base_dir = '/Users/atsuyakatougi/Desktop/master'
    
    # 入力パス
    # 変更点: 平均化されたJSD行列の代わりに、独立したビューのベクトルデータを読み込む
    input_vectors_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'separate_view_vectors.csv')
    pano_metadata_path = os.path.join(base_dir, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
    
    # 出力ディレクトリ
    output_suffix = f"independent_{N_CLUSTERS}"
    results_output_dir = os.path.join(base_dir, 'docs', 'results')
    profiles_output_dir = os.path.join(results_output_dir, 'cluster_profiles')
    data_output_dir = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m')
    os.makedirs(results_output_dir, exist_ok=True)
    os.makedirs(profiles_output_dir, exist_ok=True)

    # --- 1. データ読み込み ---
    print(f"独立ビューのベクトルデータを読み込み中: {input_vectors_path}...")
    try:
        views_df = pd.read_csv(input_vectors_path)
    except FileNotFoundError:
        print(f"エラー: ベクトルデータファイルが見つかりません: {input_vectors_path}")
        return
    
    print("読み込み完了。データの一部:")
    print(views_df.head())

    # --- 2. 階層的クラスタリング ---
    print("クラスタリングのための特徴量を準備中...")
    # 'point_id' と 'direction' を除く数値列を特徴量として使用
    feature_cols = views_df.columns.drop(['point_id', 'direction'])
    feature_matrix = views_df[feature_cols].values

    # 各ビューの一意なIDを作成（デンドログラムのラベル用）
    view_ids = views_df['point_id'] + '_' + views_df['direction']
    
    print("ビュー間の距離を計算中 (ユークリッド距離)...")
    # pdistは特徴ベクトルから直接、距離行列（凝縮形式）を計算する
    distance_matrix_condensed = pdist(feature_matrix, metric='euclidean')

    print(f"{N_CLUSTERS}個のクラスタで階層的クラスタリングを実行中 (average法)...")
    linked = linkage(distance_matrix_condensed, method='average')

    # デンドログラムを生成して保存
    plt.figure(figsize=(20, 10))
    dendrogram(
        linked,
        orientation='top',
        labels=view_ids.tolist(),
        distance_sort='descending',
        show_leaf_counts=True
    )
    plt.title(f'Hierarchical Clustering Dendrogram for Independent Views (Average Linkage, {N_CLUSTERS} Clusters)')
    plt.xlabel('View ID (point_id + direction)')
    plt.ylabel('Distance (Euclidean)')
    plt.xticks(rotation=90, fontsize=2) # ラベルが多すぎるのでフォントを小さく
    plt.tight_layout()
    dendrogram_output_path = os.path.join(results_output_dir, f'dendrogram_{output_suffix}.png')
    plt.savefig(dendrogram_output_path)
    print(f"デンドログラムを保存しました: {dendrogram_output_path}")
    plt.close()

    # 指定したクラスタ数でクラスタを切り出す (fclusterは1始まり)
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    views_df['cluster'] = clusters - 1  # 0から始まるように調整
    
    print("クラスタリング結果:")
    print(views_df['cluster'].value_counts().sort_index())

    # クラスタ割り当てをCSVファイルに保存
    cluster_assignments_output_path = os.path.join(data_output_dir, f'cluster_assignments_{output_suffix}.csv')
    views_df[['point_id', 'direction', 'cluster']].to_csv(cluster_assignments_output_path, index=False)
    print(f"クラスタ割り当てを保存しました: {cluster_assignments_output_path}")
    
    # --- 5. 地理空間マッピング ---
    print("パノラマメタデータを読み込み中...")
    coordinates_df = get_coordinates_df(pano_metadata_path)
    
    if coordinates_df is None:
        print("座標データが見つからないため、地理空間マッピングをスキップします。")
        return

    # 座標データとビューごとのクラスタ情報をマージ
    # 'pano_id' を 'point_id' にリネームして、views_df とマージする
    merged_df = pd.merge(
        views_df,
        coordinates_df.rename(columns={'pano_id': 'point_id'}),
        on='point_id',
        how='inner'
    )
    
    print("マッピング用のデータが準備できました。データの一部:")
    print(merged_df.head())

    # --- デバッグ ---
    print("\n--- デバッグ情報 ---")
    print(f"マッピング用データフレームの形状: {merged_df.shape}")
    print("point_idごとの行数（上位5件）:")
    print(merged_df['point_id'].value_counts().head())
    print("direction列のユニークな値:")
    print(merged_df['direction'].unique())
    print("--------------------\n")
    # --- デバッグ終了 ---

    print("インタラクティブな地図を生成中...")
    # 地図の中心座標を計算
    map_center = [merged_df['latitude'].mean(), merged_df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=14)

    # クラスタごとの色を設定
    colors = plt.cm.get_cmap('viridis', N_CLUSTERS)
    
    # 地図の見やすさのためのオフセット値（緯度・経度）
    # この値は地図のズームレベルによって調整が必要
    offset_val = 0.00005 

    # 各方角に対応するオフセットを定義
    direction_offsets = {
        'F': (offset_val, 0),       # 前方 (North)
        'R': (0, offset_val),       # 右 (East)
        'B': (-offset_val, 0),      # 後方 (South)
        'L': (0, -offset_val)       # 左 (West)
    }

    # デバッグ用に最初のpoint_idを追跡
    test_point_id_printed = False
    test_point_id = merged_df['point_id'].iloc[0] if not merged_df.empty else None

    for idx, row in merged_df.iterrows():
        cluster_id = row['cluster']
        direction = row['direction']
        
        # オフセットを取得
        lat_offset, lon_offset = direction_offsets.get(direction, (0, 0))
        
        # オフセットを適用した位置を計算
        location = [row['latitude'] + lat_offset, row['longitude'] + lon_offset]

        # --- デバッグ（特定の1地点のみ）---
        if row['point_id'] == test_point_id and not test_point_id_printed:
             print(f"地点ID {row['point_id']} の方角 {direction}: 元座標 [{row['latitude']}, {row['longitude']}] -> 計算後座標 {location}")
        # --- デバッグ終了 ---
        
        # RGBAカラーをHex形式に変換
        color_rgba = colors(cluster_id)
        color_hex = '#%02x%02x%02x' % (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))
            
        folium.CircleMarker(
            location=location,
            radius=3,  # マーカーを小さくして見やすくする
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.9,
            tooltip=f"地点ID: {row['point_id']}<br>方角: {direction}<br>クラスタ: {cluster_id}"
        ).add_to(m)

    # 1地点分のログが出力されたらフラグを更新
    if test_point_id:
        test_point_id_printed = True

    map_output_path = os.path.join(results_output_dir, f'cluster_map_{output_suffix}.html')
    m.save(map_output_path)
    print(f"クラスタ地図を保存しました: {map_output_path}")

    # --- 4. クラスタの特性評価 ---
    print("クラスタのプロファイルを計算中...")
    
    # views_dfにはすでに比率とクラスタIDが含まれているので、そのまま利用
    # 特徴量として使った列のみを対象にする
    feature_cols = views_df.columns.drop(['point_id', 'direction', 'cluster'])
    cluster_profiles = views_df.groupby('cluster')[feature_cols].mean()

    print("クラスタごとのプロファイル:")
    print(cluster_profiles)
    
    # プロファイルテーブルをCSVとして保存
    profiles_table_path = os.path.join(profiles_output_dir, f'cluster_profiles_table_{output_suffix}.csv')
    cluster_profiles.to_csv(profiles_table_path)
    print(f"クラスタプロファイルのテーブルを保存しました: {profiles_table_path}")

    # プロファイルを棒グラフで可視化
    num_clusters = len(cluster_profiles)
    for i, cluster_id in enumerate(cluster_profiles.index):
        plt.figure(figsize=(15, 7))
        # 100倍してパーセント表示にする
        profile = cluster_profiles.loc[cluster_id] * 100
        profile = profile[profile > 1].sort_values(ascending=False) # 1%以上の要素のみ表示

        sns.barplot(x=profile.index, y=profile.values, palette='viridis')
        plt.title(f'Cluster {cluster_id} Profile (Average Segmentation Ratios > 1%)')
        plt.ylabel('Average Ratio (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_output_path = os.path.join(profiles_output_dir, f'cluster_{cluster_id}_profile_{output_suffix}.png')
        plt.savefig(plot_output_path)
        # print(f"クラスタ{cluster_id}のプロファイルグラフを保存しました: {plot_output_path}")
        plt.close()
    print(f"{num_clusters}個のクラスタプロファイルグラフを保存しました。")

    # ここに後続の処理（マッピングなど）を実装していく

    print("\nデータ読み込み処理が完了しました。")

if __name__ == '__main__':
    main()
