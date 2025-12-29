# -*- coding: utf-8 -*-
"""
このスクリプトは、景観のセグメンテーション比率に基づき、各ビュー（四方の画像）を
独立したデータポイントとして扱って分析ワークフローを実行します。
クラスタリング、特性評価、地理空間マッピングを単一のスクリプトで処理します。

主な処理手順：
1.  各ビューのセグメンテーション比率データ（separate_view_vectors.csv）を読み込みます。
2.  各ビューの特徴ベクトルから距離行列を計算します。
3.  指定されたクラスタ数に基づき、階層的クラスタリングを実行します。
4.  各クラスタの平均的な景観構成（プロファイル）を算出します。
5.  各ビューのクラスタ所属情報を、インタラクティブな地図上に色分けしてプロットします。
    - マーカーの色は、その地点で最も優勢なクラスタを反映します。
    - マーカーをクリックすると、4方向すべてのクラスタ情報がポップアップで表示されます。
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

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
    input_vectors_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'separate_view_vectors.csv')
    pano_metadata_path = os.path.join(base_dir, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
    
    # 出力ディレクトリ
    output_suffix = f"independent_{N_CLUSTERS}_popup"
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
    feature_cols = views_df.columns.drop(['point_id', 'direction'])
    feature_matrix = views_df[feature_cols].values
    view_ids = views_df['point_id'] + '_' + views_df['direction']
    
    print("ビュー間の距離を計算中 (ユークリッド距離)...")
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
    plt.xticks(rotation=90, fontsize=2)
    plt.tight_layout()
    dendrogram_output_path = os.path.join(results_output_dir, f'dendrogram_{output_suffix}.png')
    plt.savefig(dendrogram_output_path)
    print(f"デンドログラムを保存しました: {dendrogram_output_path}")
    plt.close()

    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    views_df['cluster'] = clusters - 1
    
    print("クラスタリング結果:")
    print(views_df['cluster'].value_counts().sort_index())

    cluster_assignments_output_path = os.path.join(data_output_dir, f'cluster_assignments_{output_suffix}.csv')
    views_df[['point_id', 'direction', 'cluster']].to_csv(cluster_assignments_output_path, index=False)
    print(f"クラスタ割り当てを保存しました: {cluster_assignments_output_path}")

    # --- 3. クラスタの特性評価 ---
    print("クラスタのプロファイルを計算中...")
    feature_cols = views_df.columns.drop(['point_id', 'direction', 'cluster'])
    cluster_profiles = views_df.groupby('cluster')[feature_cols].mean()
    profiles_table_path = os.path.join(profiles_output_dir, f'cluster_profiles_table_{output_suffix}.csv')
    cluster_profiles.to_csv(profiles_table_path)
    print(f"クラスタプロファイルのテーブルを保存しました: {profiles_table_path}")

    num_clusters = len(cluster_profiles)
    for i, cluster_id in enumerate(cluster_profiles.index):
        plt.figure(figsize=(15, 7))
        profile = cluster_profiles.loc[cluster_id] * 100
        profile = profile[profile > 1].sort_values(ascending=False)
        sns.barplot(x=profile.index, y=profile.values, palette='viridis')
        plt.title(f'Cluster {cluster_id} Profile (Average Segmentation Ratios > 1%)')
        plt.ylabel('Average Ratio (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_output_path = os.path.join(profiles_output_dir, f'cluster_{cluster_id}_profile_{output_suffix}.png')
        plt.savefig(plot_output_path)
        plt.close()
    print(f"{num_clusters}個のクラスタプロファイルグラフを保存しました。")
    
    # --- 4. 地理空間マッピング（ポップアップ形式） ---
    print("\n地理空間マッピングを開始します...")
    coordinates_df = get_coordinates_df(pano_metadata_path)
    
    if coordinates_df is None:
        print("座標データが見つからないため、地理空間マッピングをスキップします。")
        return

    merged_df = pd.merge(
        views_df,
        coordinates_df.rename(columns={'pano_id': 'point_id'}),
        on='point_id',
        how='inner'
    )
    
    print("マッピング用のデータが準備できました。")

    print("インタラクティブな地図を生成中...")
    if not merged_df.empty:
        map_center = [merged_df['latitude'].mean(), merged_df['longitude'].mean()]
    else:
        map_center = [41.7687, 140.7288] # デフォルト値（函館市）
    
    m = folium.Map(location=map_center, zoom_start=14)

    colors = plt.cm.get_cmap('viridis', N_CLUSTERS)
    
    def get_hex_color(cluster_id):
        color_rgba = colors(int(cluster_id))
        return '#%02x%02x%02x' % (int(color_rgba[0]*255), int(color_rgba[1]*255), int(color_rgba[2]*255))

    for point_id, group in merged_df.groupby('point_id'):
        if group.empty:
            continue
        
        dominant_cluster = group['cluster'].mode()[0]
        marker_color = get_hex_color(dominant_cluster)
        
        lat = group['latitude'].iloc[0]
        lon = group['longitude'].iloc[0]
        
        popup_html = f"<b>地点ID: {point_id}</b><br><hr><b>各方向のクラスタ:</b><br>"
        directions_order = ['F', 'R', 'B', 'L']
        direction_data = {row['direction']: row['cluster'] for _, row in group.iterrows()}
        
        popup_html += "<table>"
        for direction in directions_order:
            cluster_id = direction_data.get(direction)
            if cluster_id is not None:
                color = get_hex_color(cluster_id)
                popup_html += (
                    f"<tr>"
                    f"<td><b>{direction}:</b></td>"
                    f"<td><span style='background-color:{color}; border:1px solid #000; display:inline-block; width:12px; height:12px; margin-right:5px;'></span></td>"
                    f"<td>クラスタ {cluster_id}</td>"
                    f"</tr>"
                )
        popup_html += "</table>"
        
        iframe = folium.IFrame(popup_html, width=200, height=150)
        popup = folium.Popup(iframe, max_width=200)

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.8,
            popup=popup,
            tooltip=f"地点ID: {point_id}<br>代表クラスタ: {dominant_cluster}"
        ).add_to(m)

    map_output_path = os.path.join(results_output_dir, f'cluster_map_{output_suffix}.html')
    m.save(map_output_path)
    print(f"クラスタ地図を保存しました: {map_output_path}")

    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()
