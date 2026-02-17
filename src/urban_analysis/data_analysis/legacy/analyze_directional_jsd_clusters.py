# -*- coding: utf-8 -*-
"""
このスクリプトは、方向別JSDスコア（景観の特異性スコア）に基づくクラスタリングと地図化を行います。

主な処理手順：
1.  `directional_jsd_scores.csv` を読み込みます。
2.  各地点の方向別JSDスコア (`avg_jsd_front`, `avg_jsd_right` 等) を特徴量として階層的クラスタリングを実行します。
3.  クラスタリング結果を地図（Folium）にプロットします。
4.  クラスタごとの特徴（平均JSDスコアなど）を集計し、保存します。
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
                "point_id": item["pano_id"], # キーを point_id に統一
                "latitude": item["api_location"][0],
                "longitude": item["api_location"][1]
            }
            for item in metadata if "api_location" in item and item["api_location"] and "pano_id" in item
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
    N_CLUSTERS = 6  # クラスタ数を設定（必要に応じて調整）
    # ---

    base_dir = '/Users/atsuyakatougi/Desktop/master'
    
    # 入力パス
    input_scores_path = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m', 'directional_jsd_scores.csv')
    pano_metadata_path = os.path.join(base_dir, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
    
    # 出力ディレクトリ
    results_output_dir = os.path.join(base_dir, 'docs', 'results')
    data_output_dir = os.path.join(base_dir, 'data', 'processed', 'segmentation_results_50m')
    os.makedirs(results_output_dir, exist_ok=True)

    # --- 1. データ読み込み ---
    print(f"データを読み込み中: {input_scores_path}...")
    try:
        df = pd.read_csv(input_scores_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_scores_path}")
        return

    # クラスタリングに使用する特徴量を選択
    # 方向ごとのJSDスコアを使用
    feature_cols = ['avg_jsd_front', 'avg_jsd_right', 'avg_jsd_back', 'avg_jsd_left']
    
    # 欠損値がある場合は0で埋める（あるいは削除）
    X = df[feature_cols].fillna(0).values

    # --- 2. 階層的クラスタリング ---
    print(f"階層的クラスタリングを実行中 (Ward法, {N_CLUSTERS}クラスタ)...")
    
    # 距離行列を計算せずに直接 linkage にデータを渡す場合、metric='euclidean' がデフォルト
    # Ward法はユークリッド距離に基づく
    linked = linkage(X, method='ward')

    # デンドログラムを保存
    plt.figure(figsize=(12, 7))
    dendrogram(
        linked,
        orientation='top',
        # labels=df['point_id'].values, # ラベルが多いと見づらいので省略または調整
        truncate_mode='lastp',  # 末端のクラスタのみ表示
        p=30,
        show_leaf_counts=True,
        no_labels=True
    )
    plt.title(f'Directional JSD Clustering Dendrogram (Ward Linkage)')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    dendrogram_path = os.path.join(results_output_dir, 'directional_jsd_dendrogram.png')
    plt.savefig(dendrogram_path)
    print(f"デンドログラムを保存しました: {dendrogram_path}")
    plt.close()

    # クラスタ割り当て
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    df['cluster'] = clusters

    print("クラスタごとのサンプル数:")
    print(df['cluster'].value_counts().sort_index())

    # --- 3. クラスタの特徴分析 ---
    print("クラスタごとの平均スコア:")
    cluster_profile = df.groupby('cluster')[feature_cols + ['most_distinct_jsd_score']].mean()
    print(cluster_profile)
    
    profile_path = os.path.join(results_output_dir, 'directional_jsd_cluster_profiles.csv')
    cluster_profile.to_csv(profile_path)
    print(f"クラスタプロファイルを保存しました: {profile_path}")

    # --- 4. 地図プロット ---
    print("地図を作成中...")
    coords_df = get_coordinates_df(pano_metadata_path)
    
    if coords_df is not None:
        # 座標データをマージ
        merged_df = pd.merge(df, coords_df, on='point_id', how='inner')
        
        if not merged_df.empty:
            map_center = [merged_df['latitude'].mean(), merged_df['longitude'].mean()]
            m = folium.Map(location=map_center, zoom_start=13)
            
            # 色の生成
            colors = sns.color_palette('tab10', n_colors=N_CLUSTERS).as_hex()
            
            for _, row in merged_df.iterrows():
                cluster_idx = int(row['cluster']) - 1 # 0-indexed for colors
                color = colors[cluster_idx % len(colors)]
                
                popup_text = (
                    f"Point ID: {row['point_id']}<br>"
                    f"Cluster: {row['cluster']}<br>"
                    f"Most Distinct Dir: {row['most_distinct_direction']}<br>"
                    f"Score: {row['most_distinct_jsd_score']:.3f}"
                )
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(m)
            
            map_path = os.path.join(results_output_dir, 'directional_jsd_cluster_map.html')
            m.save(map_path)
            print(f"地図を保存しました: {map_path}")
        else:
            print("警告: 座標データとのマージ結果が空です。point_idが一致していない可能性があります。")
    
    print("処理完了。")

if __name__ == '__main__':
    main()
