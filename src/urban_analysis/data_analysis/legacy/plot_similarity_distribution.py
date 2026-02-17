# -*- coding: utf-8 -*-
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
from math import radians, sin, cos, sqrt, atan2

# --- 設定項目 ---

# 類似度を計算する施設のサンプル数
SAMPLE_SIZE = 50

# POIからの最大距離（メートル）。この範囲内の画像のみが分析対象となる
MAX_DISTANCE_METERS = 200

# --- ヘルパー関数 ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """2つの緯度経度座標間の距離をメートル単位で計算する（ハーバーサイン公式）"""
    R = 6371000  # 地球の半径（メートル）
    
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

# --- メインロジック ---

def plot_similarity_distribution_with_distance_filter(project_root, sample_size, max_distance):
    """
    指定された距離範囲内の画像のみを対象として、テキストと画像のコサイン類似度を計算し、
    その分布をヒストグラムとしてプロット・保存する。
    """
    print("類似度分布の分析を開始します...")
    print(f"設定: 施設サンプル数={sample_size}, 画像の最大距離={max_distance}m")

    # --- 1. パス定義 ---
    poi_data_path = os.path.join(project_root, 'data', 'processed', 'poi', 'filtered_facilities.json')
    facility_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'facility_embeddings.npy')
    image_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'image_embeddings.npy')
    image_filenames_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_filenames.json')
    image_gps_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_gps_data.json')
    output_image_path = os.path.join(project_root, 'images', 'similarity_distribution_with_distance.png')

    # --- 2. データ読み込み ---
    print("データと埋め込みファイルを読み込んでいます...")
    try:
        with open(poi_data_path, 'r', encoding='utf-8') as f:
            poi_data = json.load(f)
        with open(image_filenames_path, 'r', encoding='utf-8') as f:
            image_filenames = json.load(f)
        with open(image_gps_path, 'r', encoding='utf-8') as f:
            image_gps_data = json.load(f)
        facility_embeddings = np.load(facility_emb_path)
        image_embeddings = np.load(image_emb_path)
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりませんでした。 {e}")
        print("これまでの手順が全て正常に完了しているか確認してください。")
        return

    # --- 3. データ準備 ---
    num_facilities = facility_embeddings.shape[0]
    print(f"施設数: {num_facilities}, 画像数: {image_embeddings.shape[0]}")
    
    image_gps_list = [image_gps_data.get(fname) for fname in image_filenames]

    # --- 4. 類似度計算 ---
    if num_facilities > sample_size:
        print(f"分析のため、施設を{sample_size}件サンプリングします...")
        facility_indices = random.sample(range(num_facilities), sample_size)
    else:
        print("全施設を分析対象とします。")
        facility_indices = list(range(num_facilities))

    all_similarities = []
    pois_with_images_in_range = 0
    
    print("コサイン類似度を計算中...")
    for i in tqdm.tqdm(facility_indices, desc="施設を処理中"):
        poi_info = poi_data[i]
        text_embedding = facility_embeddings[i]
        poi_location = poi_info.get('google_places_data', {}).get('find_place_geometry', {}).get('location')

        if not poi_location:
            continue

        # 指定距離内の画像をフィルタリング
        nearby_image_indices = []
        for img_idx, gps_info in enumerate(image_gps_list):
            if gps_info and 'lat' in gps_info and 'lon' in gps_info:
                distance = haversine_distance(poi_location['lat'], poi_location['lng'], gps_info['lat'], gps_info['lon'])
                if distance <= max_distance:
                    nearby_image_indices.append(img_idx)
        
        if not nearby_image_indices:
            continue  # このPOIの近くには対象画像がないのでスキップ

        pois_with_images_in_range += 1
        
        # 近くの画像のみを対象に類似度を計算
        nearby_image_embeddings = image_embeddings[nearby_image_indices]
        sims = cosine_similarity(text_embedding.reshape(1, -1), nearby_image_embeddings)
        all_similarities.extend(sims.flatten())

    if not all_similarities:
        print(f"\n警告: 指定された距離（{max_distance}m）内に、分析対象となる画像を持つ施設が一つも見つかりませんでした。")
        print("MAX_DISTANCE_METERSの値を大きくして再試行してください。")
        return

    # --- 5. 分布をプロット ---
    print("グラフをプロット中...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.histplot(all_similarities, bins=100, kde=True, ax=ax)
    
    title = f'Text-Image Cosine Similarity Distribution (Images within {max_distance}m)'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    mean_sim = np.mean(all_similarities)
    ax.axvline(mean_sim, color='r', linestyle='--', label=f'Mean ({mean_sim:.3f})')
    ax.legend()
    
    # --- 6. グラフを保存 ---
    try:
        fig.savefig(output_image_path, dpi=300)
        print(f"\nグラフを保存しました: {output_image_path}")
    except IOError as e:
        print(f"エラー: グラフの保存に失敗しました - {e}")

    # --- 7. パーセンタイル値を計算して表示 ---
    p50 = np.percentile(all_similarities, 50)
    p75 = np.percentile(all_similarities, 75)
    p90 = np.percentile(all_similarities, 90)
    p95 = np.percentile(all_similarities, 95)

    print("\n--- 類似度スコアのパーセンタイル値 ---")
    print(f"分析対象POI数: {pois_with_images_in_range}/{len(facility_indices)}")
    print(f"総類似度ペア数: {len(all_similarities)}")
    print(f"中央値 (50パーセンタイル): {p50:.4f}")
    print(f"上位25%%のしきい値 (75パーセンタイル): {p75:.4f}")
    print(f"上位10%%のしきい値 (90パーセンタイル): {p90:.4f}")
    print(f"上位5%%のしきい値 (95パーセンタイル): {p95:.4f}")

if __name__ == '__main__':
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    try:
        import matplotlib
        import seaborn
        import sklearn
        import tqdm
    except ImportError:
        print("エラー: 必要なライブラリが見つかりません。")
        print("pip install matplotlib seaborn scikit-learn tqdm でインストールしてください。")
    else:
        plot_similarity_distribution_with_distance_filter(project_root_dir, SAMPLE_SIZE, MAX_DISTANCE_METERS)
