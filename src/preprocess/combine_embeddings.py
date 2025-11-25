# -*- coding: utf-8 -*-
import json
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

# --- 設定項目 ---
# Alpha: テキストと画像の重要度のバランスを調整
# 1.0に近づくほどテキストを重視し、0.0に近づくほど画像を重視する
ALPHA = 0.7

# Sigma (メートル): 地理的重みの減衰度合いを制御
# 値が小さいほど、より近くの画像のみが強い影響力を持つ
SIGMA = 200.0 

# 類似度のしきい値: この値未満の類似度を持つ画像は「無関係」とみなし、計算から除外する
SIMILARITY_THRESHOLD = 0.2418

# 最大距離（メートル）: ここで指定した距離より遠い画像は計算から除外する
MAX_DISTANCE_METERS = 200

# --- ヘルパー関数 ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """2つの緯度経度座標間の距離をメートル単位で計算する（ハーバーサイン公式）"""
    R = 6371000  # 地球の半径（メートル）
    
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return float('inf')

    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

def gaussian_weight(distance, sigma):
    """ガウス関数に基づき、距離から重みを計算する"""
    return np.exp(-distance**2 / (2 * sigma**2))

# --- メインロジック ---

def combine_embeddings(project_root, alpha, sigma, similarity_threshold, max_distance):
    """
    類似度でフィルタリングした後、地理的距離で重み付けを行い、
    テキストと画像の埋め込みを合成する。
    """
    print("埋め込み合成プロセスを開始します...")
    print(f"パラメータ: alpha={alpha}, sigma={sigma}m, similarity_threshold={similarity_threshold}, max_distance={max_distance}m")

    # --- 1. パス定義 ---
    poi_data_path = os.path.join(project_root, 'data', 'processed', 'poi', 'filtered_facilities.json')
    facility_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'facility_embeddings.npy')
    image_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'image_embeddings.npy')
    image_filenames_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_filenames.json')
    image_gps_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_gps_data.json')
    output_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'combined_facility_embeddings_02418_07.npy')

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
        return

    # --- 3. データ準備 ---
    image_gps_list = [image_gps_data.get(fname) for fname in image_filenames]
    new_facility_embeddings = []

    print(f"{len(poi_data)}件の施設を処理します...")
    # --- 4. メイン処理ループ ---
    for i, facility in enumerate(tqdm.tqdm(poi_data, desc="施設を処理中")):
        text_embedding = facility_embeddings[i]
        poi_location = facility.get('google_places_data', {}).get('find_place_geometry', {}).get('location')

        if not poi_location:
            new_facility_embeddings.append(text_embedding)
            continue

        # --- 4a. 類似度による画像のフィルタリング ---
        text_embedding_reshaped = text_embedding.reshape(1, -1)
        all_sim_scores = cosine_similarity(text_embedding_reshaped, image_embeddings).flatten()
        
        candidate_indices = np.where(all_sim_scores > similarity_threshold)[0]
        
        if len(candidate_indices) == 0:
            new_facility_embeddings.append(text_embedding)
            continue

        # --- 4b. 候補画像の地理的重みを計算 ---
        geo_weights = []
        final_candidate_indices = []
        for img_idx in candidate_indices:
            gps_info = image_gps_list[img_idx]
            if gps_info and 'lat' in gps_info and 'lon' in gps_info:
                distance = haversine_distance(poi_location['lat'], poi_location['lng'], gps_info['lat'], gps_info['lon'])
                if distance <= max_distance:
                    geo_weights.append(gaussian_weight(distance, sigma))
                    final_candidate_indices.append(img_idx)

        if not final_candidate_indices:
            new_facility_embeddings.append(text_embedding)
            continue

        # --- 4c. 地理的重みを正規化 ---
        weight_sum = np.sum(geo_weights)
        if weight_sum > 0:
            normalized_weights = np.array(geo_weights) / weight_sum
        else:
            new_facility_embeddings.append(text_embedding)
            continue

        # --- 4d. 合成画像ベクトルを生成 ---
        candidate_embeddings = image_embeddings[final_candidate_indices]
        weights_reshaped = normalized_weights.reshape(-1, 1)
        combined_image_embedding = np.sum(candidate_embeddings * weights_reshaped, axis=0)

        # --- 4e. 最終的なベクトルを合成 ---
        new_embedding = alpha * text_embedding + (1 - alpha) * combined_image_embedding
        new_facility_embeddings.append(new_embedding)

    # --- 5. 結果を保存 ---
    final_embeddings_array = np.array(new_facility_embeddings)
    np.save(output_path, final_embeddings_array)

    print("\n--- 合成完了 ---")
    print(f"{final_embeddings_array.shape[0]}件の新しい埋め込みベクトルを生成しました。")
    print(f"保存先: {output_path}")

if __name__ == '__main__':
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    try:
        import sklearn
        import tqdm
    except ImportError:
        print("エラー: 必要なライブラリ 'scikit-learn' または 'tqdm' が見つかりません。")
        print("pip install scikit-learn tqdm でインストールしてください。")
    else:
        combine_embeddings(project_root_dir, ALPHA, SIGMA, SIMILARITY_THRESHOLD, MAX_DISTANCE_METERS)
