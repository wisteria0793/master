# -*- coding: utf-8 -*-
import numpy as np
import json
import os
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
from math import radians, sin, cos, sqrt, atan2

# --- 設定項目 ---

# 表示するトップNの数
TOP_N = 30

# POIからの最大距離（メートル）。この範囲内の画像のみが分析対象となる
MAX_DISTANCE_METERS = 50

# --- ヘルパー関数 ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """2つの緯度経度座標間の距離をメートル単位で計算する（ハーバーサイン公式）"""
    R = 6371000  # 地球の半径（メートル）
    
    # 座標がNoneでないことを確認
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

# --- メインロジック ---

def inspect_top_sim_with_distance(project_root, top_n, max_distance):
    """
    指定された距離範囲内のペアの中から、コサイン類似度が最も高いトップNを表示する。
    """
    print("類似度上位ペアの調査を開始します...")
    print(f"設定: トップN={top_n}, 画像の最大距離={max_distance}m")

    # --- 1. パス定義 ---
    poi_data_path = os.path.join(project_root, 'data', 'processed', 'poi', 'filtered_facilities.json')
    facility_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'facility_embeddings.npy')
    image_emb_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'image_embeddings.npy')
    image_filenames_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_filenames.json')
    image_gps_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_gps_data.json')

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
    num_facilities = facility_embeddings.shape[0]
    num_images = image_embeddings.shape[0]
    print(f"施設数: {num_facilities}, 画像数: {num_images}")
    
    image_gps_list = [image_gps_data.get(fname) for fname in image_filenames]

    # --- 4. 類似度上位ペアの探索 ---
    # 効率化のため、最小ヒープ（min-heap）を使用して上位N件を保持する
    top_pairs_heap = []

    print(f"全ペア（最大約{num_facilities * num_images}件）を計算します。これには数分かかる場合があります...")
    for i in tqdm.tqdm(range(num_facilities), desc="施設を処理中"):
        poi_info = poi_data[i]
        poi_location = poi_info.get('google_places_data', {}).get('find_place_geometry', {}).get('location')

        if not poi_location:
            continue

        text_embedding = facility_embeddings[i]

        for j, image_gps in enumerate(image_gps_list):
            if image_gps and 'lat' in image_gps and 'lon' in image_gps:
                # 距離を計算し、範囲外ならスキップ
                distance = haversine_distance(poi_location['lat'], poi_location['lng'], image_gps['lat'], image_gps['lon'])
                if distance > max_distance:
                    continue
                
                # 距離範囲内の場合のみ、類似度を計算
                image_embedding = image_embeddings[j]
                score = cosine_similarity(text_embedding.reshape(1, -1), image_embedding.reshape(1, -1))[0][0]
                
                # ヒープを使って効率的にトップNを保持
                if len(top_pairs_heap) < top_n:
                    heapq.heappush(top_pairs_heap, (score, i, j, distance))
                elif score > top_pairs_heap[0][0]:
                    heapq.heappushpop(top_pairs_heap, (score, i, j, distance))

    # --- 5. 結果表示 ---
    if not top_pairs_heap:
        print(f"\n警告: 指定された距離（{max_distance}m）内に、分析対象となるペアが一つも見つかりませんでした。")
        return

    # 類似度が高い順にソートして表示
    sorted_top_pairs = sorted(top_pairs_heap, key=lambda x: x[0], reverse=True)

    print(f"\n--- 類似度トップ{top_n}（距離{max_distance}m以内）のテキスト・画像ペア ---")
    for rank, (score, facility_idx, image_idx, dist) in enumerate(sorted_top_pairs, 1):
        poi_name = poi_data[facility_idx].get('name', '不明なPOI')
        image_name = image_filenames[image_idx]
        
        print(f"{rank:>2}. Score: {score:.4f} | Dist: {dist:.1f}m | POI: \"{poi_name}\" | Image: \"{image_name}\"")

if __name__ == '__main__':
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    try:
        import sklearn
        import tqdm
    except ImportError:
        print("エラー: 必要なライブラリが見つかりません。")
        print("pip install scikit-learn tqdm でインストールしてください。")
    else:
        inspect_top_sim_with_distance(project_root_dir, TOP_N, MAX_DISTANCE_METERS)