import os
import json
import requests
import time
import sys
import math
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm library not found. Please install with 'pip install tqdm'")

def haversine_distance(p1, p2):
    """2つの緯度経度座標間の距離をメートルで計算する（ハーベサイン公式）"""
    R = 6371000  # 地球の半径（メートル）
    lat1, lon1 = p1
    lat2, lon2 = p2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def interpolate(p1, p2, fraction):
    """2つの緯度経度座標間を線形補間する"""
    lat = p1[0] + (p2[0] - p1[0]) * fraction
    lon = p1[1] + (p2[1] - p1[1]) * fraction
    return (lat, lon)

def check_and_add_pano(point, api_key, unique_panos, download_queue, metadata_url, min_distance):
    """
    指定座標でメタデータを確認し、ユニークかつ十分に離れていればキューに追加する
    戻り値: 'added', 'skipped', 'not_found'
    """
    lat, lon = point
    location = f"{lat},{lon}"
    params = {'location': location, 'radius': 50, 'source': 'outdoor', 'key': api_key}
    
    try:
        response = requests.get(metadata_url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') == 'OK':
            pano_id = data.get('pano_id')
            if not pano_id or pano_id in unique_panos:
                return 'skipped' # 重複するpano_id

            api_location = (data['location']['lat'], data['location']['lng'])
            
            # 最小距離チェック
            for item in download_queue:
                if haversine_distance(api_location, item['api_location']) < min_distance:
                    return 'skipped' # 近すぎる

            # すべてのチェックをクリア
            unique_panos.add(pano_id)
            download_queue.append({
                'pano_id': pano_id,
                'original_lat': point[0],
                'original_lon': point[1],
                'api_location': api_location
            })
            return 'added'
            
        elif data.get('status') != 'ZERO_RESULTS':
            tqdm.write(f"\nAPI Error for {location}: {data.get('status')}")
            
    except requests.exceptions.RequestException as e:
        tqdm.write(f"\nRequest failed: {e}")

    return 'not_found'

def fetch_street_view_images():
    """
    Google Street View画像を収集する高度なスクリプト。
    距離・重複チェック、補間探索、tqdmによる進捗表示、ログ出力を行う。
    """
    load_dotenv('.env')

    # --- Configuration ---
    test_mode = os.getenv("SV_TEST_MODE", 'true').lower() == 'true'
    TEST_DOWNLOAD_LIMIT = 3
    TEST_ROAD_LIMIT = 50

    INPUT_FILE_PATH = 'data/processed/road_points/road_points_per_way_50m_filtered.json'
    if not os.path.exists(INPUT_FILE_PATH):
        INPUT_FILE_PATH = 'road_points_per_way_50m.json'

    if test_mode:
        print(f"--- Running in TEST MODE: Will process {TEST_ROAD_LIMIT} roads and stop after {TEST_DOWNLOAD_LIMIT} images. ---")
        OUTPUT_DIR = './data/raw/street_view_images_test'
    else:
        OUTPUT_DIR = './data/raw/street_view_images_50m_optimized'
    
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'fetch_log.json')

    IMAGE_SIZE = "640x640"
    HEADINGS = [0, 90, 180, 270]
    METADATA_DELAY = 0.05
    DOWNLOAD_DELAY = 0.2
    INTERPOLATION_STEPS = 4
    MIN_DISTANCE_METERS = 10

    # --- API Key & Preparations ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("!!! ERROR: 'GOOGLE_API_KEY' not found in env.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            roads_data = list(json.load(f).values())
        print(f"Loaded {len(roads_data)} total roads from {INPUT_FILE_PATH}.")
    except FileNotFoundError:
        sys.exit(f"ERROR: Input file not found at {INPUT_FILE_PATH}")

    if test_mode:
        roads_data = roads_data[:TEST_ROAD_LIMIT]

    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    static_url = "https://maps.googleapis.com/maps/api/streetview"

    # ---------------------------------------------------------
    # Phase 1: Metadata Check with Progress Bar and Logging
    # ---------------------------------------------------------
    print("\n--- Phase 1: Metadata Check (Free) ---")
    print(f"Searching with min distance rule: {MIN_DISTANCE_METERS}m. Progress will be logged.")

    unique_panos = set()
    download_queue = []
    check_log = []

    for road in tqdm(roads_data, desc="Processing Roads"):
        if not road: continue

        for j in range(len(road) - 1):
            if test_mode and len(download_queue) >= TEST_DOWNLOAD_LIMIT: break
            start_point, end_point = road[j], road[j+1]
            
            time.sleep(METADATA_DELAY)
            result = check_and_add_pano(start_point, api_key, unique_panos, download_queue, metadata_url, MIN_DISTANCE_METERS)
            check_log.append({'lat': start_point[0], 'lon': start_point[1], 'type': 'main', 'status': result})

            if result != 'not_found':
                continue
            
            # 補間探索
            for k in range(1, INTERPOLATION_STEPS + 1):
                if test_mode and len(download_queue) >= TEST_DOWNLOAD_LIMIT: break
                
                inter_point = interpolate(start_point, end_point, k / (INTERPOLATION_STEPS + 1))
                time.sleep(METADATA_DELAY)
                inter_result = check_and_add_pano(inter_point, api_key, unique_panos, download_queue, metadata_url, MIN_DISTANCE_METERS)
                check_log.append({'lat': inter_point[0], 'lon': inter_point[1], 'type': 'interpolated', 'status': inter_result})
                
                if inter_result == 'added':
                    break
        
        if test_mode and len(download_queue) >= TEST_DOWNLOAD_LIMIT: break
        
        if road and not (test_mode and len(download_queue) >= TEST_DOWNLOAD_LIMIT):
            time.sleep(METADATA_DELAY)
            final_result = check_and_add_pano(road[-1], api_key, unique_panos, download_queue, metadata_url, MIN_DISTANCE_METERS)
            check_log.append({'lat': road[-1][0], 'lon': road[-1][1], 'type': 'final', 'status': final_result})

    print(f"\n\nPhase 1 Complete. Total unique panoramas found: {len(download_queue)}")
    
    # ログファイルの保存
    print(f"Saving detailed log to {LOG_FILE_PATH}...")
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(check_log, f, indent=2)

    if not download_queue: sys.exit("No valid panoramas found. Exiting.")

    # ---------------------------------------------------------
    # Phase 2: Image Download (Billable)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Image Download (Billable) ---")
    
    if not test_mode:
        final_download_count = len(download_queue) * len(HEADINGS)
        user_consent = input(f"!!! Ready to download {final_download_count} images. Proceed? (yes/no): ")
        if user_consent.lower() != 'yes': sys.exit("Cancelled.")

    count = 0
    for item in tqdm(download_queue, desc="Downloading Images"):
        if test_mode and count >= TEST_DOWNLOAD_LIMIT:
            tqdm.write(f"\n--- TEST MODE COMPLETE: Download limit of {TEST_DOWNLOAD_LIMIT} reached. ---")
            break

        pano_id = item['pano_id']
        
        for heading in HEADINGS:
            params = {'size': IMAGE_SIZE, 'pano': pano_id, 'heading': heading, 'fov': 90, 'key': api_key}
            try:
                response = requests.get(static_url, params=params, timeout=15)
                
                if response.status_code == 200:
                    filename = f"pano_{pano_id}_h{heading}.jpg"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, 'wb') as f: f.write(response.content)
                else:
                    tqdm.write(f"Error {response.status_code} for pano {pano_id}")
            except Exception as e:
                tqdm.write(f"Download failed for {pano_id}: {e}")
            time.sleep(DOWNLOAD_DELAY)
        count += 1

    # --- --- --- --- ---
    # Final step: Save the download queue as metadata for the downloaded images
    # --- --- --- --- ---
    METADATA_FILE_PATH = os.path.join(OUTPUT_DIR, 'pano_metadata.json')
    print(f"\nSaving panorama metadata to {METADATA_FILE_PATH}...")
    try:
        with open(METADATA_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(download_queue, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved metadata for {len(download_queue)} panoramas.")
    except Exception as e:
        print(f"Error saving metadata file: {e}", file=sys.stderr)


    print("\n--- Process Completed ---")

if __name__ == "__main__":
    fetch_street_view_images()