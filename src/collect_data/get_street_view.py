import os
import json
import requests
import time
import sys
from dotenv import load_dotenv

def fetch_street_view_images():
    """
    Google Street View画像を効率的に収集するスクリプト
    1. Metadata API (無料) で画像の有無とPano IDを確認・重複除外
    2. Static API (有料) で有効なPano IDを使って画像を確実に入手
    """
    # .env ファイルから環境変数を読み込む
    load_dotenv('.env')

    # --- Test Mode Configuration ---
    # テストモード: デフォルトをTrueにして、1枚取得で終了するように設定
    test_mode = os.getenv("SV_TEST_MODE", 'true').lower() == 'true'
    
    # 画像ダウンロードを行う件数の上限（テスト時は1枚で終了）
    TEST_DOWNLOAD_LIMIT = 1
    
    # Metadataチェックを行う件数の上限（画像が見つかるまで探すので少し多めに設定）
    TEST_METADATA_SEARCH_LIMIT = 20

    # --- Configuration ---
    # パスの設定（環境に合わせて調整してください）
    INPUT_FILE_PATH = 'data/processed/road_points/road_points_50m_filtered.json'
    
    # ファイルが見つからない場合のフォールバック（カレントディレクトリ）
    if not os.path.exists(INPUT_FILE_PATH) and os.path.exists('road_points_50m_filtered.json'):
        INPUT_FILE_PATH = 'road_points_50m_filtered.json'

    if test_mode:
        print(f"--- Running in TEST MODE: Will stop after saving {TEST_DOWNLOAD_LIMIT} image(s). ---")
        OUTPUT_DIR = './data/raw/street_view_images_test'
    else:
        OUTPUT_DIR = './data/raw/street_view_images_50m_optimized'
    
    IMAGE_SIZE = "640x640"
    HEADINGS = [0] # 必要であれば [0, 90, 180, 270] のように追加
    
    METADATA_DELAY = 0.05 
    DOWNLOAD_DELAY = 0.2

    # --- 1. API Key Check ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("!!! ERROR: 'GOOGLE_API_KEY' not found in env.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Preparations ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
        
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            points = json.load(f)
        print(f"Loaded {len(points)} total coordinates from {INPUT_FILE_PATH}.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_FILE_PATH}", file=sys.stderr)
        sys.exit(1)

    # テストモード時は検索範囲も絞る
    if test_mode:
        points = points[:TEST_METADATA_SEARCH_LIMIT]

    # API Endpoints
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    static_url = "https://maps.googleapis.com/maps/api/streetview"

    # ---------------------------------------------------------
    # Phase 1: Metadata Check (Free)
    # 重複するPano IDを除外し、存在する画像のリストを作成する
    # ---------------------------------------------------------
    print("\n--- Phase 1: Metadata Check (Free) ---")
    print("Checking availability and filtering duplicates...")

    unique_panos = set()
    download_queue = []

    for i, point in enumerate(points):
        lat, lon = point
        location = f"{lat},{lon}"
        
        params = {
            'location': location,
            'radius': 50,  
            'source': 'outdoor', # 屋内画像を除外
            'key': api_key
        }

        try:
            response = requests.get(metadata_url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'OK':
                pano_id = data.get('pano_id')
                
                if pano_id and pano_id not in unique_panos:
                    unique_panos.add(pano_id)
                    download_queue.append({
                        'pano_id': pano_id,
                        'original_lat': lat,
                        'original_lon': lon
                    })
                    print(f"\rFound: {len(unique_panos)} | Checked: {i+1}/{len(points)}", end="")
                    
                    # ★テストモード最適化: 
                    # ダウンロード対象が1つでも見つかれば、これ以上の探索をやめてPhase 2へ進む
                    if test_mode and len(download_queue) >= TEST_DOWNLOAD_LIMIT:
                        print(f"\nTest target found. Proceeding to download phase.")
                        break

            elif data.get('status') == 'ZERO_RESULTS':
                # 画像がない場合は単にスキップ
                pass
            else:
                print(f"\nMetadata Error for {location}: {data.get('status')}")

        except requests.exceptions.RequestException as e:
            print(f"\nRequest failed: {e}")
        
        time.sleep(METADATA_DELAY)

    print(f"\nPhase 1 Complete. Valid Panoramas found: {len(download_queue)}")
    
    if len(download_queue) == 0:
        print("No valid panoramas found in the tested range. Exiting.")
        sys.exit(0)

    # ---------------------------------------------------------
    # Phase 2: Image Download (Billable)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Image Download (Billable) ---")
    
    if not test_mode:
        user_consent = input(f"!!! Ready to download {len(download_queue) * len(HEADINGS)} images. Proceed? (yes/no): ")
        if user_consent.lower() != 'yes':
            print("Cancelled.")
            sys.exit(0)
    else:
        print("Test mode active: Automatically proceeding with limited download.")

    count = 0
    
    for item in download_queue:
        pano_id = item['pano_id']
        
        for heading in HEADINGS:
            params = {
                'size': IMAGE_SIZE,
                'pano': pano_id, # 正確なIDを指定
                'heading': heading,
                'fov': 90,
                'key': api_key
            }

            try:
                print(f"Downloading pano {pano_id} (h={heading})...")
                response = requests.get(static_url, params=params, timeout=15)
                
                if response.status_code == 200:
                    filename = f"pano_{pano_id}_h{heading}.jpg"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    count += 1
                    print(f"SUCCESS: Saved {filename}")
                    
                    # ★テストモード終了条件:
                    # 指定枚数（1枚）保存したらスクリプトを終了する
                    if test_mode and count >= TEST_DOWNLOAD_LIMIT:
                        print(f"\n--- TEST MODE COMPLETE: Limit of {TEST_DOWNLOAD_LIMIT} image reached. Exiting. ---")
                        sys.exit(0)

                else:
                    print(f"Error {response.status_code} for pano {pano_id}")

            except Exception as e:
                print(f"Download failed for {pano_id}: {e}")

            time.sleep(DOWNLOAD_DELAY)

    print("\n--- Process Completed ---")

if __name__ == "__main__":
    fetch_street_view_images()