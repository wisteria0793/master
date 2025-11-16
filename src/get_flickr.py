import flickrapi
import json
import pprint # 結果を見やすく表示するために使用
import os
from dotenv import load_dotenv


# .envファイルのパスを指定して読み込む
load_dotenv('.env')




# --- 1. 環境変数からAPIキーとシークレットの読み込み ---
api_key = os.getenv('FLICKR_API_KEY')
api_secret = os.getenv('FLICKR_SECRET_KEY')

# --- 2. 函館の検索座標と設定 ---
# 函館市役所付近の緯度・経度を使用
HAKODATE_LAT = 41.7687
HAKODATE_LON = 140.7289
SEARCH_RADIUS_KM = 10 # 検索範囲を10kmに設定
PER_PAGE = 5  # 取得する写真の数を5枚に制限

# --- 3. Flickr APIクライアントの初期化 ---
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json')

print(f"--- 函館周辺 ({HAKODATE_LAT}, {HAKODATE_LON}, 半径{SEARCH_RADIUS_KM}km) の写真検索を開始します ---")

try:
    # --- 4. flickr.photos.searchメソッドで写真を検索 ---
    # extrasパラメータで、写真のURLや撮影日などの追加情報を取得
    response = flickr.photos.search(
        lat=HAKODATE_LAT,
        lon=HAKODATE_LON,
        radius=SEARCH_RADIUS_KM, # 検索範囲（km）
        per_page=PER_PAGE,        # 1ページあたりの写真枚数
        has_geo=1,                # 地理情報を持つ写真のみを検索
        safe_search=1,            # セーフサーチ（安全なコンテンツのみ）
        extras='url_m,date_taken,owner_name,geo', # 取得したい追加情報 (url_m: 中サイズの画像URL)
        sort='date-posted-desc'   # 投稿日順（新しい順）
    )

    # JSONレスポンスをPythonオブジェクトに変換
    data = json.loads(response.decode('utf-8'))

    # --- 5. 結果の処理 ---
    if data['stat'] == 'ok':
        photos = data['photos']['photo']
        total_photos = data['photos']['total']

        print(f"**合計 {total_photos} 件** の写真が見つかりました。（うち {len(photos)} 件を表示）\n")

        for i, photo in enumerate(photos):
            print(f"--- 写真 {i+1} ---")
            print(f"  タイトル: {photo.get('title', 'N/A')}")
            print(f"  撮影者: {photo.get('ownername', 'N/A')}")
            print(f"  撮影日: {photo.get('datetaken', 'N/A')}")
            print(f"  緯度: {photo.get('latitude', 'N/A')}")
            print(f"  経度: {photo.get('longitude', 'N/A')}")

            # 画像URL (url_m) があれば表示
            if 'url_m' in photo:
                print(f"  画像URL (中サイズ): {photo['url_m']}")
            else:
                print("  画像URL: 利用可能なサイズがありません。")

            # オリジナルのFlickrページURL
            photo_url = f"https://www.flickr.com/photos/{photo['owner']}/{photo['id']}"
            print(f"  Flickrページ: {photo_url}\n")
    else:
        print(f"エラーが発生しました: {data['message']}")

except Exception as e:
    print(f"予期せぬエラー: {e}")