import requests
import json
import os
from io import BytesIO
from PIL import Image  # pip install pillow
from dotenv import load_dotenv

# .envファイルのパスを指定して読み込む
load_dotenv('.env')

def get_place_details(api_key, place_id):
    """
    Google Places APIを使って特定の場所の詳細情報（レビューを含む）を取得する関数
    
    Parameters:
        api_key (str): Google Places APIのAPIキー
        place_id (str): 特定の場所のPlace ID
    
    Returns:
        dict: 場所の詳細情報
    """
    endpoint_url = "https://maps.googleapis.com/maps/api/place/details/json"
    
    params = {
        'place_id': place_id,
        'fields': 'name,formatted_address,geometry,rating,formatted_phone_number,website,opening_hours,price_level,review,photos',
        'language': 'ja',
        'key': api_key
    }
    
    response = requests.get(endpoint_url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"APIリクエストエラー: {response.status_code}", "details": response.text}

def get_place_photo(api_key, photo_reference, max_width=800):
    """
    Google Places APIを使って場所の写真を取得する関数
    
    Parameters:
        api_key (str): Google Places APIのAPIキー
        photo_reference (str): 写真の参照ID
        max_width (int): 写真の最大幅
    
    Returns:
        bytes: 写真データ (バイナリ)
    """
    endpoint_url = "https://maps.googleapis.com/maps/api/place/photo"
    
    params = {
        'photoreference': photo_reference,
        'maxwidth': max_width,
        'key': api_key
    }
    
    response = requests.get(endpoint_url, params=params)
    
    if response.status_code == 200:
        return response.content
    else:
        print(f"写真の取得に失敗: {response.status_code}")
        return None

def save_place_photos(api_key, photos, place_name, max_photos=3):
    """
    場所の写真を保存する関数
    
    Parameters:
        api_key (str): Google Places APIのAPIキー
        photos (list): 写真情報のリスト
        place_name (str): 場所の名前（ファイル名のプレフィックス用）
        max_photos (int): 保存する写真の最大数
    
    Returns:
        list: 保存された写真のファイルパスリスト
    """
    saved_files = []
    
    # 保存用のディレクトリを作成
    os.makedirs("place_photos", exist_ok=True)
    
    # 安全なファイル名を作成
    safe_place_name = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in place_name)
    safe_place_name = safe_place_name.replace(' ', '_')
    
    for i, photo in enumerate(photos[:max_photos]):
        photo_reference = photo.get('photo_reference')
        if photo_reference:
            photo_data = get_place_photo(api_key, photo_reference)
            if photo_data:
                file_path = f"place_photos/{safe_place_name}_{i+1}.jpg"
                with open(file_path, 'wb') as f:
                    f.write(photo_data)
                saved_files.append(file_path)
                print(f"写真を保存しました: {file_path}")
                
                # PILを使用して画像を表示（オプション）
                try:
                    img = Image.open(BytesIO(photo_data))
                    img.show()  # 画像を表示（デスクトップ環境の場合）
                except Exception as e:
                    print(f"画像の表示に失敗しました: {e}")
    
    return saved_files

def search_place_by_text(api_key, query, location=None, radius=None):
    """
    テキスト検索で場所を検索する関数
    
    Parameters:
        api_key (str): Google Places APIのAPIキー
        query (str): 検索クエリ
        location (tuple, optional): 緯度と経度のタプル (lat, lng)
        radius (int, optional): 検索範囲（メートル単位）
    
    Returns:
        dict: 検索結果
    """
    endpoint_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    
    params = {
        'query': query,
        'language': 'ja',
        'revies_limits': 10,
        'key': api_key
    }
    
    if location:
        params['location'] = f"{location[0]},{location[1]}"
        
    if radius:
        params['radius'] = radius
    
    response = requests.get(endpoint_url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"APIリクエストエラー: {response.status_code}", "details": response.text}

def display_reviews(reviews, max_reviews=3):
    """
    レビューを表示する関数
    
    Parameters:
        reviews (list): レビューのリスト
        max_reviews (int): 表示するレビューの最大数
    """
    if not reviews:
        print("レビューはありません")
        return
    
    print(f"\n=== レビュー ({min(len(reviews), max_reviews)}/{len(reviews)}) ===")
    for i, review in enumerate(reviews[:max_reviews]):
        print(f"\nレビュー {i+1}:")
        print(f"評価者: {review.get('author_name', '名前なし')}")
        print(f"評価: {review.get('rating', 'なし')} / 5")
        print(f"日時: {review.get('relative_time_description', 'なし')}")
        
        # レビューテキストの表示（長すぎる場合は省略）
        review_text = review.get('text', 'テキストなし')
        if len(review_text) > 200:
            review_text = review_text[:200] + "..."
        print(f"コメント: {review_text}")



if __name__ == "__main__":
    API_KEY = os.getenv('GOOGLE_PLACES_API')    # Google Places APIkeyの読み込み

    
    # テキスト検索の例
    search_query = "函館山"  # 検索したい場所を指定
    place_search = search_place_by_text(API_KEY, search_query)
    
    if "results" in place_search and len(place_search["results"]) > 0:
        place = place_search["results"][0]
        print(f"検索結果: {place['name']}, Place ID: {place['place_id']}")
        
        # 見つかったPlace IDを使って詳細情報を取得
        place_details = get_place_details(API_KEY, place['place_id'])
        
        if "result" in place_details:
            result = place_details["result"]
            print(f"\n=== {result.get('name')} の詳細情報 ===")
            print(f"住所: {result.get('formatted_address', 'なし')}")
            if 'geometry' in result and 'location' in result['geometry']:
                location = result['geometry']['location']
                print(f"座標: 緯度 {location.get('lat')}, 経度 {location.get('lng')}")
            print(f"評価: {result.get('rating', 'なし')}")
            print(f"電話番号: {result.get('formatted_phone_number', 'なし')}")
            print(f"ウェブサイト: {result.get('website', 'なし')}")
            
            # レビューの取得と表示
            reviews = result.get('reviews', [])
            print(f"number of reviews: {len(reviews)}")
            display_reviews(reviews, max_reviews=3)
            
    else:
        print(f"検索結果がありません: {search_query}")