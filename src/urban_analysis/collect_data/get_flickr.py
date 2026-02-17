import flickrapi
import json
import urllib.request
import os
import configparser
import datetime
import time 
import sys
import pprint
from dotenv import load_dotenv

# .envファイルのパスを指定して読み込む
load_dotenv('.env')




# --- 1. 環境変数からAPIキーとシークレットの読み込み ---


# --- 1. 定数と設定 ---
HAKODATE_BBOX = '140.5,41.7,141.0,41.9' 
PER_PAGE = 500  
DOWNLOAD_DIR = './data/raw/hakodate_all_photos'
MAX_RETRIES = 3         
START_PAGE_NUMBER = 1   

# ✅ レート制限対策のコア設定: 1時間3600回 (1回/秒) の制限を確実に下回るための間隔
RATE_LIMIT_DELAY = 1.05 
INNER_LOOP_DELAY = RATE_LIMIT_DELAY # 写真1枚あたりの待機時間として適用


api_key = os.getenv('FLICKR_API_KEY')
api_secret = os.getenv('FLICKR_SECRET_KEY')

    
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json') 

# --- 3. フォルダの準備 ---
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

print(f"--- 函館市内（bbox: {HAKODATE_BBOX}）の全件収集を開始します ---")

# ==============================================================================
# ユーティリティ関数
# ==============================================================================

def get_photo_comments(photo_id):
    """指定された写真IDのコメントを取得し、JSON保存用のリストで返します。"""
    
    # ✅ レート制限対策: コメント取得APIコール直前に待機
    time.sleep(RATE_LIMIT_DELAY) 
    
    try:
        comments_response = flickr.photos.comments.getList(photo_id=photo_id)
        comments_data = json.loads(comments_response.decode('utf-8'))
        
        comment_count = 0
        comment_list = []

        if comments_data['comments'].get('comment'):
            comments = comments_data['comments']['comment']
            if not isinstance(comments, list): comments = [comments]
            comment_count = len(comments)
            for comment in comments:
                comment_list.append({
                    'author_name': comment['authorname'],
                    'content': comment['_content']
                })
        
        print(f"       コメント取得成功: {comment_count}件 (間隔 {RATE_LIMIT_DELAY}秒)")
        return comment_list
    except Exception as e:
        print(f"       コメント取得失敗: エラーが発生しました ({e})")
        return []


def fetch_page_data(current_page):
    """Flickr APIからページデータを取得し、失敗時にリトライします。"""
    
    params = {
        'bbox': HAKODATE_BBOX,
        'per_page': PER_PAGE,
        'page': current_page,
        'has_geo': 1,
        'safe_search': 1,
        'extras': 'url_m,date_taken,owner_name,geo,tags,secret,description,views,date_upload,count_faves',
        'sort': 'date-posted-desc'
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            # ✅ レート制限対策: ページ取得APIコール直前に待機 (初回はスキップされる可能性あり)
            if current_page > 1 or attempt > 0:
                time.sleep(RATE_LIMIT_DELAY) 
                
            response = flickr.photos.search(**params)
            data = json.loads(response.decode('utf-8'))
            
            if data.get('stat') == 'fail':
                raise Exception(f"Flickr API Status Fail: {data.get('message', 'Unknown Error')}")

            print(f"   [API SUCCESS] ページ {current_page} のデータ取得に成功しました。")
            return data 
            
        except Exception as e:
            print(f"\n⚠️ リクエスト失敗 (ページ {current_page}, 試行 {attempt + 1}/{MAX_RETRIES}): {e}")
            
            if attempt < MAX_RETRIES - 1:
                wait_time = 5 * (attempt + 1)
                # エラー時の待機時間はレート制限とは別で、サーバー回復を待つ時間
                print(f"   {wait_time}秒待機してからリトライします...")
                time.sleep(wait_time)
            else:
                print(f"   リトライ回数を超過しました。このページはスキップします。")
                return None
    return None


# ==============================================================================
# メイン処理 (全件収集ロジック)
# ==============================================================================
try:
    # --- 4. 最初の検索: 総枚数と総ページ数を取得 ---
    # 初回リクエストは RATE_LIMIT_DELAY の適用外として、すぐに実行を試みる (fetch_page_data内で待機する可能性あり)
    initial_data = fetch_page_data(1) 
    
    if initial_data is None:
        print("初回リクエストに失敗しました。プログラムを終了します。")
        sys.exit(1)

    photos_meta = initial_data['photos']
    total_photos = int(photos_meta['total'])
    total_pages = int(photos_meta['pages'])
    
    print(f"💡 合計 {total_photos} 件の写真が見つかりました。（全 {total_pages} ページ）")
    if START_PAGE_NUMBER > 1:
         print(f"✅ 処理をページ {START_PAGE_NUMBER} から再開します。\n")
    else:
         print("\n")

    # --- ページネーションのループ ---
    for current_page in range(START_PAGE_NUMBER, total_pages + 1):
        
        # 処理する写真リストを取得
        if current_page == 1:
            current_photos = photos_meta['photo']
        else:
            print(f"   --> ページ {current_page}/{total_pages} のデータを取得中...")
            # fetch_page_data内で既に待機処理が含まれている
            page_data = fetch_page_data(current_page)
            
            if page_data is None:
                continue 
            current_photos = page_data['photos']['photo']
        
        
        print(f"   --- ページ {current_page}/{total_pages} を処理中 ({len(current_photos)}枚) ---")
        
        # --- 写真ごとの詳細処理（JSON保存とダウンロード） ---
        for i, photo in enumerate(current_photos):
            photo_id = photo['id']
            image_url = photo.get('url_m')
            
            # --- メタデータ収集と整形 ---
            date_upload_unix = photo.get('dateupload', 'N/A')
            date_upload_readable = 'N/A'
            if date_upload_unix != 'N/A' and date_upload_unix.isdigit():
                date_upload_readable = datetime.datetime.fromtimestamp(int(date_upload_unix)).strftime('%Y-%m-%d %H:%M:%S')

            # コメントの取得 (API呼び出し #2) - この関数内で RATE_LIMIT_DELAY が実行される
            comments_list = get_photo_comments(photo_id) 

            # JSON格納データ生成 (省略)
            photo_metadata = {
                'id': photo_id, 'secret': photo.get('secret', 'N/A'),
                'title': photo.get('title', 'N/A'), 'owner_name': photo.get('ownername', 'N/A'),
                'url_page': f"https://www.flickr.com/photos/{photo['owner']}/{photo_id}", 'url_image_m': image_url,
                'datetime': {'taken': photo.get('datetaken', 'N/A'), 'uploaded_unix': date_upload_unix, 'uploaded_readable': date_upload_readable,},
                'stats': {'views': photo.get('views', '0'), 'faves': photo.get('count_faves', '0'),},
                'location': {'latitude': photo.get('latitude', 'N/A'), 'longitude': photo.get('longitude', 'N/A'), 'accuracy': photo.get('accuracy', 'N/A'),},
                'tags': photo.get('tags', 'N/A').split(' '), 'description': photo.get('description', {}).get('_content', 'N/A'),
                'comments': comments_list
            }

            # 簡潔なステータス表示
            print(f"     [{i+1}/{len(current_photos)}] ID: {photo_id[:10]}... | Title: {photo.get('title', 'N/A')[:30]}...")

            # --- JSONファイルの保存 ---
            json_filename = os.path.join(DOWNLOAD_DIR, f"{photo_id}.json")
            if not os.path.exists(json_filename): 
                try:
                    with open(json_filename, 'w', encoding='utf-8') as f:
                        json.dump(photo_metadata, f, ensure_ascii=False, indent=4)
                    print(f"       ✅ JSON保存: 成功")
                except Exception as json_e:
                    print(f"       ❌ JSON保存: 失敗 ({json_e})")
            else:
                print(f"       ➡️ JSON保存: スキップ (ファイルが存在)")

            
            # --- 画像ファイルのダウンロード ---
            if image_url:
                image_filename = os.path.join(DOWNLOAD_DIR, f"{photo_id}_{photo.get('secret', 'no_secret')}.jpg")
                if not os.path.exists(image_filename): 
                    try:
                        urllib.request.urlretrieve(image_url, image_filename)
                        print(f"       ✅ 画像ダウンロード: 成功")
                    except Exception as dl_e:
                        print(f"       ❌ 画像ダウンロード: 失敗 ({dl_e})")
                else:
                    print(f"       ➡️ 画像ダウンロード: スキップ (ファイルが存在)")
            
            # ページリクエストとコメントリクエストの間に、追加のAPIコールがないため、
            # この位置での sleep は不要になりました。sleep は API コールの直前で行われます。
            pass 
        
    print(f"\n✅ 全 {total_photos} 件の写真の処理が完了しました。")

except Exception as e:
    print(f"\n❌ 予期せぬエラーが発生しました: {e}")