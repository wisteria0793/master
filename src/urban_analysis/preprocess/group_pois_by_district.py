# 各地区ごとに存在するPOIを抽出

import json
import re
from collections import defaultdict

def extract_district(address):
    """住所文字列から町名を抽出する"""
    if not address or '函館市' not in address:
        return None
    # 「函館市」の直後から「町」までの文字列を抽出する
    match = re.search(r'函館市(.+?町)', address)
    if match:
        return match.group(1).strip()
    return None

def group_pois_by_district(file_path, output_path):
    """施設データを町名ごとにグループ化してJSONファイルに出力する"""
    pois_by_district = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_facilities = json.load(f)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {file_path}")
        return
    except json.JSONDecodeError:
        print(f"エラー: ファイルのJSON形式が正しくありません - {file_path}")
        return

    for facility in all_facilities:
        # google_places_data内の住所を優先して使用し、なければトップレベルのaddressを使用
        address = facility.get('google_places_data', {}).get('find_place_formatted_address')
        if not address:
            address = facility.get('address', '')

        district = extract_district(address)
        
        if district:
            name = facility.get('name')
            place_id = facility.get('google_places_data', {}).get('place_id')
            
            if name and place_id:
                pois_by_district[district].append({
                    'name': name,
                    'place_id': place_id
                })

    # 結果をJSONファイルに書き出す
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pois_by_district, f, ensure_ascii=False, indent=4)
        print(f"処理が完了しました。結果は {output_path} に出力されました。")
    except IOError as e:
        print(f"エラー: ファイルの書き込みに失敗しました - {e}")

if __name__ == "__main__":
    INPUT_FILE_PATH = '../../data/processed/filtered_facilities.json'
    OUTPUT_FILE_PATH = '../../data/processed/pois_by_district.json'
    group_pois_by_district(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
