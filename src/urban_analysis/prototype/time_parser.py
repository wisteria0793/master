import json
import re
from pathlib import Path

def parse_time_to_minutes(time_str):
    """ 'HH:MM' 形式の文字列を、0時からの経過分数に変換 """
    if not time_str:
        return None
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return None

def extract_time_windows(json_path):
    """
    Google Places API等のJSONデータから営業時間を抽出し、
    {poi_name: (open_time_minutes, close_time_minutes)} の辞書を返す。
    解析できない場合は (0, 1440) をデフォルトとする。
    """
    path = Path(json_path)
    if not path.exists():
        return {}
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    time_windows = {}
    pattern = re.compile(r'(\d{1,2}):(\d{2})')
    
    for p in data:
        name = p.get('name', '')
        if not name:
            continue
            
        hours_str = p.get('hours', '')
        if hours_str:
            matches = pattern.findall(hours_str)
            if matches:
                times = [int(h) * 60 + int(m) for h, m in matches]
                open_t = min(times)
                close_t = max(times)
                
                # 24時（24:00）越えの表記（例: 26:00 -> 26*60）にも対応可能
                time_windows[name] = (open_t, close_t)
            else:
                time_windows[name] = (0, 1440) # デフォルト: 24時間営業扱い
        else:
            time_windows[name] = (0, 1440) # デフォルト: 24時間営業扱い
            
    return time_windows

if __name__ == "__main__":
    # テスト
    test_path = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'poi' / 'filtered_facilities_final.json'
    tw = extract_time_windows(test_path)
    print(f"Loaded {len(tw)} POI time windows.")
    sample_keys = list(tw.keys())[:5]
    for k in sample_keys:
        print(f"{k}: {tw[k]}")
