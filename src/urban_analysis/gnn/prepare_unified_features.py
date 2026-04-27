# -*- coding: utf-8 -*-
"""
実験18: POI機能データと道路景観データを統合し、統一GNN学習用の入力データを作成する。
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'

# 入力パス
POI_JSON_PATH = PROCESSED_DIR / 'poi' / 'filtered_facilities.json'
TEXT_EMB_PATH = PROCESSED_DIR / 'embedding' / 'sentence-transformer' / 'facility_embeddings.npy'
SV_EMB_PATH = DATA_DIR / 'new' / 'streetclip_embeddings' / 'streetclip_embeddings.npy'
SV_META_PATH = DATA_DIR / 'new' / 'streetclip_embeddings' / 'streetclip_metadata.csv'
PANO_META_PATH = RAW_DIR / 'street_view_images_50m_optimized' / 'pano_metadata.json'

# 出力パス
OUTPUT_DIR = PROCESSED_DIR / 'gnn_unified'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_temporal_features(poi):
    hours_vec = np.zeros(24)
    days_vec = np.zeros(7)
    oh = poi.get('google_places_data', {}).get('details', {}).get('opening_hours', {})
    periods = oh.get('periods')
    if not periods:
        cats = poi.get('categories', [])
        if any(c in str(cats) for c in ['居酒屋', 'バー', '夜']):
            hours_vec[18:24], hours_vec[0:2] = 1, 1
        elif any(c in str(cats) for c in ['朝市', '市場']):
            hours_vec[5:12] = 1
        else:
            hours_vec[9:18] = 1
        days_vec[:] = 1
        return np.concatenate([hours_vec, days_vec])
    for p in periods:
        if 'open' in p:
            d = p['open'].get('day')
            if d is not None: days_vec[d % 7] = 1
            if 'time' in p['open'] and 'close' in p.get('close', {}):
                try:
                    start_h = int(p['open']['time'][:2])
                    end_h = int(p['close']['time'][:2])
                    if end_h < start_h: # 翌日まで
                        hours_vec[start_h:24] = 1
                        hours_vec[0:end_h] = 1
                    else:
                        hours_vec[start_h:end_h] = 1
                except: pass
    return np.concatenate([hours_vec, days_vec])

def main():
    print("1. POIデータを処理中...")
    with open(POI_JSON_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    text_embs = np.load(TEXT_EMB_PATH)
    
    poi_nodes = []
    poi_features = []
    for i, poi in enumerate(pois):
        temp_feat = extract_temporal_features(poi)
        full_feat = np.concatenate([text_embs[i], temp_feat])
        poi_features.append(full_feat)
        
        # 座標の取得を階層化
        loc = poi.get("geometry", {}).get("location", {})
        if not loc:
            # 代替案: google_places_data.details.geometry.location
            loc = poi.get("google_places_data", {}).get("details", {}).get("geometry", {}).get("location", {})
        
        poi_nodes.append({
            "id": poi.get("place_id", f"poi_{i}"),
            "name": poi["name"],
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
            "type": "poi"
        })
    
    print(f"   -> POIノード数: {len(poi_nodes)}")
    
    print("2. 道路景観データを処理中...")
    sv_embs = np.load(SV_EMB_PATH)
    sv_meta_df = pd.read_csv(SV_META_PATH)
    with open(PANO_META_PATH, 'r') as f:
        pano_meta = json.load(f)
    pano_coords = {p["pano_id"]: (p["original_lat"], p["original_lon"]) for p in pano_meta}
    
    # point_idごとに平均化
    sv_meta_df['index'] = sv_meta_df.index
    unique_points = sv_meta_df.groupby('point_id')['index'].apply(list).to_dict()
    
    sv_nodes = []
    sv_features = []
    for pid, indices in unique_points.items():
        if pid not in pano_coords: continue
        
        mean_feat = np.mean(sv_embs[indices], axis=0)
        sv_features.append(mean_feat)
        lat, lng = pano_coords[pid]
        sv_nodes.append({
            "id": pid,
            "name": f"SV_{pid}",
            "lat": lat,
            "lng": lng,
            "type": "sv"
        })
        
    print(f"   -> SVノード数: {len(sv_nodes)}")
    
    # 保存
    np.save(OUTPUT_DIR / 'poi_features.npy', np.array(poi_features))
    np.save(OUTPUT_DIR / 'sv_features.npy', np.array(sv_features))
    
    with open(OUTPUT_DIR / 'nodes_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(poi_nodes + sv_nodes, f, ensure_ascii=False, indent=2)
        
    print(f"完了。データは {OUTPUT_DIR} に保存されました。")

if __name__ == "__main__":
    main()
