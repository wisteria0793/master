# -*- coding: utf-8 -*-
"""
実験18.1: 指定された町名に属するPOIと、その周辺のStreetViewポイントを抽出する。
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

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
OUTPUT_DIR = PROCESSED_DIR / 'gnn_unified_filtered'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ターゲット町名
TARGET_TOWNS = [
    "末広町", "若松町", "東雲町", "新川町", "千歳町", "海岸町", "松川町", "上新川町", 
    "大森町", "松風町", "旭町", "栄町", "宝来町", "元町", "谷地頭町", "青柳町", 
    "住吉町", "弥生町", "大町", "弁天町", "入舟町", "豊川町", "大手町"
]

def extract_temporal_features(poi):
    hours_vec = np.zeros(24)
    days_vec = np.zeros(7)
    oh = poi.get('google_places_data', {}).get('details', {}).get('opening_hours', {})
    periods = oh.get('periods', [])
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
                    if end_h < start_h:
                        hours_vec[start_h:24] = 1
                        hours_vec[0:end_h] = 1
                    else:
                        hours_vec[start_h:end_h] = 1
                except: pass
    return np.concatenate([hours_vec, days_vec])

def main():
    print("1. POIデータをフィルタリング中...")
    with open(POI_JSON_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    text_embs = np.load(TEXT_EMB_PATH)
    
    filtered_indices = []
    filtered_nodes = []
    filtered_features = []
    
    for i, poi in enumerate(pois):
        addr = poi.get('address', '')
        if any(town in addr for town in TARGET_TOWNS):
            filtered_indices.append(i)
            temp_feat = extract_temporal_features(poi)
            full_feat = np.concatenate([text_embs[i], temp_feat])
            filtered_features.append(full_feat)
            
            loc = poi.get("geometry", {}).get("location", {})
            if not loc:
                loc = poi.get("google_places_data", {}).get("details", {}).get("geometry", {}).get("location", {})
            
            filtered_nodes.append({
                "id": poi.get("place_id", f"poi_{i}"),
                "name": poi["name"],
                "lat": loc.get("lat"),
                "lng": loc.get("lng"),
                "type": "poi",
                "address": addr
            })
            
    print(f"   -> 抽出されたPOI数: {len(filtered_nodes)} (元の {len(pois)} 地点中)")
    
    if not filtered_nodes:
        print("エラー: 条件に合うPOIが見つかりませんでした。")
        return

    print("2. 周辺のStreetViewポイントを抽出中...")
    sv_embs = np.load(SV_EMB_PATH)
    sv_meta_df = pd.read_csv(SV_META_PATH)
    with open(PANO_META_PATH, 'r') as f:
        pano_meta = json.load(f)
    pano_coords_map = {p["pano_id"]: (p["original_lat"], p["original_lon"]) for p in pano_meta}
    
    # SVポイントの座標リスト作成
    sv_ids = []
    sv_full_coords = []
    for pid in sv_meta_df['point_id'].unique():
        if pid in pano_coords_map:
            sv_ids.append(pid)
            sv_full_coords.append(pano_coords_map[pid])
            
    sv_coords_arr = np.array(sv_full_coords)
    poi_coords_arr = np.array([[n['lat'], n['lng']] for n in filtered_nodes])
    
    # KDTreeでPOIから500m以内のSVポイントを特定
    # 函館付近の簡易メートル変換
    sv_m = sv_coords_arr * np.array([111000, 82000])
    poi_m = poi_coords_arr * np.array([111000, 82000])
    
    sv_tree = KDTree(sv_m)
    # 各POIから500m以内のSVポイントを探す
    nearby_sv_indices = set()
    for p_m in poi_m:
        indices = sv_tree.query_ball_point(p_m, r=500)
        nearby_sv_indices.update(indices)
        
    print(f"   -> 周辺500m以内に {len(nearby_sv_indices)} 地点のStreetViewポイントを特定しました。")
    
    # 特定されたSVの平均ベクトル計算
    unique_points_group = sv_meta_df.groupby('point_id').indices
    
    final_sv_nodes = []
    final_sv_features = []
    
    for idx in nearby_sv_indices:
        pid = sv_ids[idx]
        if pid not in unique_points_group: continue
        
        indices_in_npy = unique_points_group[pid]
        mean_feat = np.mean(sv_embs[indices_in_npy], axis=0)
        final_sv_features.append(mean_feat)
        lat, lng = sv_full_coords[idx]
        final_sv_nodes.append({
            "id": pid,
            "name": f"SV_{pid}",
            "lat": lat,
            "lng": lng,
            "type": "sv"
        })
        
    # 保存
    np.save(OUTPUT_DIR / 'poi_features.npy', np.array(filtered_features))
    np.save(OUTPUT_DIR / 'sv_features.npy', np.array(final_sv_features))
    
    with open(OUTPUT_DIR / 'nodes_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_nodes + final_sv_nodes, f, ensure_ascii=False, indent=2)
        
    print(f"完了。データは {OUTPUT_DIR} に保存されました。")

if __name__ == "__main__":
    main()
