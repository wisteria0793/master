# -*- coding: utf-8 -*-
"""
実験18.2: 全域統合モデル（18.0）の結果から、指定された地域の埋め込みベクトルのみを抽出する。
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
import os

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GLOBAL_UNIFIED_CSV = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified' / 'unified_embeddings.csv'
POI_JSON_PATH = PROJECT_ROOT / 'data' / 'processed' / 'poi' / 'filtered_facilities.json'

OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_global_context'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TOWNS = [
    "末広町", "若松町", "東雲町", "新川町", "千歳町", "海岸町", "松川町", "上新川町", 
    "大森町", "松風町", "旭町", "栄町", "宝来町", "元町", "谷地頭町", "青柳町", 
    "住吉町", "弥生町", "大町", "弁天町", "入舟町", "豊川町", "大手町"
]

def main():
    print("1. 全域埋め込みデータを読み込み中...")
    df_global = pd.read_csv(GLOBAL_UNIFIED_CSV)
    
    print("2. POI住所情報を読み込み、フィルタリング対象を特定中...")
    with open(POI_JSON_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    # 町名フィルタリング
    target_names = set()
    for poi in pois:
        addr = poi.get('address', '')
        if any(town in addr for town in TARGET_TOWNS):
            target_names.add(poi["name"])
            
    print(f"   -> ターゲットPOI名数: {len(target_names)}")
    
    # POIの抽出 (名前で一致)
    df_poi_subset = df_global[(df_global['type'] == 'poi') & (df_global['name'].isin(target_names))].copy()
    
    print(f"   -> 抽出されたPOI数: {len(df_poi_subset)}")
    df_sv_all = df_global[df_global['type'] == 'sv'].copy()
    
    # KDTreeで周辺ポイント検索
    poi_coords = df_poi_subset[['lat', 'lng']].values
    sv_coords = df_sv_all[['lat', 'lng']].values
    
    # メートル換算
    poi_m = poi_coords * np.array([111000, 82000])
    sv_m = sv_coords * np.array([111000, 82000])
    
    sv_tree = KDTree(sv_m)
    nearby_sv_indices = set()
    for pm in poi_m:
        indices = sv_tree.query_ball_point(pm, r=500)
        nearby_sv_indices.update(indices)
        
    df_sv_subset = df_sv_all.iloc[list(nearby_sv_indices)].copy()
    print(f"   -> ターゲットSV数: {len(df_sv_subset)}")
    
    # 結合して保存
    df_subset = pd.concat([df_poi_subset, df_sv_subset], ignore_index=True)
    subset_csv = OUTPUT_DIR / 'subset_embeddings.csv'
    df_subset.to_csv(subset_csv, index=False)
    
    print(f"完了。サブセットデータを保存しました: {subset_csv}")

if __name__ == "__main__":
    main()
