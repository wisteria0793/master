# -*- coding: utf-8 -*-
"""
最適化されたクラスタ数によるラベル付与スクリプト
景観(k=12)と機能(k=13)をそれぞれの最新特徴量に対して適用し、結果をCSVとして保存する。
"""

import numpy as np
import pandas as pd
import json
from sklearn.cluster import KMeans
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
STREETCLIP_EMB = PROJECT_ROOT / 'data' / 'new' / 'streetclip_embeddings' / 'streetclip_embeddings.npy'
STREETCLIP_META = PROJECT_ROOT / 'data' / 'new' / 'streetclip_embeddings' / 'streetclip_metadata.csv'
FACILITY_EMB = PROJECT_ROOT / 'data' / 'processed' / 'embedding' / 'facility_embeddings_final.npy'
FACILITY_JSON = PROJECT_ROOT / 'data' / 'processed' / 'poi' / 'filtered_facilities_final.json'

OUTPUT_LS = PROJECT_ROOT / 'data' / 'processed' / 'refined_landscape_clusters_k12.csv'
OUTPUT_POI = PROJECT_ROOT / 'data' / 'processed' / 'refined_poi_clusters_k8.csv'

def main():
    print("最適化されたクラスタリングを開始します...")

    # 1. 景観データのクラスタリング (k=12)
    if STREETCLIP_EMB.exists() and STREETCLIP_META.exists():
        print("景観データの処理中 (k=12)...")
        ls_emb = np.load(STREETCLIP_EMB)
        df_ls_meta = pd.read_csv(STREETCLIP_META)
        
        # ※ metadata は 15987 行あるが、embeddings も同じか確認
        if len(ls_emb) == len(df_ls_meta):
            kmeans_ls = KMeans(n_clusters=12, random_state=42, n_init=10)
            ls_labels = kmeans_ls.fit_predict(ls_emb)
            
            df_ls_meta['cluster'] = ls_labels
            # 必要最小限の列で保存
            df_ls_meta[['point_id', 'angle', 'direction', 'cluster']].to_csv(OUTPUT_LS, index=False)
            print(f"景観クラスターを保存しました: {OUTPUT_LS}")
        else:
            print(f"Warning: Landscape dimensions mismatch ({len(ls_emb)} vs {len(df_ls_meta)})")
    
    # 2. 機能データのクラスタリング (k=8)
    if FACILITY_EMB.exists() and FACILITY_JSON.exists():
        print("機能データの処理中 (k=8)...")
        poi_emb = np.load(FACILITY_EMB)
        with open(FACILITY_JSON, 'r', encoding='utf-8') as f:
            poi_data = json.load(f)
            
        if len(poi_emb) == len(poi_data):
            # 新しい統計的最適値 K=8 を採用
            kmeans_poi = KMeans(n_clusters=8, random_state=42, n_init=10)
            poi_labels = kmeans_poi.fit_predict(poi_emb)
            
            # 結果用データフレーム作成
            poi_results = []
            for i, f in enumerate(poi_data):
                # 階層の深い位置から座標を取得
                geo = f.get('google_places_data', {}).get('find_place_geometry', {}).get('location', {})
                poi_results.append({
                    'id': f.get('id', f'poi_{i}'),
                    'name': f.get('name'),
                    'lat': geo.get('lat'),
                    'lng': geo.get('lng'),
                    'cluster': poi_labels[i]
                })
            
            pd.DataFrame(poi_results).to_csv(OUTPUT_POI, index=False)
            print(f"機能クラスターを保存しました: {OUTPUT_POI}")
        else:
            print(f"Warning: POI dimensions mismatch ({len(poi_emb)} vs {len(poi_data)})")

if __name__ == "__main__":
    main()
