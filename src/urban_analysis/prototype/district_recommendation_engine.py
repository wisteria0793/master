# -*- coding: utf-8 -*-
"""
統合地区ベース観光推薦エンジン (v1)
起点となるPOIを指定すると、同じ統合地区内のPOIを推薦し、
その地区の景観・機能的特徴に基づいた推薦理由を提示する。
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import cosine

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
NODES_INFO_CSV = PROJECT_ROOT / 'data' / 'processed' / 'final_unified_nodes.csv'
POI_JSON = PROJECT_ROOT / 'data' / 'processed' / 'poi' / 'filtered_facilities_final.json'
POI_EMB_NPY = PROJECT_ROOT / 'data' / 'processed' / 'embedding' / 'facility_embeddings_final.npy'

class DistrictRecommendEngine:
    def __init__(self):
        print("エンジンを初期化中...")
        # 1. 地点情報のロード
        self.df_nodes = pd.read_csv(NODES_INFO_CSV)
        self.df_poi_nodes = self.df_nodes[self.df_nodes['type'] == 'poi'].copy()
        
        # 2. メタデータのロード
        with open(POI_JSON, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            self.poi_meta = {}
            for i, p in enumerate(raw_data):
                # IDがない場合は、CSVと整合させるためにインデックスベースのIDを使用
                pid = p.get('id', f'poi_{i}')
                p['id'] = pid
                self.poi_meta[pid] = p
        
        # 3. ベクトルのロード (意味的類似度用)
        self.poi_embeddings = np.load(POI_EMB_NPY)
        # IDからインデックスへのマッピング
        self.id_to_idx = {row['id']: i for i, row in self.df_poi_nodes.iterrows()}

    def get_recommendations(self, start_poi_id, top_n=5):
        if start_poi_id not in self.id_to_idx:
            return f"Error: POI ID '{start_poi_id}' が見つかりません。"
        
        start_node = self.df_poi_nodes[self.df_poi_nodes['id'] == start_poi_id].iloc[0]
        cluster_id = start_node['unified_cluster']
        start_emb_idx = self.id_to_idx[start_poi_id]
        start_vec = self.poi_embeddings[start_emb_idx]
        
        print(f"起点: {self.poi_meta[start_poi_id]['name']} (統合地区: {cluster_id})")
        
        # 同じクラスター内の他のPOIを抽出
        candidates = self.df_poi_nodes[
            (self.df_poi_nodes['unified_cluster'] == cluster_id) & 
            (self.df_poi_nodes['id'] != start_poi_id)
        ].copy()
        
        if candidates.empty:
            return "同じ地区内に他の候補POIが見つかりませんでした。"
        
        # スコアリング（コサイン類似度）
        scores = []
        for i, row in candidates.iterrows():
            target_idx = self.id_to_idx[row['id']]
            target_vec = self.poi_embeddings[target_idx]
            sim = 1 - cosine(start_vec, target_vec)
            scores.append(sim)
        
        candidates['score'] = scores
        recommendations = candidates.sort_values('score', ascending=False).head(top_n)
        
        # 地区の特徴を分析 (説明用)
        # 地区内の景観構成を取得
        ls_in_cluster = self.df_nodes[
            (self.df_nodes['unified_cluster'] == cluster_id) & 
            (self.df_nodes['type'] == 'ls')
        ]
        top_ls_labels = ls_in_cluster['cluster'].value_counts().head(2).index.tolist()
        
        # 結果の整形
        results = {
            "start_poi": self.poi_meta[start_poi_id]['name'],
            "district_id": int(cluster_id),
            "district_context": f"この地区は景観タイプ {top_ls_labels} が特徴的なエリアです。",
            "recommendations": []
        }
        
        for _, row in recommendations.iterrows():
            meta = self.poi_meta[row['id']]
            desc = meta.get('description', '説明なし')
            if isinstance(desc, list):
                desc = " ".join(desc)
            
            results["recommendations"].append({
                "id": row['id'],
                "name": meta['name'],
                "similarity": round(row['score'], 3),
                "description": desc[:100] + '...'
            })
            
        return results

if __name__ == "__main__":
    engine = DistrictRecommendEngine()
    
    # テスト実行 (例: 函館朝市周辺などがあれば)
    test_id = 'place_id:ChIJ-Y-m9gEAdV8R_D396N3Tstk' # 適当なID。実際には存在するIDを指定
    # 最初のPOI IDでテスト
    first_id = engine.df_poi_nodes.iloc[0]['id']
    res = engine.get_recommendations(first_id)
    print(json.dumps(res, ensure_ascii=False, indent=2))
