# -*- coding: utf-8 -*-
"""
実験18.4: アプローチB - POIのテキスト埋め込み（E5）のみを使用したクラスタリング。
空間的な近接性や営業時間を一切考慮せず、純粋な機能的類似性で分類する。
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'approach_b'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS = 20

def main():
    print("POI特徴量データを読み込み中...")
    poi_feats = np.load(INPUT_DIR / 'poi_features.npy')
    with open(INPUT_DIR / 'nodes_metadata.json', 'r', encoding='utf-8') as f:
        nodes_meta = json.load(f)
    
    # POIノードのみを抽出
    poi_meta = [n for n in nodes_meta if n['type'] == 'poi']
    print(f"対象POI数: {len(poi_meta)}")
    
    # テキスト埋め込み（最初の768次元）のみを抽出
    # prepare_filtered_unified_features.py では [text_emb, temp_feat] の順で結合されている
    text_features = poi_feats[:, :768]
    
    print(f"テキスト（機能）のみのクラスタリングを実行中 (k={N_CLUSTERS})...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(text_features)
    
    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, N_CLUSTERS, criterion='maxclust')
    
    # 結果の保存
    df_results = pd.DataFrame(poi_meta)
    df_results['text_cluster_id'] = clusters - 1
    
    output_path = OUTPUT_DIR / 'poi_text_only_clusters.csv'
    df_results.to_csv(output_path, index=False)
    print(f"クラスタリング結果を保存しました: {output_path}")
    
    # 各クラスタの主要キーワード（簡易表示）
    print("\n【テキスト（機能）クラスタの傾向】")
    for cid in range(N_CLUSTERS):
        cluster_names = df_results[df_results['text_cluster_id'] == cid]['name'].tolist()
        print(f"Cluster {cid:02d}: {', '.join(cluster_names[:5])}...")

if __name__ == "__main__":
    main()
