import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import sys

# プロジェクトのルートディレクトリ
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

POI_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', 'filtered_facilities.json')
TEXT_EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'embedding', 'sentence-transformer', 'facility_embeddings.npy')

# ... (load_poi_data function remains the same)

def load_poi_data():
    """POIデータを読み込み、DataFrameとして返す"""
    with open(POI_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    poi_list = []
    valid_indices = [] # 保持するPOIの元のインデックスを記録
    
    # 除外するキーワードのリスト
    exclude_keywords = ["閉店", "休業", "休館"]
    
    for i, poi in enumerate(pois):
        name = poi.get('name', '')
        
        # 除外キーワードが含まれているかチェック
        if any(keyword in name for keyword in exclude_keywords):
            continue
            
        geom = poi.get('google_places_data', {}).get('find_place_geometry', {}).get('location', {})
        lat = geom.get('lat')
        lng = geom.get('lng')
        
        if lat and lng:
            poi_list.append({
                'name': name,
                'address': poi.get('address'),
                'categories': ', '.join(poi.get('categories', [])),
                'lat': lat,
                'lng': lng,
                'original_index': i # 元のインデックスを保持
            })
            valid_indices.append(i)
            
    return pd.DataFrame(poi_list), valid_indices

def main(n_clusters=20):
    OUTPUT_CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', f'poi_text_clusters_{n_clusters}.csv')
    OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, 'docs', 'results', f'poi_text_dendrogram_{n_clusters}.png')

    print("POIデータと施設紹介文ベクトル (Sentence-BERT) を読み込み中...")
    poi_df, valid_indices = load_poi_data()
    all_embeddings = np.load(TEXT_EMBEDDING_PATH)
    
    # 除外されなかったPOIに対応するエンベディングのみを抽出
    embeddings = all_embeddings[valid_indices]
    
    print(f"元のPOI数: {len(all_embeddings)} -> フィルタリング後のPOI数: {len(poi_df)}")
        
    print(f"階層的クラスタリング (Ward法, k={n_clusters}) を実行中...")
    linked = linkage(embeddings, method='ward')
    
    # クラスタの割り当て
    clusters = fcluster(linked, n_clusters, criterion='maxclust')
    poi_df['text_cluster'] = clusters - 1
    
    # CSV保存
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    poi_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"CSV結果を保存しました: {OUTPUT_CSV_PATH}")

    # デンドログラムの描画
    print("デンドログラムを生成中...")
    plt.figure(figsize=(15, 8))
    dendrogram(linked,
               truncate_mode='lastp',  # 最後のp個のクラスタのみ表示
               p=n_clusters,           # 指定したクラスタ数でカット
               leaf_rotation=90.,
               leaf_font_size=12.,
               show_contracted=True)
    plt.title(f'Hierarchical Clustering Dendrogram (POI Text Embeddings, n_clusters={n_clusters})')
    plt.xlabel('Cluster size (or index if leaf)')
    plt.ylabel('Distance (Ward)')
    
    # 画像保存
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plt.savefig(OUTPUT_IMAGE_PATH)
    plt.close()
    print(f"デンドログラム画像を保存しました: {OUTPUT_IMAGE_PATH}")
    
    print("\n--- テキストクラスタごとのPOI数 ---")
    print(poi_df['text_cluster'].value_counts().sort_index())
    
    # 各クラスタの代表的なPOIを少しだけ表示
    print("\n--- 各クラスタのサンプルPOI ---")
    for i in range(n_clusters):
        sample_pois = poi_df[poi_df['text_cluster'] == i]['name'].head(3).tolist()
        print(f"クラスタ {i:2d}: {', '.join(sample_pois)}")

if __name__ == '__main__':
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n_clusters)