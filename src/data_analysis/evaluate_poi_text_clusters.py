import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEXT_EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'embedding', 'sentence-transformer', 'facility_embeddings.npy')

def main(n_clusters=20):
    CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', f'poi_text_clusters_{n_clusters}.csv')

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} が見つかりません。")
        return

    df = pd.read_csv(CSV_PATH)
    all_embeddings = np.load(TEXT_EMBEDDING_PATH)
    
    # フィルタリング済みのインデックスを用いてエンベディングを取得
    valid_indices = df['original_index'].values
    embeddings = all_embeddings[valid_indices]
    labels = df['text_cluster'].values

    # 1. 意味的距離（コサイン類似度ベース）のシルエットスコア
    # Sentence-BERTの本来の性能を評価するためコサイン距離を使用
    semantic_score = silhouette_score(embeddings, labels, metric='cosine')
    
    # 2. 物理的距離（Haversine距離ベース）のシルエットスコア
    # 緯度経度をラジアンに変換してhaversine距離を計算
    coords_rad = np.radians(df[['lat', 'lng']].values)
    physical_score = silhouette_score(coords_rad, labels, metric='haversine')
    
    print(f"=== POI テキストクラスタリング (Sentence-BERT, k={n_clusters}) のシルエットスコア評価 ===")
    print(f"対象データ数: {len(df)} 件")
    print(f"実クラスタ数: {len(np.unique(labels))} 個")
    print("-" * 65)
    print(f"1. 意味的距離 (Sentence-BERTベクトル / コサイン距離)")
    print(f"   Score: {semantic_score:.4f}")
    print("   => 1に近いほど、各クラスタが「意味的（機能・紹介文の文脈）」によくまとまっており、")
    print("      他のクラスタと明確に分離できていることを示します（クラスタリングの品質）。")
    print("-" * 65)
    print(f"2. 物理的距離 (緯度・経度 / Haversine距離)")
    print(f"   Score: {physical_score:.4f}")
    print("   => 1に近いほど「特定のエリアに同じ意味のPOIが固まっている」ことを示し、")
    print("      0やマイナスに近いほど「同じ意味のPOIが函館市内に分散している」ことを示します。")
    print("      （※今回はテキストでクラスタリングしたため、この値が低いのは自然な結果です）")
    print("=================================================================")

if __name__ == '__main__':
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n_clusters)