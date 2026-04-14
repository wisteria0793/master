import os
import json
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

# Base directory for the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Use the combined embeddings (StreetCLIP + GNN) evaluated as best
POI_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', 'filtered_facilities.json')
EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', 'embeddings_dim64_combined_mean.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
TEXT_EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'embedding', 'sentence-transformer', 'facility_embeddings.npy')

def load_poi_data():
    """POIデータを読み込み、DataFrameとして返す"""
    with open(POI_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    poi_list = []
    for poi in pois:
        geom = poi.get('google_places_data', {}).get('find_place_geometry', {}).get('location', {})
        lat = geom.get('lat')
        lng = geom.get('lng')
        
        if lat and lng:
            poi_list.append({
                'name': poi.get('name'),
                'address': poi.get('address'),
                'categories': poi.get('categories', []),
                'lat': lat,
                'lng': lng
            })
            
    return pd.DataFrame(poi_list)

def load_and_cluster_embeddings(n_clusters=20):
    """エンベディングを読み込み、階層的クラスタリングを行ってクラスタIDを付与する"""
    embedding_df = pd.read_csv(EMBEDDING_PATH)
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item and item["api_location"]}
    
    embedding_df['lat'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    embedding_df['lng'] = embedding_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    embedding_df.dropna(subset=['lat', 'lng'], inplace=True)
    
    # 階層的クラスタリング (visualize_combined_gnn_clusters.pyと同様)
    feature_cols = embedding_df.columns.drop(['point_id', 'lat', 'lng'])
    features = embedding_df[feature_cols].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, n_clusters, criterion='maxclust')
    
    embedding_df['cluster'] = clusters - 1
    return embedding_df

def get_merged_poi_data(n_clusters=20):
    """POIデータとクラスタリング結果を結合する"""
    poi_df = load_poi_data()
    cluster_df = load_and_cluster_embeddings(n_clusters)
    
    # KDTreeを用いて、各POI周辺の複数Street Viewポイント（景観クラスタ）を検索
    # 緯度経度をラジアンに変換して計算
    cluster_coords = np.radians(cluster_df[['lat', 'lng']].values)
    poi_coords = np.radians(poi_df[['lat', 'lng']].values)
    
    tree = KDTree(cluster_coords)
    
    # K=5 で近傍5点を取得し、その中で最も多いクラスタを採用する（面的な考慮によるノイズ除去）
    k_neighbors = 5
    distances, indices = tree.query(poi_coords, k=k_neighbors)
    
    # 近傍点のクラスタ番号を取得
    nearest_clusters_ids = cluster_df['cluster'].values[indices]
    
    # 各POIに対する多数決クラスタを取得
    majority_clusters = []
    for row in nearest_clusters_ids:
        # np.bincountで出現回数をカウントし、argmaxで最頻値を取得
        majority_cluster = np.bincount(row).argmax()
        majority_clusters.append(majority_cluster)
    
    # 最寄りポイント(1点目)の情報も参考として紐付けつつ、多数決で決定したクラスタを付与
    poi_df['nearest_sv_point_id'] = cluster_df['point_id'].values[indices[:, 0]]
    poi_df['cluster'] = majority_clusters
    poi_df['sv_distance_rad'] = distances[:, 0]  # 参考：最寄り1点との距離(ラジアン)
    
    # エンベディングベクトルも結合（後続のコサイン類似度計算用）
    # ※コサイン類似度計算ではテキストベクトルを使うため、SVのエンベディングは現在は参考情報
    feature_cols = [col for col in cluster_df.columns if col not in ['point_id', 'lat', 'lng', 'cluster']]
    for col in feature_cols:
        poi_df[f'emb_{col}'] = cluster_df[col].values[indices[:, 0]]
        
    # 施設紹介文のSentence-BERTエンベディングを結合 (821件で1対1対応済みの前提)
    text_embeddings = np.load(TEXT_EMBEDDING_PATH)
    poi_df['text_embedding'] = list(text_embeddings)
        
    return poi_df

if __name__ == '__main__':
    # テスト実行用
    print("データを読み込み、クラスタリングと紐付けを行っています...")
    df = get_merged_poi_data(n_clusters=20)
    print("\n--- 結合結果のプレビュー ---")
    print(df[['name', 'cluster', 'nearest_sv_point_id']].head(10))
    print(f"\nTotal POIs processed: {len(df)}")
    print("\n--- クラスタごとのPOI分布 ---")
    print(df['cluster'].value_counts().sort_index())
