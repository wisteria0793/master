import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .data_loader import get_merged_poi_data

# プロトタイプの推薦関連パラメータ
DISTANCE_THRESHOLD_M = 3000  # 10件抽出のため少し広めに設定（3km）
TOP_N_RECOMMENDATIONS = 10     # 推薦する最大施設数

def haversine_distance(coord1, coord2):
    """Haversine法による2点間の距離計算 (メートル)"""
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c # 地球半径(m)

class RouteRecommender:
    def __init__(self, poi_df=None, n_clusters=20):
        """
        推薦エンジンの初期化
        """
        if poi_df is None:
            self.poi_df = get_merged_poi_data(n_clusters=n_clusters)
        else:
            self.poi_df = poi_df
        
    def recommend(self, target_poi_name, distance_threshold=DISTANCE_THRESHOLD_M, top_n=TOP_N_RECOMMENDATIONS):
        """
        基準となるPOI名から、徒歩圏内かつ意味的・景観的類似度が高いPOIを推薦する
        （同一クラスタに限定せず、類似度主体のランキングを行う）
        """
        # 1. 基準となるPOIを検索
        target_rows = self.poi_df[self.poi_df['name'].str.contains(target_poi_name, na=False)]
        if target_rows.empty:
            raise ValueError(f"指定されたPOI '{target_poi_name}' は見つかりませんでした。")
            
        target_poi = target_rows.iloc[0]
        target_cluster = target_poi['cluster']
        target_coords = (target_poi['lat'], target_poi['lng'])
        target_embedding = np.array(target_poi['text_embedding']).reshape(1, -1)
        
        # 2. 距離フィルタリング (まず歩行可能圏内に絞る)
        temp_df = self.poi_df.copy()
        
        # Haversine法を使用
        temp_df['distance_m'] = temp_df.apply(
            lambda row: haversine_distance(target_coords, (row['lat'], row['lng'])), axis=1
        )
        dist_filtered_df = temp_df[temp_df['distance_m'] <= distance_threshold].copy()
        
        # 自身を除外
        dist_filtered_df = dist_filtered_df[dist_filtered_df['name'] != target_poi['name']]
        
        if dist_filtered_df.empty:
            print(f"警告: {distance_threshold}m以内に対象施設が見つかりませんでした。")
            return pd.DataFrame()
            
        # 3. コサイン類似度による意味的ランキング（エンベディングを使用）
        candidate_embeddings = np.vstack(dist_filtered_df['text_embedding'].values)
        similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]
        
        # 景観クラスタによるボーナスは廃止し、純粋にテキストの類似度で評価する
        dist_filtered_df['similarity_score'] = similarities
        
        # 類似度が高い順にソートしてTop Nを取得
        recommended_df = dist_filtered_df.sort_values(by='similarity_score', ascending=False).head(top_n)
        
        return recommended_df

if __name__ == "__main__":
    # テスト実行
    recommender = RouteRecommender()
    test_poi = "函館山"
    print(f"\n基準POI: {test_poi}")
    try:
        results = recommender.recommend(test_poi)
        if not results.empty:
            print(f"--- 推薦結果 (上位 {TOP_N_RECOMMENDATIONS}件, 距離≦{DISTANCE_THRESHOLD_M}m) ---")
            for idx, row in results.iterrows():
                print(f"施設名: {row['name']}")
                print(f"距離: {row['distance_m']:.1f} m | 類似度: {row['similarity_score']:.3f} | クラスタ: {row['cluster']}")
                print("---")
        else:
            print("条件に合致する推薦POIが見つかりませんでした。")
    except Exception as e:
        print(e)
