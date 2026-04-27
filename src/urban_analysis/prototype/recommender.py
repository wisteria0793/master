import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from .data_loader import get_merged_poi_data

# プロトタイプの推薦関連パラメータ
DISTANCE_THRESHOLD_M = 3000
TOP_N_RECOMMENDATIONS = 10

def haversine_distance(coord1, coord2):
    """Haversine法による2点間の距離計算 (メートル)"""
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c # 地球半径(m)

class RouteRecommender:
    def __init__(self, poi_df=None):
        """推薦エンジンの初期化"""
        if poi_df is None:
            # 最新のGNN統合データを読み込み
            self.poi_df, self.street_df = get_merged_poi_data()
        else:
            self.poi_df = poi_df
        
    def is_currently_open(self, poi_row, target_time=None):
        """指定された時間（デフォルトは現在）にPOIが営業中か判定する"""
        if target_time is None:
            target_time = datetime.now()
        
        # 曜日 (0=日, 1=月, ..., 6=土) に変換
        day_idx = int(target_time.strftime("%w"))
        hour_idx = target_time.hour
        
        temp_feat = poi_row['temp_feat']
        # temp_feat: [0:24] hours, [24:31] days
        # 営業日チェック
        if temp_feat[24 + day_idx] < 0.5:
            return False
        # 営業時間チェック
        if temp_feat[hour_idx] < 0.5:
            return False
            
        return True

    def recommend(self, target_poi_name, target_time=None, distance_threshold=DISTANCE_THRESHOLD_M, top_n=TOP_N_RECOMMENDATIONS, filter_open=True):
        """
        基準となるPOIから、関連が深い（GNN類似度が高い）エリアのPOIを推薦する
        """
        # 1. 基準となるPOIを検索
        target_rows = self.poi_df[self.poi_df['name'].str.contains(target_poi_name, na=False)]
        if target_rows.empty:
            raise ValueError(f"指定されたPOI '{target_poi_name}' は見つかりませんでした。")
            
        target_poi = target_rows.iloc[0]
        target_coords = (target_poi['lat'], target_poi['lng'])
        target_gnn_emb = np.array(target_poi['gnn_embedding']).reshape(1, -1)
        
        # 2. フィルタリング
        temp_df = self.poi_df.copy()
        
        # 距離計算
        temp_df['distance_m'] = temp_df.apply(
            lambda row: haversine_distance(target_coords, (row['lat'], row['lng'])), axis=1
        )
        filtered_df = temp_df[temp_df['distance_m'] <= distance_threshold].copy()
        
        # 営業中フィルタ (オプション)
        if filter_open:
            filtered_df['is_open'] = filtered_df.apply(lambda row: self.is_currently_open(row, target_time), axis=1)
            filtered_df = filtered_df[filtered_df['is_open'] == True]

        # 自身を除外
        filtered_df = filtered_df[filtered_df['name'] != target_poi['name']]
        
        if filtered_df.empty:
            return pd.DataFrame()
            
        # 3. GNN埋め込みによる関連度ランキング
        candidate_embeddings = np.vstack(filtered_df['gnn_embedding'].values)
        similarities = cosine_similarity(target_gnn_emb, candidate_embeddings)[0]
        filtered_df['similarity_score'] = similarities
        
        # 同じGNNクラスタに属するものを優先（ボーナス付与）し、関連エリアとしての性質を強める
        target_cluster = target_poi['cluster']
        filtered_df.loc[filtered_df['cluster'] == target_cluster, 'similarity_score'] += 0.2
        
        # ソートして終了
        recommended_df = filtered_df.sort_values(by='similarity_score', ascending=False).head(top_n)
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
