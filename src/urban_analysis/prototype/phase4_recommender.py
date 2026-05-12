import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
import sys

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.urban_analysis.prototype.time_parser import extract_time_windows
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
POI_CLUSTERS_CSV = PROJECT_ROOT / 'data' / 'processed' / 'refined_poi_clusters_k8.csv'
LS_NODES_CSV = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs_individual' / 'landscape' / 'nodes.csv'
LS_EMB_NPY = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results_individual' / 'landscape' / 'embeddings.npy'
RAW_PLACES_JSON = PROJECT_ROOT / 'data' / 'raw' / 'output_with_google_places_jp.json'

class Phase4Recommender:
    def __init__(self):
        """フェーズ4専用推薦エンジンの初期化"""
        print("フェーズ4推薦エンジンを初期化中...")
        
        # POIデータの読み込み
        if not POI_CLUSTERS_CSV.exists():
            raise FileNotFoundError(f"POIクラスタファイルが見つかりません: {POI_CLUSTERS_CSV}")
        self.poi_df = pd.read_csv(POI_CLUSTERS_CSV)
        
        # 営業時間の読み込みと統合
        print("営業時間をパースして統合中...")
        time_windows = extract_time_windows(RAW_PLACES_JSON)
        self.poi_df['open_time'] = self.poi_df['name'].map(lambda x: time_windows.get(x, (0, 1440))[0])
        self.poi_df['close_time'] = self.poi_df['name'].map(lambda x: time_windows.get(x, (0, 1440))[1])
        
        # 景観データ（単体GNN出力）の読み込みとKMeansクラスタリング
        if not LS_NODES_CSV.exists() or not LS_EMB_NPY.exists():
            raise FileNotFoundError("景観のGNN単体学習結果が見つかりません。")
            
        print("景観GNN埋め込みを読み込み、K=12でクラスタリング中...")
        ls_nodes = pd.read_csv(LS_NODES_CSV)
        ls_embs = np.load(LS_EMB_NPY)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
        labels = kmeans.fit_predict(ls_embs)
        ls_nodes['cluster'] = labels
        
        self.ls_valid_df = ls_nodes.copy()
        
        # 緯度経度をメートル近似に変換 (函館周辺)
        lat_to_m = 111000
        lng_to_m = 82000
        
        ls_lats_m = self.ls_valid_df['lat'].values * lat_to_m
        ls_lngs_m = self.ls_valid_df['lng'].values * lng_to_m
        self.ls_coords_m = np.column_stack((ls_lats_m, ls_lngs_m))
        self.ls_tree = KDTree(self.ls_coords_m)
        
        # 各POIの景観クラスタを事前に計算しておく (Radius-based Distance-weighted Voting)
        print("POIごとの景観クラスタを計算中 (Radius=150m Distance-weighted Voting)...")
        poi_ls_clusters = []
        RADIUS = 150.0
        epsilon = 1e-5
        
        for _, row in self.poi_df.iterrows():
            try:
                poi_coord_m = np.array([float(row['lat']) * lat_to_m, float(row['lng']) * lng_to_m])
                indices = self.ls_tree.query_ball_point(poi_coord_m, r=RADIUS)
                
                if len(indices) > 0:
                    # 半径内に景観ポイントが存在する場合：重み付き投票
                    cluster_scores = {}
                    for idx in indices:
                        ls_coord_m = self.ls_coords_m[idx]
                        dist = np.linalg.norm(poi_coord_m - ls_coord_m)
                        cluster = self.ls_valid_df.iloc[idx]['cluster']
                        weight = 1.0 / (dist + epsilon)
                        cluster_scores[cluster] = cluster_scores.get(cluster, 0) + weight
                    
                    # 最大スコアのクラスタを採用
                    best_cluster = max(cluster_scores.items(), key=lambda x: x[1])[0]
                    poi_ls_clusters.append(best_cluster)
                else:
                    # 半径内に存在しない場合：最も近い1点をフォールバックとして採用
                    _, idx = self.ls_tree.query(poi_coord_m, k=1)
                    if isinstance(idx, (list, np.ndarray)):
                        idx = idx[0]
                    best_cluster = self.ls_valid_df.iloc[idx]['cluster']
                    poi_ls_clusters.append(best_cluster)
            except Exception as e:
                poi_ls_clusters.append(-1)
                
        self.poi_df['ls_cluster'] = poi_ls_clusters

    def get_landscape_cluster(self, lat, lng):
        """指定された座標の最寄り景観クラスタを取得する"""
        _, idx = self.ls_tree.query([lat, lng])
        return self.ls_valid_df.iloc[idx]['cluster']

    def recommend(self, target_poi_name, top_n=10):
        """
        起点POIから、「同景観クラスタ」のPOIを、機能の多様性を考慮して推薦する
        """
        # 1. 基準となるPOIを検索
        target_rows = self.poi_df[self.poi_df['name'].str.contains(target_poi_name, na=False)]
        if target_rows.empty:
            raise ValueError(f"指定されたPOI '{target_poi_name}' は見つかりませんでした。")
            
        target_poi = target_rows.iloc[0]
        target_poi_cluster = target_poi['cluster']
        target_ls_cluster = target_poi['ls_cluster']
        
        print(f"起点POI: {target_poi['name']} (POIクラスタ: {target_poi_cluster}, 景観クラスタ: {target_ls_cluster})")
        
        # 2. フィルタリング条件の適用
        # 「同景観クラスタ」のみに緩和（地域の特性の共有）
        candidates_df = self.poi_df[
            (self.poi_df['ls_cluster'] == target_ls_cluster) &
            (self.poi_df['name'] != target_poi['name']) # 自身を除外
        ].copy()
        
        total_candidates = len(candidates_df)
        print(f"-> 条件（景観クラスタ: {target_ls_cluster}）に合致した候補POI総数: {total_candidates}件")
        
        if candidates_df.empty:
            print(f"警告: '{target_poi_name}' と同じ条件（景観クラスタ={target_ls_cluster}）を満たす他のPOIが見つかりませんでした。")
            return pd.DataFrame(), target_poi, pd.DataFrame()
            
        # 距離計算 (近い順にソートするため)
        def haversine_distance(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return 6371000 * c
            
        candidates_df['distance_m'] = candidates_df.apply(
            lambda row: haversine_distance(target_poi['lat'], target_poi['lng'], row['lat'], row['lng']), axis=1
        )
        
        candidates_df = candidates_df.sort_values(by='distance_m')
        
        # 多様性フィルター (Diversity Filter): 同一機能クラスタは上限 2 件、起点クラスタは上限 4 件
        max_per_cluster_others = 2
        max_per_cluster_start = 6
        start_cluster = target_poi_cluster
        per_cluster_limits = {start_cluster: max_per_cluster_start}
        selected_indices = []
        cluster_counts = {}
        
        for idx, row in candidates_df.iterrows():
            func_cluster = row['cluster']
            limit = per_cluster_limits.get(func_cluster, max_per_cluster_others)
            if cluster_counts.get(func_cluster, 0) < limit:
                selected_indices.append(idx)
                cluster_counts[func_cluster] = cluster_counts.get(func_cluster, 0) + 1
            
            if len(selected_indices) >= top_n:
                break
                
        recommended_df = candidates_df.loc[selected_indices]
        return recommended_df, target_poi, candidates_df

if __name__ == "__main__":
    recommender = Phase4Recommender()
    test_poi = "金森赤レンガ倉庫"
    res_df, t_poi, _ = recommender.recommend(test_poi)
    if not res_df.empty:
        print("\n--- 推薦結果 ---")
        print(res_df[['name', 'distance_m', 'cluster', 'ls_cluster', 'open_time', 'close_time']])
    else:
        print("結果がありません。")
