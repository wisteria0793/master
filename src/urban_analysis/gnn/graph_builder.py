
import osmnx as ox
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, List, Optional

def build_graph_from_osm(osm_path: Path, points_df: pd.DataFrame) -> Data:
    """
    OpenStreetMap (OSM) データと地点データフレームから、PyTorch Geometric用のグラフデータを構築します。
    
    各地点（景観画像など）をOSMの道路ネットワーク上の最近傍ノードにマッピングし、
    道路の接続関係に基づいてエッジを定義します。
    
    Args:
        osm_path (Path): OSM XMLファイルのパス。osmnxで読み込まれます。
        points_df (pd.DataFrame): 以下のカラムを持つDataFrame。
            - 'latitude': 緯度
            - 'longitude': 経度
            - その他: 特徴量として使用される数値カラム
        
    Returns:
        Data: PyTorch GeometricのDataオブジェクト。以下の属性を持ちます。
            - x (torch.Tensor): ノードの特徴量行列 [num_nodes, num_features]。標準化済み。
            - edge_index (torch.LongTensor): グラフの接続関係 [2, num_edges]。
            
    Process:
        1. OSMデータをロードし、道路ネットワークグラフ(NetworkX)を構築。
        2. points_dfの各行（地点）を、最も近いOSMノードIDに関連付けます（'osm_node'カラム追加）。
        3. OSMグラフ上のエッジ情報を利用して、地点間のエッジリストを作成します。
           ※ 同じOSMノードにマッピングされた地点同士は接続されません（自己ループ回避）。
        4. ポイントの特徴量を抽出し、StandardScalerで標準化してテンソル化します。
    """
    print("OSM道路網からグラフを構築中...")
    
    # 1. OSMグラフのロード
    # XMLファイルから道路ネットワークを読み込み、有向グラフとして取得
    G_osm = ox.graph_from_xml(osm_path)
    
    # 2. 座標リストの作成と最近傍ノードへのスナップ
    points_coords = list(zip(points_df['latitude'], points_df['longitude']))
    
    # 各地点の座標に最も近いOSMノードを検索
    nearest_osm_nodes = ox.nearest_nodes(G_osm, [c[1] for c in points_coords], [c[0] for c in points_coords])
    points_df['osm_node'] = nearest_osm_nodes
    
    # 3. エッジの構築
    # OSMノードID -> ポイントデータのインデックス への逆引き辞書
    # 注意: 複数のポイントが同じOSMノードに紐づく場合、最後のもので上書きされる可能性があるが、
    # ここではグラフ構造の骨格としてOSMを使うため、簡易的に実装されている。
    osm_to_point_idx = {osm_node: i for i, osm_node in enumerate(points_df['osm_node'])}
    
    edge_list = []
    for u, v in G_osm.edges():
        if u in osm_to_point_idx and v in osm_to_point_idx:
            # 異なる地点にスナップされたノード間のエッジのみ追加（自己ループ除外）
            if osm_to_point_idx[u] != osm_to_point_idx[v]:
                edge_list.append((osm_to_point_idx[u], osm_to_point_idx[v]))

    # 無向グラフとして扱うため、逆方向のエッジも追加して重複を削除
    undirected_edges = set(edge_list + [(v, u) for u, v in edge_list])
    edge_index = torch.tensor(list(undirected_edges), dtype=torch.long).t().contiguous()
    
    # 4. ノード特徴量の準備
    # IDや座標などのメタデータカラムを除外して特徴量として扱う
    exclude_cols = ['point_id', 'latitude', 'longitude', 'osm_node']
    feature_cols = [c for c in points_df.columns if c not in exclude_cols]
    
    features = points_df[feature_cols].values
    
    # 標準化 (StandardScaler) : 平均0, 分散1に正規化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    x = torch.tensor(features_scaled, dtype=torch.float)
    
    # PyGデータオブジェクト作成
    data = Data(x=x, edge_index=edge_index)
    
    print("グラフ構築完了。")
    print(f"ノード数: {data.num_nodes}, エッジ数: {data.num_edges}")
    
    return data
