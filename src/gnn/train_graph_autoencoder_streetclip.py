# -*- coding: utf-8 -*-
"""
このスクリプトは、StreetCLIPによって生成された画像特徴量を入力とし、
各地点を東西南北（Front, Right, Back, Left）の4つのノードに分割したグラフ構造上で
Graph Autoencoder (GAE) を学習させます。

目的：
StreetCLIPが捉えた「景観の雰囲気」と、OSM道路網による「地理的なつながり」を
統合した、低次元の埋め込み表現を獲得すること。
"""

import pandas as pd
import numpy as np
import os
import json
import osmnx as ox
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
N_EPOCHS = 200
LEARNING_RATE = 0.0005 # 次元数が多いため少し下げる
EMBEDDING_DIM = 64 # 最終的な埋め込み次元

# --- パス設定 ---
# StreetCLIPの特徴量CSVを使用
FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'streetclip_embeddings', 'streetclip_features.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OSM_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'osm_hakodate', 'Hakodate.osm.xml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings')
EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'embeddings_dim{EMBEDDING_DIM}_streetclip_directional.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_process_data():
    """データを読み込み、前処理を行う"""
    print("データを読み込み中...")
    
    # 1. 特徴量データの読み込み
    try:
        df = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {FEATURES_PATH}")
        return None
        
    print(f"元データ: {len(df)} 行")

    # CSVには既に 'point_id', 'angle', 'direction' があるはずだが、
    # create_streetclip_embeddings.py の出力形式に依存する。
    # angle -> direction の変換が必要な場合もあるので確認
    
    if 'direction' not in df.columns:
        # もしdirection列がなければangleから作成
        angle_map = {0: 'front', 90: 'right', 180: 'back', 270: 'left'}
        df['direction'] = df['angle'].map(angle_map)
    
    # direction_idx を付与
    dir_idx_map = {'front': 0, 'right': 1, 'back': 2, 'left': 3}
    df['direction_idx'] = df['direction'].map(dir_idx_map)
    
    # NaNチェック
    df.dropna(subset=['direction', 'direction_idx'], inplace=True)

    # 2. 4方向揃っている地点のみを抽出
    point_counts = df['point_id'].value_counts()
    valid_points = point_counts[point_counts == 4].index
    df = df[df['point_id'].isin(valid_points)].copy()
    
    print(f"4方向揃っている地点数: {len(valid_points)}")
    print(f"使用するデータ行数: {len(df)}")

    # 3. 座標データの読み込みとマージ
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item and item["api_location"]}
    
    # 座標を追加
    df['latitude'] = df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    df['longitude'] = df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    # ソートして返却 (point_id, direction_idx順)
    df.sort_values(by=['point_id', 'direction_idx'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def build_directional_graph(points_df):
    """
    方向別ノードを持つグラフを構築する
    """
    print("OSM道路網から方向別グラフを構築中...")
    
    # 1. OSMデータの準備
    G_osm = ox.graph_from_xml(OSM_PATH)
    
    # ユニークな地点の座標を取得してOSMノードにスナップ
    unique_points = points_df[['point_id', 'latitude', 'longitude']].drop_duplicates()
    points_coords = list(zip(unique_points['latitude'], unique_points['longitude']))
    nearest_osm_nodes = ox.nearest_nodes(G_osm, [c[1] for c in points_coords], [c[0] for c in points_coords])
    
    pid_to_osm = dict(zip(unique_points['point_id'], nearest_osm_nodes))
    
    osm_to_pids = {}
    for pid, osm_node in tqdm(pid_to_osm.items(), desc="OSMデータ"):
        if osm_node not in osm_to_pids:
            osm_to_pids[osm_node] = []
        osm_to_pids[osm_node].append(pid)
        
    points_df['osm_node'] = points_df['point_id'].map(pid_to_osm)
    
    # 2. エッジの構築
    edge_list = []
    pid_to_node_indices = points_df.groupby('point_id').indices
    
    # --- A. 地点内エッジ (Intra-location) ---
    for pid, indices in tqdm(pid_to_node_indices.items(), desc="エッジ構築"):
        curr_indices = sorted(indices) 
        for i in range(4):
            u = curr_indices[i]
            v = curr_indices[(i + 1) % 4]
            edge_list.append((u, v))
            edge_list.append((v, u))
            
    # --- B. 地点間エッジ (Inter-location) ---
    processed_osm_edges = set()
    for u_osm, v_osm in tqdm(G_osm.edges(), desc="全結合"):
        if u_osm > v_osm: u_osm, v_osm = v_osm, u_osm
        if (u_osm, v_osm) in processed_osm_edges: continue
        processed_osm_edges.add((u_osm, v_osm))
        
        pids_u = osm_to_pids.get(u_osm, [])
        pids_v = osm_to_pids.get(v_osm, [])
        
        for pid_a in pids_u:
            for pid_b in pids_v:
                if pid_a == pid_b: continue
                nodes_a = pid_to_node_indices[pid_a]
                nodes_b = pid_to_node_indices[pid_b]
                for na in nodes_a:
                    for nb in nodes_b:
                        edge_list.append((na, nb))
                        edge_list.append((nb, na))

    unique_edges = set(edge_list)
    edge_index = torch.tensor(list(unique_edges), dtype=torch.long).t().contiguous()
    
    # 3. ノード特徴量の準備
    # 'feat_' で始まるカラムを取得
    feature_cols = [c for c in points_df.columns if c.startswith('feat_')]
    features = points_df[feature_cols].values.astype(np.float32)
    
    print(f"入力特徴量の次元数: {features.shape[1]}")
    
    # NaN/Infチェック
    if np.isnan(features).any() or np.isinf(features).any():
        print("警告: 入力特徴量にNaNまたはInfが含まれています。0で置換します。")
        features = np.nan_to_num(features)
    
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    x = torch.tensor(features_scaled, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    print("グラフ構築完了。")
    print(f"ノード数: {data.num_nodes}, エッジ数: {data.num_edges}")
    
    return data

# GAEのエンコーダ部分
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # 入力次元(768) -> 中間層(256) -> 出力次元(64)
        hidden_channels = 256
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def train_model(data):
    """GAEモデルの学習"""
    print("GAEモデルの学習を開始します...")
    in_channels = data.num_node_features
    out_channels = EMBEDDING_DIM
    
    model = GAE(GCNEncoder(in_channels, out_channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"使用デバイス: {device}")
    model = model.to(device)
    data = data.to(device)
    
    loss_history = []
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    print("学習完了。")
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("GAE Training Loss (StreetCLIP + Directional)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss_streetclip.png'))
    plt.close()

    with torch.no_grad():
        final_embeddings = model.encode(data.x, data.edge_index).cpu().numpy()
        
    return final_embeddings

def main():
    points_df = load_and_process_data()
    if points_df is None:
        return

    data = build_directional_graph(points_df)
    embeddings = train_model(data)
    
    print(f"学習済みエンベディングを保存中...")
    
    embedding_df = pd.DataFrame(embeddings)
    embedding_df['point_id'] = points_df['point_id'].values
    embedding_df['direction'] = points_df['direction'].values
    
    # カラム名: point_id, direction, emb_0, ...
    cols = ['point_id', 'direction'] + [col for col in embedding_df.columns if col not in ['point_id', 'direction']]
    embedding_df = embedding_df[cols]
    
    embedding_df.to_csv(EMBEDDING_OUTPUT_PATH, index=False)
    
    print(f"エンベディングを {EMBEDDING_OUTPUT_PATH} に保存しました。")

if __name__ == '__main__':
    main()
