# -*- coding: utf-8 -*-
"""
このスクリプトは、以下の2つの特徴量を結合し、GNN (GAE) で統合的な学習を行います。

1. StreetCLIP Embeddings (768次元): 画像の「見た目・雰囲気」
2. Segmentation GNN Embeddings (64次元): セグメンテーション比率に基づく「構造的特徴」（実験13の成果物）

目的：
視覚的な雰囲気と、構成要素に基づく構造的特徴の両方を考慮した、
よりリッチでロバストな地区特性エンベディングを生成する。
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
LEARNING_RATE = 0.0005
EMBEDDING_DIM = 64 # 最終的な埋め込み次元

# --- パス設定 ---
STREETCLIP_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'streetclip_embeddings', 'streetclip_features.csv')
SEG_GNN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings', 'embeddings_dim64_directional.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OSM_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'osm_hakodate', 'Hakodate.osm.xml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings')
EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'embeddings_dim{EMBEDDING_DIM}_combined_directional.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_process_data():
    """2つの特徴量データを読み込み、結合して前処理を行う"""
    print("データを読み込み中...")
    
    # 1. データの読み込み
    try:
        df_clip = pd.read_csv(STREETCLIP_PATH)
        df_seg = pd.read_csv(SEG_GNN_PATH)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e}")
        return None
        
    print(f"StreetCLIPデータ: {len(df_clip)} 行")
    print(f"Seg-GNNデータ: {len(df_seg)} 行")

    # 2. キーの統一とマージ
    # StreetCLIP側にdirectionがない場合への対応
    if 'direction' not in df_clip.columns:
        angle_map = {0: 'front', 90: 'right', 180: 'back', 270: 'left'}
        df_clip['direction'] = df_clip['angle'].map(angle_map)

    # 共通のキーでマージ (point_id, direction)
    # df_segのカラム名が重複しないようにサフィックスをつけるか、事前に確認
    # Seg-GNNの特徴量は '0', '1', ... となっている可能性が高いので、リネーム推奨
    
    # Seg-GNNのカラム名を変更 (emb_seg_0, emb_seg_1...)
    seg_feat_cols = [c for c in df_seg.columns if c not in ['point_id', 'direction']]
    rename_map = {c: f'seg_emb_{c}' for c in seg_feat_cols}
    df_seg.rename(columns=rename_map, inplace=True)
    
    # マージ
    print("データを結合中...")
    df_merged = pd.merge(df_clip, df_seg, on=['point_id', 'direction'], how='inner')
    print(f"結合後データ: {len(df_merged)} 行")

    # 3. direction_idx を付与
    dir_idx_map = {'front': 0, 'right': 1, 'back': 2, 'left': 3}
    df_merged['direction_idx'] = df_merged['direction'].map(dir_idx_map)
    df_merged.dropna(subset=['direction_idx'], inplace=True)

    # 4. 4方向揃っている地点のみを抽出
    point_counts = df_merged['point_id'].value_counts()
    valid_points = point_counts[point_counts == 4].index
    df_merged = df_merged[df_merged['point_id'].isin(valid_points)].copy()
    
    print(f"4方向揃っている地点数: {len(valid_points)}")
    print(f"使用するデータ行数: {len(df_merged)}")

    # 5. 座標データの読み込みとマージ
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item and item["api_location"]}
    
    df_merged['latitude'] = df_merged['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    df_merged['longitude'] = df_merged['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    df_merged.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    # ソート
    df_merged.sort_values(by=['point_id', 'direction_idx'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    
    return df_merged

def build_directional_graph(points_df):
    """方向別ノードを持つグラフを構築する (共通ロジック)"""
    print("OSM道路網から方向別グラフを構築中...")
    
    G_osm = ox.graph_from_xml(OSM_PATH)
    
    unique_points = points_df[['point_id', 'latitude', 'longitude']].drop_duplicates()
    points_coords = list(zip(unique_points['latitude'], unique_points['longitude']))
    nearest_osm_nodes = ox.nearest_nodes(G_osm, [c[1] for c in points_coords], [c[0] for c in points_coords])
    
    pid_to_osm = dict(zip(unique_points['point_id'], nearest_osm_nodes))
    
    osm_to_pids = {}
    for pid, osm_node in tqdm(pid_to_osm.items(), desc="OSMデータ"):
        if osm_node not in osm_to_pids: osm_to_pids[osm_node] = []
        osm_to_pids[osm_node].append(pid)
        
    points_df['osm_node'] = points_df['point_id'].map(pid_to_osm)
    
    edge_list = []
    pid_to_node_indices = points_df.groupby('point_id').indices
    
    # Intra-location
    for pid, indices in tqdm(pid_to_node_indices.items(), desc="エッジ構築"):
        curr_indices = sorted(indices) 
        for i in range(4):
            u, v = curr_indices[i], curr_indices[(i + 1) % 4]
            edge_list.extend([(u, v), (v, u)])
            
    # Inter-location
    processed_osm_edges = set()
    for u_osm, v_osm in tqdm(G_osm.edges(), desc="全結合"):
        if u_osm > v_osm: u_osm, v_osm = v_osm, u_osm
        if (u_osm, v_osm) in processed_osm_edges: continue
        processed_osm_edges.add((u_osm, v_osm))
        
        pids_u, pids_v = osm_to_pids.get(u_osm, []), osm_to_pids.get(v_osm, [])
        for pid_a in pids_u:
            for pid_b in pids_v:
                if pid_a == pid_b: continue
                for na in pid_to_node_indices[pid_a]:
                    for nb in pid_to_node_indices[pid_b]:
                        edge_list.extend([(na, nb), (nb, na)])

    unique_edges = set(edge_list)
    edge_index = torch.tensor(list(unique_edges), dtype=torch.long).t().contiguous()
    
    # 特徴量の準備: 'feat_' (StreetCLIP) と 'seg_emb_' (Seg-GNN) の両方を使用
    feature_cols = [c for c in points_df.columns if c.startswith('feat_') or c.startswith('seg_emb_')]
    features = points_df[feature_cols].values.astype(np.float32)
    
    print(f"結合特徴量の次元数: {features.shape[1]}")
    
    if np.isnan(features).any() or np.isinf(features).any():
        print("警告: 入力特徴量にNaNまたはInfが含まれています。0で置換します。")
        features = np.nan_to_num(features)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    x = torch.tensor(features_scaled, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    print("グラフ構築完了。")
    print(f"ノード数: {data.num_nodes}, エッジ数: {data.num_edges}")
    
    return data

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # 入力次元(約832) -> 中間(256) -> 出力(64)
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
    plt.title("GAE Training Loss (Combined: StreetCLIP + Seg-GNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss_combined.png'))
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
    
    cols = ['point_id', 'direction'] + [col for col in embedding_df.columns if col not in ['point_id', 'direction']]
    embedding_df = embedding_df[cols]
    
    embedding_df.to_csv(EMBEDDING_OUTPUT_PATH, index=False)
    
    print(f"エンベディングを {EMBEDDING_OUTPUT_PATH} に保存しました。")

if __name__ == '__main__':
    main()
