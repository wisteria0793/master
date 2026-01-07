# -*- coding: utf-8 -*-
"""
このスクリプトは、GAT (Graph Attention Network) をエンコーダとして使用する
Graph Autoencoder (GAE) を用いて、各地点の低次元特徴量（エンベディング）を学習します。
GCNの代わりにGATを用いることで、ノード間の関係性の重み付けをモデルが自動で学習します。

処理手順：
1. 景観特徴量ベクトルと座標データを読み込みます。
2. OSMnxを使い、道路ネットワークグラフを構築します。
3. PyTorch Geometric用のデータオブジェクトを作成します。
4. GATベースのGraph Autoencoderモデルを定義します。
5. モデルを学習させます。
6. 学習済みのエンコーダを使って、各地点の新しいエンベディングを抽出し、CSVファイルとして保存します。
"""

import pandas as pd
import numpy as np
import os
import json
import osmnx as ox
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GAE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
N_EPOCHS = 5000
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64 # 学習後の特徴量の次元数

# --- GATモデルのパラメータ ---
GAT_HEADS = 8
GAT_DROPOUT = 0.6

# --- パス設定 ---
FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'segmentation_results_50m', 'concatenated_vectors.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OSM_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'osm_hakodate', 'Hakodate.osm.xml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings')
EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'embeddings_gat_dim{EMBEDDING_DIM}.csv')
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, 'training_loss_gat.png')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """景観特徴量と座標データを読み込む"""
    print("データを読み込み中...")
    features_df = pd.read_csv(FEATURES_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item and item["api_location"]}
    features_df['latitude'] = features_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    features_df['longitude'] = features_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    features_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    print(f"{len(features_df)} 地点分の特徴量と座標を読み込みました。")
    return features_df

def build_graph(points_df):
    """OSMデータからグラフを構築し、PyGデータオブジェクトを作成する"""
    print("OSM道路網からグラフを構築中...")
    G_osm = ox.graph_from_xml(OSM_PATH)
    points_coords = list(zip(points_df['latitude'], points_df['longitude']))
    nearest_osm_nodes = ox.nearest_nodes(G_osm, [c[1] for c in points_coords], [c[0] for c in points_coords])
    points_df['osm_node'] = nearest_osm_nodes
    osm_to_point_idx = {osm_node: i for i, osm_node in enumerate(points_df['osm_node'])}
    
    edge_list = []
    for u, v in G_osm.edges():
        if u in osm_to_point_idx and v in osm_to_point_idx:
            if osm_to_point_idx[u] != osm_to_point_idx[v]:
                edge_list.append((osm_to_point_idx[u], osm_to_point_idx[v]))

    undirected_edges = set(edge_list + [(v, u) for u, v in edge_list])
    edge_index = torch.tensor(list(undirected_edges), dtype=torch.long).t().contiguous()
    
    feature_cols = points_df.columns.drop(['point_id', 'latitude', 'longitude', 'osm_node'])
    features = points_df[feature_cols].values
    features_scaled = StandardScaler().fit_transform(features)
    x = torch.tensor(features_scaled, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    print(f"グラフ構築完了。ノード数: {data.num_nodes}, エッジ数: {data.num_edges}")
    return data

# GAEのエンコーダ部分をGATで定義
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=1, concat=True, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=GAT_DROPOUT, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=GAT_DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_model(data):
    """Graph Autoencoder (GAT版) モデルを学習させる"""
    print("GAE-GATモデルの学習を開始します...")
    in_channels = data.num_node_features
    out_channels = EMBEDDING_DIM
    
    model = GAE(GATEncoder(in_channels, out_channels, heads=GAT_HEADS, dropout=GAT_DROPOUT))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    model = model.to(device)
    data = data.to(device)
    
    loss_history = []
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    print("学習が完了しました。")
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("GAE-GAT Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(data.x, data.edge_index).cpu().numpy()
        
    return final_embeddings

def main():
    """メインの処理フロー"""
    points_df = load_data()
    data = build_graph(points_df)
    embeddings = train_model(data)
    
    print(f"学習済みエンベディングを保存中... (次元数: {embeddings.shape[1]})")
    embedding_df = pd.DataFrame(embeddings, index=points_df['point_id'])
    embedding_df.to_csv(EMBEDDING_OUTPUT_PATH)
    
    print(f"エンベディングを {EMBEDDING_OUTPUT_PATH} に保存しました。")
    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()
