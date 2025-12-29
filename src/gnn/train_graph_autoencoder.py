# -*- coding: utf-8 -*-
"""
このスクリプトは、OpenStreetMap (OSM) の道路網データからグラフを構築し、
Graph Autoencoder (GAE) を用いて各地点の低次元特徴量（エンベディング）を学習します。

処理手順：
1. 必要なライブラリのインストールを確認 (torch, torch_geometric, osmnx)。
2. 地点ごとの景観特徴量ベクトル (concatenated_vectors.csv) と座標データを読み込みます。
3. OSMnxを使い、OSMデータ (Hakodate.osm.xml) から道路ネットワークグラフを構築します。
4. 各景観観測点を、OSMグラフ上の最も近いノードに「スナップ」します。
5. 道路の隣接関係に基づき、景観観測点間のエッジを定義し、PyTorch Geometric用のデータオブジェクトを作成します。
6. GCNベースのGraph Autoencoderモデルを定義します。
7. モデルを学習させ、グラフの構造を再構成できるようにします。
8. 学習済みのエンコーダを使って、各地点の新しいエンベディングを抽出し、CSVファイルとして保存します。
"""

# --- ライブラリのインストール案内 ---
# 以下のライブラリが必要です。事前にインストールしてください。
# pip install torch pandas numpy osmnx matplotlib scikit-learn
# PyTorch Geometricのインストールは公式サイトを参照してください:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

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

# --- 設定 ---
BASE_DIR = '/Users/atsuyakatougi/Desktop/master'
N_EPOCHS = 200
LEARNING_RATE = 0.01
EMBEDDING_DIM = 64 # 学習後の特徴量の次元数

# --- パス設定 ---
FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'segmentation_results_50m', 'concatenated_vectors.csv')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'street_view_images_50m_optimized', 'pano_metadata.json')
OSM_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'osm_hakodate', 'Hakodate.osm.xml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'gnn_embeddings')
EMBEDDING_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'embeddings_dim{EMBEDDING_DIM}.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """景観特徴量と座標データを読み込む"""
    print("データを読み込み中...")
    # 特徴量ベクトルを読み込み
    features_df = pd.read_csv(FEATURES_PATH)
    
    # 座標データを読み込み
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item and item["api_location"]}
    
    # 特徴量DFに座標を追加
    features_df['latitude'] = features_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    features_df['longitude'] = features_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    features_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    print(f"{len(features_df)} 地点分の特徴量と座標を読み込みました。")
    return features_df

def build_graph(points_df):
    """OSMデータからグラフを構築し、PyGデータオブジェクトを作成する"""
    print("OSM道路網からグラフを構築中...")
    # OSMファイルからグラフを読み込む
    G_osm = ox.graph_from_xml(OSM_PATH)
    
    # 座標リストを作成
    points_coords = list(zip(points_df['latitude'], points_df['longitude']))
    
    # 各地点をOSMグラフの最寄りノードにスナップ
    nearest_osm_nodes = ox.nearest_nodes(G_osm, [c[1] for c in points_coords], [c[0] for c in points_coords])
    
    points_df['osm_node'] = nearest_osm_nodes
    
    # point_id と OSMノードIDのマッピングを作成
    osm_to_point_idx = {osm_node: i for i, osm_node in enumerate(points_df['osm_node'])}
    
    # OSMグラフのエッジを元に、地点間のエッジリストを作成
    edge_list = []
    for u, v in G_osm.edges():
        if u in osm_to_point_idx and v in osm_to_point_idx:
            # 異なる地点にスナップされたノード間のエッジのみを追加
            if osm_to_point_idx[u] != osm_to_point_idx[v]:
                edge_list.append((osm_to_point_idx[u], osm_to_point_idx[v]))

    # 無向グラフにするために逆方向のエッジも追加し、重複を削除
    undirected_edges = set(edge_list + [(v, u) for u, v in edge_list])
    
    # PyG用のedge_indexを作成
    edge_index = torch.tensor(list(undirected_edges), dtype=torch.long).t().contiguous()
    
    # ノード特徴量を準備
    feature_cols = points_df.columns.drop(['point_id', 'latitude', 'longitude', 'osm_node'])
    features = points_df[feature_cols].values
    
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    x = torch.tensor(features_scaled, dtype=torch.float)
    
    # PyGデータオブジェクトを作成
    data = Data(x=x, edge_index=edge_index)
    
    print("グラフ構築完了。")
    print(f"ノード数: {data.num_nodes}, エッジ数: {data.num_edges}")
    
    return data

# GAEのエンコーダ部分の定義
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def train_model(data):
    """Graph Autoencoderモデルを学習させる"""
    print("GAEモデルの学習を開始します...")
    in_channels = data.num_node_features
    out_channels = EMBEDDING_DIM
    
    model = GAE(GCNEncoder(in_channels, out_channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    model = model.to(device)
    data = data.to(device)
    
    loss_history = []
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        # エンコーダから潜在表現（エンベディング）を取得
        z = model.encode(data.x, data.edge_index)
        # デコーダで隣接行列を再構成し、損失を計算
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    print("学習が完了しました。")
    
    # 損失の推移をプロット
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("GAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))
    plt.close()

    # 学習済みエンベディングを抽出
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
