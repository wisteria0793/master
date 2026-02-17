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
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# コンポーネントのインポート
from urban_analysis.config import (
    PROCESSED_DATA_DIR, 
    RAW_DATA_DIR, 
    OSM_XML_PATH
)
from urban_analysis.gnn.models import create_gae_model
from urban_analysis.gnn.graph_builder import build_graph_from_osm

# --- 設定 ---
N_EPOCHS = 200
LEARNING_RATE = 0.01
EMBEDDING_DIM = 64

# --- パス設定 ---
FEATURES_PATH = PROCESSED_DATA_DIR / 'segmentation_results_50m' / 'location_features_sorted.csv'
METADATA_PATH = RAW_DATA_DIR / 'street_view_images_50m_optimized' / 'pano_metadata.json'
OUTPUT_DIR = PROCESSED_DATA_DIR / 'gnn_embeddings'
EMBEDDING_OUTPUT_PATH = OUTPUT_DIR / f'embeddings_dim{EMBEDDING_DIM}_feature_sorted.csv'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """データ読み込みと結合処理"""
    print("データを読み込み中...")
    features_df = pd.read_csv(FEATURES_PATH)
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    coords_map = {item["pano_id"]: item["api_location"] for item in metadata if "api_location" in item}
    
    features_df['latitude'] = features_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[0])
    features_df['longitude'] = features_df['point_id'].map(lambda x: coords_map.get(x, [None, None])[1])
    features_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    return features_df

def train_model(data):
    """学習ループ"""
    print("GAEモデルの学習を開始します...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    model = create_gae_model(data.num_node_features, EMBEDDING_DIM, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    data = data.to(device)
    loss_history = []
    
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    # 学習曲線の保存
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("GAE Training Loss")
    plt.savefig(OUTPUT_DIR / 'training_loss.png')
    plt.close()

    return model.encode(data.x, data.edge_index).detach().cpu().numpy()

def main():
    points_df = load_data()
    
    # ロジックを分離した graph_builder を使用
    data = build_graph_from_osm(OSM_XML_PATH, points_df)
    
    embeddings = train_model(data)
    
    print(f"学習済みエンベディングを保存中... (次元数: {embeddings.shape[1]})")
    pd.DataFrame(embeddings, index=points_df['point_id']).to_csv(EMBEDDING_OUTPUT_PATH)
    print("完了。")

if __name__ == '__main__':
    main()
