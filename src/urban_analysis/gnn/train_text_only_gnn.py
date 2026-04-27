# -*- coding: utf-8 -*-
"""
実験18.6: アプローチC - テキスト情報のみを用いたGNN学習。
営業時間を排除することで、GNN自体の幾何学的学習効果と、時間情報の付加価値を分離する。
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from pathlib import Path

# コンポーネントのインポート
from urban_analysis.gnn.models import MultimodalResidualGATEncoder

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_FILTERED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'text_only_gnn'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ハイパーパラメータ (18.3と同一に設定)
EMBEDDING_DIM = 64
SHARED_DIM = 128
LEARNING_RATE = 0.002
N_EPOCHS = 200
DISTANCE_THRESHOLD = 150 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. データの読み込み
    print("限定地域データを読み込み中...")
    poi_feats_full = np.load(GNN_FILTERED_DIR / 'poi_features.npy')
    # テキストのみ (768次元) を抽出
    poi_feats = poi_feats_full[:, :768]
    sv_feats = np.load(GNN_FILTERED_DIR / 'sv_features.npy')
    
    with open(GNN_FILTERED_DIR / 'nodes_metadata.json', 'r') as f:
        nodes_meta = json.load(f)
    
    n_poi = len(poi_feats)
    n_sv = len(sv_feats)
    
    # 結合特徴量 (POI: 768, SV: 768)
    x_combined = np.zeros((n_poi + n_sv, 768), dtype=np.float32)
    x_combined[:n_poi, :] = poi_feats
    x_combined[n_poi:, :] = sv_feats
    
    node_types = torch.cat([torch.zeros(n_poi, dtype=torch.long), torch.ones(n_sv, dtype=torch.long)])
    
    # 2. 地理的グラフの構築 (18.3と同一ロジック)
    coords = np.array([[n['lat'], n['lng']] for n in nodes_meta])
    coords_m = coords * np.array([111000, 82000])
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(DISTANCE_THRESHOLD)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # 3. 特徴量の標準化
    scaler = StandardScaler()
    x_combined = scaler.fit_transform(x_combined)
    
    x_tensor = torch.from_numpy(x_combined).to(device)
    edge_index = edge_index.to(device)
    node_types = node_types.to(device)
    data = Data(x=x_tensor, edge_index=edge_index)
    
    # 4. モデルの構築 (POI入力次元も768に設定)
    encoder = MultimodalResidualGATEncoder(768, 768, SHARED_DIM, EMBEDDING_DIM)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"テキストのみのGNN学習開始 ({N_EPOCHS} epochs)...")
    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        z = model.encoder(data.x, data.edge_index, node_types)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 40 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
            
    # 5. 保存
    model.eval()
    with torch.no_grad():
        z = model.encoder(data.x, data.edge_index, node_types).cpu().numpy()
        
    output_csv = OUTPUT_DIR / 'text_only_gnn_embeddings.csv'
    df = pd.DataFrame(nodes_meta)
    for i in range(EMBEDDING_DIM):
        df[f'dim_{i}'] = z[:, i]
    df.to_csv(output_csv, index=False)
    print(f"テキストGNN埋め込みを保存しました: {output_csv}")

if __name__ == "__main__":
    main()
