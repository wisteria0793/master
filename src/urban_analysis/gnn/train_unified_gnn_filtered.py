# -*- coding: utf-8 -*-
"""
実験18.1: 特定地域（西部地区・駅前）に限定した統一GNN学習スクリプト。
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GAE
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_UNIFIED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
OUTPUT_DIR = GNN_UNIFIED_DIR

# ハイパーパラメータ (解像度を高めるため、距離しきい値を少し絞る)
EMBEDDING_DIM = 64
SHARED_DIM = 128
LEARNING_RATE = 0.005
N_EPOCHS = 300
DISTANCE_THRESHOLD = 200 # 地域が限定されているため200mに設定

class MultimodalGATEncoder(torch.nn.Module):
    def __init__(self, poi_in_channels, sv_in_channels, shared_channels, out_channels):
        super(MultimodalGATEncoder, self).__init__()
        self.poi_lin = torch.nn.Linear(poi_in_channels, shared_channels)
        self.sv_lin = torch.nn.Linear(sv_in_channels, shared_channels)
        self.conv1 = GATConv(shared_channels, shared_channels, heads=4, concat=True)
        self.conv2 = GATConv(shared_channels * 4, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, node_types):
        x_out = torch.zeros((x.size(0), self.poi_lin.out_features), device=x.device)
        poi_mask = (node_types == 0)
        sv_mask = (node_types == 1)
        if poi_mask.any():
            x_out[poi_mask] = self.poi_lin(x[poi_mask][:, :799])
        if sv_mask.any():
            x_out[sv_mask] = self.sv_lin(x[sv_mask][:, :768])
        x = x_out.relu()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. データの読み込み
    print("限定地域データを読み込み中...")
    poi_feats = np.load(GNN_UNIFIED_DIR / 'poi_features.npy')
    sv_feats = np.load(GNN_UNIFIED_DIR / 'sv_features.npy')
    with open(GNN_UNIFIED_DIR / 'nodes_metadata.json', 'r') as f:
        nodes_meta = json.load(f)
    
    n_poi = len(poi_feats)
    n_sv = len(sv_feats)
    
    x_combined = np.zeros((n_poi + n_sv, 799), dtype=np.float32)
    x_combined[:n_poi, :799] = poi_feats
    x_combined[n_poi:, :768] = sv_feats
    
    node_types = torch.cat([torch.zeros(n_poi, dtype=torch.long), torch.ones(n_sv, dtype=torch.long)])
    
    # 2. 地理的グラフの構築
    print(f"距離しきい値 {DISTANCE_THRESHOLD}m で限定地域グラフを構築中...")
    coords = np.array([[n['lat'], n['lng']] for n in nodes_meta])
    coords_m = coords * np.array([111000, 82000])
    
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(DISTANCE_THRESHOLD)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # 3. 特徴量の標準化
    scaler_poi = StandardScaler()
    scaler_sv = StandardScaler()
    x_combined[:n_poi, :799] = scaler_poi.fit_transform(x_combined[:n_poi, :799])
    x_combined[n_poi:, :768] = scaler_sv.fit_transform(x_combined[n_poi:, :768])
    
    x_tensor = torch.from_numpy(x_combined).to(device)
    edge_index = edge_index.to(device)
    node_types = node_types.to(device)
    data = Data(x=x_tensor, edge_index=edge_index)
    
    # 4. モデルの学習
    model = GAE(MultimodalGATEncoder(799, 768, SHARED_DIM, EMBEDDING_DIM)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"学習開始 ({N_EPOCHS} epochs)...")
    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index, node_types)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
            
    # 5. 保存
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, node_types).cpu().numpy()
        
    np.save(OUTPUT_DIR / 'unified_embeddings.npy', z)
    df = pd.DataFrame(nodes_meta)
    for i in range(EMBEDDING_DIM):
        df[f'dim_{i}'] = z[:, i]
    df.to_csv(OUTPUT_DIR / 'unified_embeddings.csv', index=False)
    print(f"限定地域エンベディングを保存しました: {OUTPUT_DIR / 'unified_embeddings.csv'}")

if __name__ == "__main__":
    main()
