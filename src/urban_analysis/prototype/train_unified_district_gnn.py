# -*- coding: utf-8 -*-
"""
統合地区GNN学習スクリプト
Graph Auto-Encoder (GAE) を用いて、景観と機能のクラスター情報を空間的に融合した
「統合地区埋め込みベクトル」を生成する。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GAE
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results'

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        # 1層目: 25 -> 32 dims, 4 heads
        self.conv1 = GATConv(in_channels, 32, heads=4, dropout=0.1)
        # 2層目: 32*4 -> out_channels, 1 head
        self.conv2 = GATConv(32 * 4, out_channels, heads=1, concat=False, dropout=0.1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def train():
    print("データを読み込み中...")
    x = torch.from_numpy(np.load(INPUT_DIR / 'features.npy')).float()
    edge_index = torch.from_numpy(np.load(INPUT_DIR / 'edge_index.npy')).long()
    
    data = Data(x=x, edge_index=edge_index)
    
    in_channels = x.shape[1]
    out_channels = 16 # 統合ベクトルの次元数
    
    print(f"モデルを構築中 (Input: {in_channels} dims, Latent: {out_channels} dims)...")
    model = GAE(GATEncoder(in_channels, out_channels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    model = model.to(device)
    data = data.to(device)
    
    print("学習開始...")
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')
            
    # 学習完了後の埋め込み抽出
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    
    # 最終的なベクトルの保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / 'unified_district_embeddings.npy', z.cpu().numpy())
    print(f"統合埋め込みを保存しました: {OUTPUT_DIR / 'unified_district_embeddings.npy'}")
    
if __name__ == "__main__":
    train()
