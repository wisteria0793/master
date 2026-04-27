# -*- coding: utf-8 -*-
"""
単体GNN学習スクリプト
景観単体・機能単体それぞれに対して GAE (GAT) を適用し、空間的に平滑化された埋め込みを得る。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GAE
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
INPUT_DIR_BASE = PROJECT_ROOT / 'data' / 'processed' / 'gnn_inputs_individual'
OUTPUT_DIR_BASE = PROJECT_ROOT / 'data' / 'processed' / 'gnn_results_individual'

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 32, heads=4, dropout=0.1)
        self.conv2 = GATConv(32 * 4, out_channels, heads=1, concat=False, dropout=0.1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def train_individual(name, in_channels):
    print(f"\n--- {name} のGNN学習を開始 ---")
    data_path = INPUT_DIR_BASE / name
    x = torch.from_numpy(np.load(data_path / 'features.npy')).float()
    edge_index = torch.from_numpy(np.load(data_path / 'edge_index.npy')).long()
    
    # エッジがない場合のガード
    if edge_index.shape[1] == 0:
        print(f"Warning: No edges found for {name}. Skipping.")
        return

    data = Data(x=x, edge_index=edge_index)
    latent_dim = 16
    
    model = GAE(GATEncoder(in_channels, latent_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')
            
    # 埋め込み保存
    save_path = OUTPUT_DIR_BASE / name
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / 'embeddings.npy', z.cpu().detach().numpy())
    print(f"学習完了、埋め込みを保存しました: {save_path}")

def main():
    # 1. 景観単体 (in=12)
    train_individual('landscape', 12)
    
    # 2. 機能単体 (in=8)
    train_individual('function', 8)

if __name__ == "__main__":
    main()
