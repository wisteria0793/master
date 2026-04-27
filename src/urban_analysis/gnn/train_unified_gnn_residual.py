# -*- coding: utf-8 -*-
"""
実験18.3: 景観保存型（Residual GAT）を用いた特定地域限定の統合GNN学習スクリプト。
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
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ハイパーパラメータ
EMBEDDING_DIM = 64
SHARED_DIM = 128
LEARNING_RATE = 0.002
N_EPOCHS = 200
DISTANCE_THRESHOLD = 150 # より局所的な差異を強調するため150mに短縮

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. データの読み込み
    print("限定地域データを読み込み中...")
    poi_feats = np.load(GNN_FILTERED_DIR / 'poi_features.npy')
    sv_feats = np.load(GNN_FILTERED_DIR / 'sv_features.npy')
    with open(GNN_FILTERED_DIR / 'nodes_metadata.json', 'r') as f:
        nodes_meta = json.load(f)
    
    n_poi = len(poi_feats)
    n_sv = len(sv_feats)
    
    x_combined = np.zeros((n_poi + n_sv, 799), dtype=np.float32)
    x_combined[:n_poi, :799] = poi_feats
    x_combined[n_poi:, :768] = sv_feats
    
    node_types = torch.cat([torch.zeros(n_poi, dtype=torch.long), torch.ones(n_sv, dtype=torch.long)])
    
    # 2. 地理的グラフの構築
    print(f"距離しきい値 {DISTANCE_THRESHOLD}m で高密度グラフを構築中...")
    coords = np.array([[n['lat'], n['lng']] for n in nodes_meta])
    coords_m = coords * np.array([111000, 82000])
    
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(DISTANCE_THRESHOLD)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # 重複の削除と自己ループの防止 (念のため)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    
    # 3. 特徴量の標準化
    scaler_poi = StandardScaler()
    scaler_sv = StandardScaler()
    x_combined[:n_poi, :799] = scaler_poi.fit_transform(x_combined[:n_poi, :799])
    x_combined[n_poi:, :768] = scaler_sv.fit_transform(x_combined[n_poi:, :768])
    
    x_tensor = torch.from_numpy(x_combined).to(device)
    edge_index = edge_index.to(device)
    node_types = node_types.to(device)
    data = Data(x=x_tensor, edge_index=edge_index)
    
    # 4. 残差型モデルの学習
    # GAEの中で MultimodalResidualGATEncoder を使用
    encoder = MultimodalResidualGATEncoder(799, 768, SHARED_DIM, EMBEDDING_DIM)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"景観保存型学習開始 ({N_EPOCHS} epochs)...")
    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        # forwardにnode_typesを渡す必要がある
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
        
    np.save(OUTPUT_DIR / 'residual_embeddings.npy', z)
    df = pd.DataFrame(nodes_meta)
    for i in range(EMBEDDING_DIM):
        df[f'dim_{i}'] = z[:, i]
    df.to_csv(OUTPUT_DIR / 'residual_embeddings.csv', index=False)
    print(f"景観保存型埋め込みを保存しました: {OUTPUT_DIR / 'residual_embeddings.csv'}")

if __name__ == "__main__":
    main()
