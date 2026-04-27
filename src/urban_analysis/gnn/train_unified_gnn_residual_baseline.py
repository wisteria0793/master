# -*- coding: utf-8 -*-
"""
実験18.11 Baseline: 単純クラスタリング済みPOI(10d)を用いた、18.3形式の統合学習(GAE)スクリプト。
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
import sys
from pathlib import Path
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
# models.py があるディレクトリをパスに追加
sys.path.append(str(PROJECT_ROOT / 'src' / 'urban_analysis' / 'gnn'))

from models import MultimodalResidualGATEncoder

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_FILTERED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
BASELINE_SEED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_baseline_18_3'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual_baseline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ハイパーパラメータ (18.3に準拠)
EMBEDDING_DIM = 64
SHARED_DIM = 128
LEARNING_RATE = 0.002
N_EPOCHS = 200
DISTANCE_THRESHOLD = 150

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. データの読み込み
    print("Baseline(18.3形式)データを読み込み中...")
    # 生データの単純クラスタリング結果 (10d)
    poi_seeds = np.load(BASELINE_SEED_DIR / 'raw_poi_seeds.npy') 
    sv_feats = np.load(GNN_FILTERED_DIR / 'sv_features.npy') # 768d
    with open(GNN_FILTERED_DIR / 'nodes_metadata.json', 'r') as f:
        nodes_meta = json.load(f)
    
    n_poi = len(poi_seeds)
    n_sv = len(sv_feats)
    node_types = torch.cat([torch.zeros(n_poi, dtype=torch.long), torch.ones(n_sv, dtype=torch.long)])
    
    # 2. 地理的グラフの構築
    coords = np.array([[n['lat'], n['lng']] for n in nodes_meta])
    coords_m = coords * np.array([111000, 82000])
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(DISTANCE_THRESHOLD)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    
    # 3. 特徴量の整形
    # 景観側は標準化
    scaler_sv = StandardScaler()
    sv_feats_scaled = scaler_sv.fit_transform(sv_feats)
    
    # 全体をSVの次元(768d)に合わせてパディング
    x_combined = np.zeros((n_poi + n_sv, 768), dtype=np.float32)
    x_combined[:n_poi, :10] = poi_seeds # 10次元シード
    x_combined[n_poi:, :768] = sv_feats_scaled
    
    x_tensor = torch.from_numpy(x_combined).to(device)
    edge_index = edge_index.to(device)
    node_types = node_types.to(device)
    data = Data(x=x_tensor, edge_index=edge_index)
    
    # 4. 学習
    # インプット次元は POI=10, SV=768
    encoder = MultimodalResidualGATEncoder(10, 768, SHARED_DIM, EMBEDDING_DIM)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Baseline(18.3形式)学習開始 ({N_EPOCHS} epochs)...")
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
        
    np.save(OUTPUT_DIR / 'residual_embeddings_baseline.npy', z)
    df = pd.DataFrame(nodes_meta)
    for i in range(EMBEDDING_DIM):
        df[f'dim_{i}'] = z[:, i]
    df.to_csv(OUTPUT_DIR / 'residual_embeddings_baseline.csv', index=False)
    print(f"Baseline用埋め込みを保存しました: {OUTPUT_DIR / 'residual_embeddings_baseline.csv'}")

if __name__ == "__main__":
    main()
