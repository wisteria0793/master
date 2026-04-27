# -*- coding: utf-8 -*-
"""
実験18.11 Proposed: 事前処理済みPOI(64d)を用いた統合学習(GAE)スクリプト。
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
sys.path.append(str(PROJECT_ROOT / 'src' / 'urban_analysis' / 'gnn'))

from models import MultimodalResidualGATEncoder

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_FILTERED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
POI_GNN_EMB_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_embeddings'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual_proposed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ハイパーパラメータ
EMBEDDING_DIM = 64
SHARED_DIM = 128
LEARNING_RATE = 0.002
N_EPOCHS = 200
DISTANCE_THRESHOLD = 150

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. データの読み込み
    print("Proposed用データを読み込み中...")
    # 事前GNN済みデータ (702地点) を読み込む
    df_poi_gnn = pd.read_csv(POI_GNN_EMB_DIR / 'hakodate_temporal_w50.0_f0.3_poi_gnn_embeddings.csv')
    sv_feats = np.load(GNN_FILTERED_DIR / 'sv_features.npy')
    with open(GNN_FILTERED_DIR / 'nodes_metadata.json', 'r') as f:
        nodes_meta = json.load(f)
    
    # 統合学習用メタデータに含まれる374地点のPOIのみを、original_idx をキーにして抽出
    poi_meta = [n for n in nodes_meta if n['type'] == 'poi']
    
    poi_list = []
    missing_count = 0
    for n in poi_meta:
        # id が "poi_123" の形式であることを想定し、123 を抽出
        try:
            o_idx = int(n['id'].split('_')[1])
            row = df_poi_gnn[df_poi_gnn['original_idx'] == o_idx]
            if len(row) > 0:
                feat = row[[f'{i}' for i in range(64)]].values[0].astype(np.float32)
                poi_list.append(feat)
            else:
                # 万が一見つからない場合は、平均値などで埋めるかスキップ（通常はあるはず）
                print(f"Warning: original_idx {o_idx} ({n['name']}) not found in Proposed features.")
                poi_list.append(np.zeros(64, dtype=np.float32))
                missing_count += 1
        except:
            # place_id などが入っている場合
            row = df_poi_gnn[df_poi_gnn['name'] == n['name']]
            if len(row) > 0:
                feat = row[[f'{i}' for i in range(64)]].values[0].astype(np.float32)
                poi_list.append(feat)
            else:
                poi_list.append(np.zeros(64, dtype=np.float32))
                missing_count += 1
    
    poi_feats = np.array(poi_list)
    print(f"POIフィルタリング完了: {len(poi_feats)} 地点 (欠損: {missing_count})")
    
    n_poi = len(poi_feats)
    n_sv = len(sv_feats)
    
    # 特徴量の次元が異なるため、最大次元に合わせてパディングするか、Encoder側で対応する
    # 今回は POI(64d) と SV(768d) なので、それぞれを個別にLinear投影する ResidualGAT を利用可能
    
    node_types = torch.cat([torch.zeros(n_poi, dtype=torch.long), torch.ones(n_sv, dtype=torch.long)])
    
    # 2. 地理的グラフの構築
    coords = np.array([[n['lat'], n['lng']] for n in nodes_meta])
    coords_m = coords * np.array([111000, 82000])
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(DISTANCE_THRESHOLD)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    
    # 3. 特徴量の標準化
    scaler_poi = StandardScaler()
    scaler_sv = StandardScaler()
    poi_feats_scaled = scaler_poi.fit_transform(poi_feats)
    sv_feats_scaled = scaler_sv.fit_transform(sv_feats)
    
    x_combined = np.zeros((n_poi + n_sv, 768), dtype=np.float32) # SVの768dに合わせる
    x_combined[:n_poi, :64] = poi_feats_scaled
    x_combined[n_poi:, :768] = sv_feats_scaled
    
    x_tensor = torch.from_numpy(x_combined).to(device)
    edge_index = edge_index.to(device)
    node_types = node_types.to(device)
    data = Data(x=x_tensor, edge_index=edge_index)
    
    # 4. モデルの学習
    # POIの入力次元は 64 に変更
    encoder = MultimodalResidualGATEncoder(64, 768, SHARED_DIM, EMBEDDING_DIM)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Proposed学習開始 ({N_EPOCHS} epochs)...")
    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()
        # forwardロジック: POI部分(0:64)とSV部分(0:768)を適切に切り出す必要があるためモデルを調整するか
        # Model側の実装に合わせてテンソルを渡す
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
        
    np.save(OUTPUT_DIR / 'residual_embeddings_proposed.npy', z)
    df = pd.DataFrame(nodes_meta)
    for i in range(EMBEDDING_DIM):
        df[f'dim_{i}'] = z[:, i]
    df.to_csv(OUTPUT_DIR / 'residual_embeddings_proposed.csv', index=False)
    print(f"Proposed用埋め込みを保存しました: {OUTPUT_DIR / 'residual_embeddings_proposed.csv'}")

if __name__ == "__main__":
    main()
