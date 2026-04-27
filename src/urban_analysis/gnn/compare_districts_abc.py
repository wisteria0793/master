# -*- coding: utf-8 -*-
"""
実験18.10: 統合手法の5パターン比較（A/B/C/D/E）
A: シード (POI-GNNあり)
B: シード (POI-GNNなし)
C: 直接統合 (Raw) ※パターンEの別名
D: 直接統合 (POI-GNNあり/64d)
E: 直接統合 (POI-GNNなし/799d)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import linkage, fcluster
import folium
import matplotlib.pyplot as plt
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_FILTERED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
RESIDUAL_A_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual'
SEED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'dual_seeds'
OUTPUT_DIR = PROJECT_ROOT / 'docs/results/comparison_approach_abc_de'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS_FINAL = 20
DISTANCE_THRESHOLD = 150

class SmoothingGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SmoothingGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class DirectProjectionGNN(nn.Module):
    def __init__(self, poi_dim, sv_dim, mid_dim, out_dim):
        super(DirectProjectionGNN, self).__init__()
        self.poi_proj = nn.Linear(poi_dim, mid_dim)
        self.sv_proj = nn.Linear(sv_dim, mid_dim)
        self.conv1 = GCNConv(mid_dim, mid_dim)
        self.conv2 = GCNConv(mid_dim, out_dim)
    
    def forward(self, x_poi, x_sv, edge_index, node_types):
        h = torch.zeros((len(node_types), self.poi_proj.out_features), device=x_poi.device)
        h[node_types == 0] = self.poi_proj(x_poi)
        h[node_types == 1] = self.sv_proj(x_sv)
        
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h

def load_base_data():
    with open(GNN_FILTERED_DIR / 'nodes_metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    df_poi_seeds = pd.read_csv(SEED_DIR / 'poi_seeds.csv')
    df_sv_seeds = pd.read_csv(SEED_DIR / 'sv_seeds.csv')
    
    coords = np.array([[n['lat'], n['lng']] for n in meta])
    coords_m = coords * np.array([111000, 82000])
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(DISTANCE_THRESHOLD)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return meta, df_poi_seeds, df_sv_seeds, edge_index

def run_pattern_seed(meta, df_poi_seeds, df_sv_seeds, edge_index, mode='a'):
    print(f"実行中: パターン {mode.upper()} (Seed-based)")
    n_nodes = len(meta)
    x = np.zeros((n_nodes, 20), dtype=np.float32)
    
    poi_col = 'seed_a_id' if mode == 'a' else 'seed_b_id'
    for i, row in df_poi_seeds.iterrows():
        x[i, row[poi_col]] = 1.0
    
    n_poi = len(df_poi_seeds)
    for i, row in df_sv_seeds.iterrows():
        x[n_poi + i, 10 + row['landscape_seed_id']] = 1.0
        
    torch.manual_seed(42)
    model = SmoothingGNN(20, 32, 16)
    with torch.no_grad():
        z = model(torch.from_numpy(x), edge_index).numpy()
    
    linked = linkage(z, method='ward')
    clusters = fcluster(linked, N_CLUSTERS_FINAL, criterion='maxclust') - 1
    return clusters

def run_pattern_direct(meta, edge_index, poi_mode='raw'):
    """
    poi_mode: 'raw' (799d) or 'gnn' (64d)
    """
    print(f"実行中: パターン {'D' if poi_mode == 'gnn' else 'E'} (Direct / POI-{poi_mode.upper()})")
    
    if poi_mode == 'raw':
        poi_feats = np.load(GNN_FILTERED_DIR / 'poi_features.npy')
    else:
        # 実験18.3で作ったGNN埋め込み(64d)を読み込む
        df_gnn = pd.read_csv(RESIDUAL_A_DIR / 'residual_embeddings.csv')
        poi_feats = df_gnn[df_gnn['type'] == 'poi'][[c for c in df_gnn.columns if c.startswith('dim_')]].values
        
    sv_feats = np.load(GNN_FILTERED_DIR / 'sv_features.npy')
    
    n_poi = len(poi_feats)
    n_sv = len(sv_feats)
    node_types = torch.cat([torch.zeros(n_poi, dtype=torch.long), torch.ones(n_sv, dtype=torch.long)])
    
    poi_dim = poi_feats.shape[1]
    torch.manual_seed(42)
    model = DirectProjectionGNN(poi_dim, 768, 64, 16)
    model.eval()
    
    with torch.no_grad():
        z = model(
            torch.from_numpy(poi_feats.astype(np.float32)),
            torch.from_numpy(sv_feats.astype(np.float32)),
            edge_index,
            node_types
        ).numpy()
        
    linked = linkage(z, method='ward')
    clusters = fcluster(linked, N_CLUSTERS_FINAL, criterion='maxclust') - 1
    return clusters

def create_map(meta, clusters, title, filename):
    print(f"地図生成: {filename}")
    df = pd.DataFrame(meta)
    df['cluster_id'] = clusters
    
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=15)
    cmap = plt.cm.get_cmap('tab20', N_CLUSTERS_FINAL)
    
    for _, row in df.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap(row['cluster_id'])[:3])
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=8 if row['type'] == 'poi' else 4,
            color='black' if row['type'] == 'poi' else color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=1.0 if row['type'] == 'poi' else 0.5,
            popup=f"{row['type'].upper()}: {row.get('name', row.get('id'))} (C: {row['cluster_id']})"
        ).add_to(m)
    
    m.save(OUTPUT_DIR / filename)

def main():
    meta, df_poi, df_sv, edge_index = load_base_data()
    
    # カテゴリ1: シードベース
    c_a = run_pattern_seed(meta, df_poi, df_sv, edge_index, mode='a')
    create_map(meta, c_a, "Pattern A", "map_pattern_a_seed_gnn.html")
    
    c_b = run_pattern_seed(meta, df_poi, df_sv, edge_index, mode='b')
    create_map(meta, c_b, "Pattern B", "map_pattern_b_seed_raw.html")
    
    # カテゴリ2: 直接統合
    c_d = run_pattern_direct(meta, edge_index, poi_mode='gnn')
    create_map(meta, c_d, "Pattern D", "map_pattern_d_direct_gnn.html")
    
    c_e = run_pattern_direct(meta, edge_index, poi_mode='raw')
    create_map(meta, c_e, "Pattern E", "map_pattern_e_direct_raw.html")
    
    # 旧パターンC (Eと同じ)
    import shutil
    shutil.copy(OUTPUT_DIR / "map_pattern_e_direct_raw.html", OUTPUT_DIR / "map_pattern_c_fully_direct.html")
    
    print(f"\nすべての地図(A, B, C, D, E)を保存しました: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
