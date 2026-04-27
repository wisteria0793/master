# -*- coding: utf-8 -*-
"""
実験18.9: 統合分析スクリプト。
シード（POI/景観クラスタID）をグラフ上でGNN伝播させ、最終的な「統合地区」を作成し比較する。
Aパターン（POI-GNNあり） vs Bパターン（POI-GNNなし）
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import linkage, fcluster
import folium
import matplotlib.pyplot as plt
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
GNN_FILTERED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_filtered'
SEED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'dual_seeds'
OUTPUT_DIR = PROJECT_ROOT / 'docs/results/comparison_approach_b' # 以前のディレクトリを再利用
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLUSTERS_FINAL = 20
DISTANCE_THRESHOLD = 150

class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def run_integration(seed_type='a'):
    print(f"\n--- パターン {seed_type.upper()}（POI-GNN {'あり' if seed_type == 'a' else 'なし'}）の統合実行 ---")
    
    # 1. データの読み込み
    with open(GNN_FILTERED_DIR / 'nodes_metadata.json', 'r', encoding='utf-8') as f:
        nodes_meta = json.load(f)
    df_poi_seeds = pd.read_csv(SEED_DIR / 'poi_seeds.csv')
    df_sv_seeds = pd.read_csv(SEED_DIR / 'sv_seeds.csv')
    
    n_nodes = len(nodes_meta)
    
    # 2. グラフ構築
    coords = np.array([[n['lat'], n['lng']] for n in nodes_meta])
    coords_m = coords * np.array([111000, 82000])
    tree = KDTree(coords_m)
    pairs = tree.query_pairs(DISTANCE_THRESHOLD)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # 3. 特徴量（シードID）の作成
    # POIシード(10種) + 景観シード(10種) = 20次元の One-hot 的な特徴量
    x = np.zeros((n_nodes, 20), dtype=np.float32)
    
    # POI側
    seed_col = 'seed_a_id' if seed_type == 'a' else 'seed_b_id'
    for i, row in df_poi_seeds.iterrows():
        # nodes_meta内のでのインデックスを探す (POIは前半にあると仮定、またはIDで紐付け)
        # 今回は単純化のため、nodes_metaの順序がpoi -> svである前提で処理（prepare_filteredでそのように作成）
        x[i, row[seed_col]] = 1.0 # 0-9次元
        
    # SV側
    n_poi = len(df_poi_seeds)
    for i, row in df_sv_seeds.iterrows():
        offset = n_poi
        x[offset + i, 10 + row['landscape_seed_id']] = 1.0 # 10-19次元
        
    # 4. GNNによる空間平滑化
    # 学習なしの固定GCNを通すことで、近傍のクラスタID情報を伝播配分させる
    device = torch.device('cpu')
    x_tensor = torch.from_numpy(x).to(device)
    edge_index = edge_index.to(device)
    
    # シンプルな2層GCNで、各ノードに「周囲のクラスタ構成」を埋め込む
    # 重みはランダム初期化で固定(Seed固定)
    torch.manual_seed(42)
    model = SimpleGNN(20, 32, 16)
    model.eval()
    with torch.no_grad():
        z = model(x_tensor, edge_index).numpy()
        
    # 5. 最終クラスタリング
    linked = linkage(z, method='ward')
    final_clusters = fcluster(linked, N_CLUSTERS_FINAL, criterion='maxclust') - 1
    
    df_result = pd.DataFrame(nodes_meta)
    df_result['cluster_id'] = final_clusters
    return df_result

def create_map(df, title, filename):
    print(f"地図生成中: {filename}")
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=15)
    cmap = plt.cm.get_cmap('tab20', N_CLUSTERS_FINAL)
    
    for _, row in df.iterrows():
        color = '#%02x%02x%02x' % tuple(int(x*255) for x in cmap(row['cluster_id'])[:3])
        if row['type'] == 'poi':
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=8, color='black', weight=1, fill=True, fill_color=color, fill_opacity=1.0,
                popup=f"POI: {row['name']} (Cluster: {row['cluster_id']})"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.4,
                popup=f"SV: Cluster {row['cluster_id']}"
            ).add_to(m)
    
    m.save(OUTPUT_DIR / filename)

def main():
    # AとBの両方を実行
    df_a = run_integration(seed_type='a')
    df_b = run_integration(seed_type='b')
    
    create_map(df_a, "Pattern A (POI-GNN YES)", "integrated_map_pattern_a.html")
    create_map(df_b, "Pattern B (POI-GNN NO)", "integrated_map_pattern_b.html")
    
    print(f"\n完了。2つの比較地図を保存しました: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
