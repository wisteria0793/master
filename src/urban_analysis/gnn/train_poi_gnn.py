# -*- coding: utf-8 -*-
"""
このスクリプトは、POI (Point of Interest) のテキスト埋め込みと地理的関係を統合し、
Graph Autoencoder (GAE) を用いて各POIの新しい低次元特徴量を学習します。

処理手順：
1. 施設データ (filtered_facilities.json) とテキスト埋め込み (facility_embeddings.npy) を読み込みます。
2. OSMデータから道路ネットワークに基づいたグラフ、または地理的距離に基づいたグラフを構築します。
3. PyTorch Geometric用のデータオブジェクトを作成します。
4. GCNベースのGAEモデルを学習させます。
5. 学習済みのエンコーダを使って、各POIの統合されたエンベディングを抽出し、保存します。
"""

import pandas as pd
import json
import numpy as np
import torch
import os
from pathlib import Path
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# コンポーネントのインポート
try:
    from urban_analysis.config import (
        PROCESSED_DATA_DIR, 
        RAW_DATA_DIR, 
        OSM_XML_PATH,
        PROJECT_ROOT
    )
    from urban_analysis.gnn.models import create_gae_model
    from urban_analysis.gnn.graph_builder import build_graph_from_osm
except ImportError:
    # 直接実行する場合の相対パス対応（必要に応じて調整）
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from urban_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, OSM_XML_PATH, PROJECT_ROOT
    from urban_analysis.gnn.models import create_gae_model
    from urban_analysis.gnn.graph_builder import build_graph_from_osm

# --- 設定 ---
N_EPOCHS = 200
LEARNING_RATE = 0.01
EMBEDDING_DIM = 64
DISTANCE_THRESHOLD_METERS = 300 # 地理的近接エッジのしきい値を300mに縮小
USE_TEMPORAL_FEATURES = True  # 営業時間・営業日を特徴量に含める場合は True
TEMPORAL_WEIGHT = 50.0         # 時間的特徴の重要度を大幅に強化
TEMPORAL_EDGE_FILTER = True   # 営業時間が重ならないエッジを削除
TEMPORAL_OVERLAP_THRESHOLD = 0.3 # 重なりのしきい値を厳格化（0.3）

# --- パス設定 ---
# PROCESSED_DATA_DIR が 'new' を向いている場合があるため、プロジェクト構成に合わせて調整
REAL_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
POI_JSON_PATH = REAL_PROCESSED_DIR / 'poi' / 'filtered_facilities.json'
# NOTE: フォルダ名は sentence-transformer ですが、中身は multilingual-e5-base のベクトルです
TEXT_EMB_PATH = REAL_PROCESSED_DIR / 'embedding' / 'sentence-transformer' / 'facility_embeddings.npy'
OUTPUT_DIR = REAL_PROCESSED_DIR / 'gnn_embeddings'

# 特徴量構成に応じてファイル名を変更
if USE_TEMPORAL_FEATURES:
    suffix = f"w{TEMPORAL_WEIGHT}"
    if TEMPORAL_EDGE_FILTER:
        suffix += f"_f{TEMPORAL_OVERLAP_THRESHOLD}"
    prefix = f'hakodate_temporal_{suffix}'
else:
    prefix = 'hakodate_spatial'
EMBEDDING_OUTPUT_PATH = OUTPUT_DIR / f'{prefix}_poi_gnn_embeddings.csv'
EMBEDDING_NPY_PATH = OUTPUT_DIR / f'{prefix}_poi_gnn_embeddings.npy'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_temporal_features(poi):
    """営業時間を24次元、営業日を7次元のバイナリベクトルに変換する"""
    hours_vec = np.zeros(24)
    days_vec = np.zeros(7)
    
    oh = poi.get('google_places_data', {}).get('details', {}).get('opening_hours', {})
    periods = oh.get('periods')
    
    if not periods:
        # データがない場合の補完 (9時-17時をデフォルトとする)
        cats = poi.get('categories', [])
        if any(c in str(cats) for c in ['居酒屋', 'バー', '夜']):
            hours_vec[18:24] = 1
            hours_vec[0:2] = 1
        elif any(c in str(cats) for c in ['朝市', '市場']):
            hours_vec[5:12] = 1
        else:
            hours_vec[9:18] = 1
        days_vec[:] = 1 # 基本毎日営業と仮定
        return np.concatenate([hours_vec, days_vec])

    for p in periods:
        # 営業日の記録
        if 'open' in p:
            d = p['open'].get('day')
            if d is not None:
                days_vec[d % 7] = 1
            
            # 営業時間の記録 (簡易的に全曜日の総和をとる)
            try:
                open_t = int(p['open']['time'][:2])
                if 'close' in p:
                    close_t = int(p['close']['time'][:2])
                    if close_t < open_t: # 日を跨ぐ場合
                        hours_vec[open_t:24] = 1
                        hours_vec[0:close_t] = 1
                    else:
                        hours_vec[open_t:close_t] = 1
                else:
                    hours_vec[open_t:24] = 1 # 24時間営業など
            except (ValueError, KeyError):
                pass
                
    return np.concatenate([hours_vec, days_vec])

def load_poi_data():
    """POIデータとテキスト埋め込みを読み込み、Dataフレームと特徴量行列を返す"""
    print(f"POIデータを読み込み中: {POI_JSON_PATH}")
    with open(POI_JSON_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    poi_list = []
    for i, poi in enumerate(pois):
        # 函館市のみを抽出
        addr = poi.get('google_places_data', {}).get('find_place_formatted_address', '')
        details_addr = poi.get('google_places_data', {}).get('details', {}).get('formatted_address', '')
        if '函館市' not in addr and '函館市' not in details_addr:
            continue
            
        # 座標の抽出
        geom = poi.get('google_places_data', {}).get('find_place_geometry', {}).get('location', {})
        lat = geom.get('lat')
        lng = geom.get('lng')
        
        # 時間特徴の抽出
        temp_feat = extract_temporal_features(poi)
        
        poi_list.append({
            'point_id': i, 
            'name': poi.get('name', f'POI_{i}'),
            'latitude': lat,
            'longitude': lng,
            'temp_feat': temp_feat
        })
    
    df = pd.DataFrame(poi_list)
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    print(f"テキスト埋め込みを読み込み中: {TEXT_EMB_PATH}")
    text_embeddings = np.load(TEXT_EMB_PATH)
    
    # 座標が欠損していた地点、および函館市外のフィルタリング結果に基づくインデックスを取得
    valid_indices = df['point_id'].values
    
    # テキスト埋め込みと時間特徴を結合
    text_feats = text_embeddings[valid_indices]
    
    if USE_TEMPORAL_FEATURES:
        print(f"時間特徴（営業時間・営業日）を統合中... (Weight: {TEMPORAL_WEIGHT})")
        temp_feats = np.stack(df['temp_feat'].values)
        # combined_features の中ではまだ生値（または生時系列）を使用
        combined_features = np.concatenate([text_feats, temp_feats], axis=1)
    else:
        print("空間・意味特徴のみを使用中...")
        combined_features = text_feats
    
    # dfのpoint_idをリセットして0から連番にする（グラフ構築用）
    df = df.reset_index(drop=True)
    df['original_idx'] = valid_indices
    
    return df, combined_features

def build_proximity_graph(df, features):
    """地理的な距離に基づいてエッジを構築する (KDTree)"""
    print(f"地理的距離（{DISTANCE_THRESHOLD_METERS}m）に基づいてグラフを構築中...")
    
    # 緯度・経度をメートル近似に変換するための係数（函館付近 41.7N）
    # lat 1度 = 約111km, lng 1度 = 約111km * cos(41.7) = 約83km
    coords = df[['latitude', 'longitude']].values
    
    # 簡易的なユークリッド距離計算のためのスケーリング（厳密なHaversineではないが、接続関係の定義には十分）
    scale_coords = coords.copy()
    scale_coords[:, 0] *= 111000
    scale_coords[:, 1] *= 83000
    
    tree = KDTree(scale_coords)
    
    # 各点から指定距離内の近傍を検索
    pairs = list(tree.query_pairs(DISTANCE_THRESHOLD_METERS))
    
    # 営業時間に基づいたエッジフィルタリング
    final_edges = []
    if TEMPORAL_EDGE_FILTER and USE_TEMPORAL_FEATURES:
        print(f"時間的エッジフィルタリングを適用中... (Threshold: {TEMPORAL_OVERLAP_THRESHOLD})")
        temp_feats = np.stack(df['temp_feat'].values)
        # 営業時間（0-23次元）のみを取り出し、バイナリ化
        # (Jaccard係数的な重複度合いを計算)
        hours_bits = temp_feats[:, 0:24] > 0.5
        
        filtered_count = 0
        for u, v in pairs:
            # 論理和と論理積で重なりを評価
            intersection = np.logical_and(hours_bits[u], hours_bits[v]).sum()
            union = np.logical_or(hours_bits[u], hours_bits[v]).sum()
            
            # 分母が0（両方不明など）の場合は、とりあえず接続維持
            jaccard = intersection / union if union > 0 else 1.0
            
            if jaccard >= TEMPORAL_OVERLAP_THRESHOLD:
                final_edges.append((u, v))
            else:
                filtered_count += 1
        print(f"フィルタリング完了: {filtered_count} 個のエッジを削除しました。")
    else:
        final_edges = pairs

    # 無向グラフにするため逆方向も追加
    undirected_edges = []
    for u, v in final_edges:
        undirected_edges.append((u, v))
        undirected_edges.append((v, u))
    
    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    
    # 特徴量の標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 標準化の「後」に重みを適用する（重要：これによって重みが正規化に打ち消されない）
    if USE_TEMPORAL_FEATURES:
        # テキスト埋め込み（768次元）の後の31次元（24+7）が時間特徴
        text_dim = 768 # or features.shape[1] - 31
        features_scaled[:, text_dim:] *= TEMPORAL_WEIGHT
        print(f"時間軸に重み {TEMPORAL_WEIGHT} を適用しました。")

    x = torch.tensor(features_scaled, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    print(f"グラフ構築完了。ノード数: {data.num_nodes}, エッジ数: {data.num_edges}")
    return data

def train_model(data):
    """GAEモデルの学習"""
    print("POI GNN (GAE) モデルの学習を開始します...")
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
    plt.title("POI GNN (GAE) Training Loss")
    plt.savefig(OUTPUT_DIR / 'poi_training_loss.png')
    plt.close()

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).detach().cpu().numpy()
    return embeddings

def main():
    points_df, features = load_poi_data()
    
    # OSMではなく距離ベースでエッジを生成
    data = build_proximity_graph(points_df, features)
    
    embeddings = train_model(data)
    
    print(f"学習済み統合エンベディングを保存中... (次元数: {embeddings.shape[1]})")
    
    # CSV形式での保存
    res_df = pd.DataFrame(embeddings)
    res_df['name'] = points_df['name']
    res_df['original_idx'] = points_df['original_idx']
    res_df.to_csv(EMBEDDING_OUTPUT_PATH, index=False)
    
    # numpy形式での保存（ユーザーの要望に近い可能性があるため）
    np.save(EMBEDDING_NPY_PATH, embeddings)
    
    print(f"保存完了: {EMBEDDING_OUTPUT_PATH}")
    print(f"保存完了: {EMBEDDING_NPY_PATH}")

if __name__ == '__main__':
    main()
