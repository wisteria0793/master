# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from pathlib import Path
import sys

USE_TEMPORAL_FEATURES = True # True にすると営業時間統合版を解析
TEMPORAL_WEIGHT = 50.0
TEMPORAL_EDGE_FILTER = True
TEMPORAL_OVERLAP_THRESHOLD = 0.3
MANUAL_K = 12

# パス設定
SRC_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
POI_PATH = DATA_DIR / 'poi' / 'filtered_facilities.json'

# 特徴量構成に応じてファイル名を変更
if USE_TEMPORAL_FEATURES:
    suffix = f"w{TEMPORAL_WEIGHT}"
    if TEMPORAL_EDGE_FILTER:
        suffix += f"_f{TEMPORAL_OVERLAP_THRESHOLD}"
    prefix = f'hakodate_temporal_{suffix}'
else:
    prefix = 'hakodate_spatial'

GNN_EMBEDDING_PATH = DATA_DIR / 'gnn_embeddings' / f'{prefix}_poi_gnn_embeddings.npy'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_hex_colors(n):
    """N個のHEXカラーコードを取得する"""
    cmap = plt.get_cmap('tab20')
    if n <= 20:
        return [mcolors.to_hex(cmap(i / 20)) for i in range(n)]
    else:
        cmap_large = plt.get_cmap('gist_ncar')
        return [mcolors.to_hex(cmap_large(i / n)) for i in range(n)]

def load_data():
    print(f"データを読み込み中: {POI_PATH}")
    with open(POI_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    # 函館市のみを抽出
    poi_list = []
    for i, poi in enumerate(pois):
        addr = poi.get('google_places_data', {}).get('find_place_formatted_address', '')
        details_addr = poi.get('google_places_data', {}).get('details', {}).get('formatted_address', '')
        if '函館市' not in addr and '函館市' not in details_addr:
            continue
            
        geom = poi.get('google_places_data', {}).get('find_place_geometry', {}).get('location', {})
        lat = geom.get('lat')
        lng = geom.get('lng')
        if lat and lng:
            poi_list.append({
                'original_idx': i,
                'name': poi.get('name', f'POI_{i}'),
                'lat': lat,
                'lng': lng,
                'categories': ", ".join(poi.get('categories', [])),
                'opening_hours': ", ".join(poi.get('google_places_data', {}).get('details', {}).get('opening_hours', {}).get('weekday_text', ['不明']))
            })
    
    df = pd.DataFrame(poi_list)
    print(f"GNN埋め込みを読み込み中: {GNN_EMBEDDING_PATH}")
    all_embeddings = np.load(GNN_EMBEDDING_PATH)
    
    # train_poi_gnn.py では座標があるものだけ抜いて連番に直して学習しているため、
    # ここでも同様のフィルタリング後の順序になっているはず。
    # (train_poi_gnn.py の load_poi_data 参照)
    return df, all_embeddings

def find_optimal_k(embeddings, min_k=5, max_k=40):
    print(f"クラスタ数 k を {min_k} から {max_k} まで変化させてシルエットスコアを計算中...")
    linked = linkage(embeddings, method='ward')
    
    k_values = range(min_k, max_k + 1)
    scores = []
    
    for k in k_values:
        labels = fcluster(linked, k, criterion='maxclust')
        score = silhouette_score(embeddings, labels)
        scores.append(score)
        print(f"k={k:2d}: Silhouette Score = {score:.4f}")
        
    optimal_k = k_values[np.argmax(scores)]
    print(f"\n=> 最適なクラスタ数: {optimal_k} (最大スコア: {max(scores):.4f})")
    
    # グラフの保存
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, marker='o')
    plt.title('Silhouette Scores for POI GNN Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = OUTPUT_DIR / f'{prefix}_poi_gnn_silhouette_scores.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"シルエットスコアの推移グラフを保存しました: {plot_path}")
    
    return optimal_k, linked

def create_map(df, labels, optimal_k):
    print(f"マップ (k={optimal_k}) を生成中...")
    df['cluster'] = labels
    
    colors = get_hex_colors(optimal_k)
    m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=13, tiles='CartoDB positron')
    
    feature_groups = {}
    for i in range(optimal_k):
        # 1-indexed for display
        fg = folium.FeatureGroup(name=f"Cluster {i+1}")
        feature_groups[i+1] = fg
        m.add_child(fg)

    for _, row in df.iterrows():
        cluster_id = int(row['cluster']) # 1 to optimal_k
        color = colors[cluster_id - 1]
        
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 250px;">
            <h4 style="margin-bottom: 5px;">{row['name']}</h4>
            <p style="margin: 0;"><b>GNNクラスタ:</b> {cluster_id}</p>
            <p style="margin: 0; font-size: 12px; color: #555;"><b>カテゴリ:</b> {row['categories']}</p>
            <hr style="margin: 5px 0;">
            <p style="margin: 0; font-size: 11px; color: #777;"><b>営業時間:</b><br>{row['opening_hours'].replace(', ', '<br>')}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=6,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1
        ).add_to(feature_groups[cluster_id])
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    map_path = OUTPUT_DIR / f'{prefix}_poi_gnn_cluster_map_k{optimal_k}.html'
    m.save(str(map_path))
    print(f"地図上にプロットしたHTMLマップを保存しました: {map_path}")
    
    # 結果CSVの保存
    csv_path = DATA_DIR / 'gnn_embeddings' / f'{prefix}_poi_gnn_clusters_k{optimal_k}.csv'
    df.to_csv(csv_path, index=False)
    print(f"クラスタ結果をCSV保存しました: {csv_path}")

def main():
    df, embeddings = load_data()
    
    linked = linkage(embeddings, method='ward')
    
    if MANUAL_K is not None:
        print(f"クラスタ数を固定値 k={MANUAL_K} に設定してプロットします...")
        optimal_k = MANUAL_K
    else:
        # 最適kの探索
        optimal_k, _ = find_optimal_k(embeddings)
    
    # 最終的なクラスタリング
    labels = fcluster(linked, optimal_k, criterion='maxclust')
    
    # 地図生成
    create_map(df, labels, optimal_k)

if __name__ == '__main__':
    main()
