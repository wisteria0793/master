import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

# プロジェクトのルートディレクトリ
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

POI_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', 'filtered_facilities.json')
TEXT_EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'embedding', 'sentence-transformer', 'facility_embeddings.npy')
OUTPUT_PLOT_PATH = os.path.join(BASE_DIR, 'docs', 'results', 'poi_text_silhouette_scores.png')

def load_filtered_data():
    with open(POI_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    valid_indices = []
    exclude_keywords = ["閉店", "休業", "休館"]
    
    coords = []
    
    for i, poi in enumerate(pois):
        name = poi.get('name', '')
        if any(keyword in name for keyword in exclude_keywords):
            continue
            
        geom = poi.get('google_places_data', {}).get('find_place_geometry', {}).get('location', {})
        lat = geom.get('lat')
        lng = geom.get('lng')
        if lat and lng:
            valid_indices.append(i)
            coords.append([lat, lng])
            
    all_embeddings = np.load(TEXT_EMBEDDING_PATH)
    return all_embeddings[valid_indices], np.array(coords)

def main():
    print("データとエンベディングを読み込み中...")
    embeddings, coords = load_filtered_data()
    
    # 緯度経度をラジアンに変換してhaversine距離を計算するための準備
    coords_rad = np.radians(coords)

    print("階層的クラスタリング (Ward法) を実行中...")
    linked = linkage(embeddings, method='ward')
    
    min_k = 2
    max_k = 40
    k_values = range(min_k, max_k + 1)
    
    semantic_scores = []
    physical_scores = []
    
    print(f"クラスタ数 k を {min_k} から {max_k} まで変化させてシルエットスコアを計算中...")
    
    for k in k_values:
        labels = fcluster(linked, k, criterion='maxclust')
        
        # 意味的距離（コサイン）でのシルエットスコアを計算
        s_score = silhouette_score(embeddings, labels, metric='cosine')
        semantic_scores.append(s_score)
        
        # 物理的距離（Haversine）でのシルエットスコアを計算
        p_score = silhouette_score(coords_rad, labels, metric='haversine')
        physical_scores.append(p_score)
        
        print(f"k={k:2d}: Semantic Score = {s_score:.4f}, Physical Score = {p_score:.4f}")
        
    optimal_k = k_values[np.argmax(semantic_scores)]
    max_score = max(semantic_scores)
    print(f"\n=> 意味的距離における最適なクラスタ数: {optimal_k} (最大スコア: {max_score:.4f})")

    # プロットの作成 (2軸グラフ)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Semantic Silhouette Score (Cosine)', color=color)
    ax1.plot(k_values, semantic_scores, marker='o', linestyle='-', color=color, label='Semantic (Cosine)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k (Semantic) = {optimal_k}')

    # 右側のY軸を追加
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Physical Silhouette Score (Haversine)', color=color)
    ax2.plot(k_values, physical_scores, marker='s', linestyle='-', color=color, label='Physical (Haversine)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Silhouette Scores for POI Text Clustering vs Number of Clusters')
    ax1.set_xticks(np.arange(min_k, max_k + 1, 2))
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 凡例の結合
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    plt.close()
    
    print(f"シルエットスコアの推移グラフを保存しました: {OUTPUT_PLOT_PATH}")

if __name__ == '__main__':
    main()