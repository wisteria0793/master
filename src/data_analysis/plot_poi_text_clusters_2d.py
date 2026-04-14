import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
POI_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', 'filtered_facilities.json')
TEXT_EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'embedding', 'sentence-transformer', 'facility_embeddings.npy')

def load_filtered_embeddings():
    with open(POI_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    valid_indices = []
    exclude_keywords = ["閉店", "休業", "休館"]
    
    for i, poi in enumerate(pois):
        name = poi.get('name', '')
        if any(keyword in name for keyword in exclude_keywords):
            continue
            
        geom = poi.get('google_places_data', {}).get('find_place_geometry', {}).get('location', {})
        if geom.get('lat') and geom.get('lng'):
            valid_indices.append(i)
            
    all_embeddings = np.load(TEXT_EMBEDDING_PATH)
    return all_embeddings[valid_indices]

def main(n_clusters=20):
    CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', f'poi_text_clusters_{n_clusters}.csv')
    OUTPUT_PLOT_PATH = os.path.join(BASE_DIR, 'docs', 'results', f'poi_text_clusters_tsne_{n_clusters}.png')

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} が見つかりません。先にクラスタリングを実行してください。")
        return
        
    print(f"データとエンベディングを読み込み中... (k={n_clusters})")
    df = pd.read_csv(CSV_PATH)
    embeddings = load_filtered_embeddings()
    
    if len(df) != len(embeddings):
        print(f"エラー: データフレームの行数({len(df)})とエンベディングの数({len(embeddings)})が一致しません。")
        return

    print("t-SNEによる次元削減を実行中 (少し時間がかかります)...")
    # t-SNEを使って768次元(?)のSentence-BERTベクトルを2次元に圧縮
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)
    
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    
    print("2Dプロットを描画・保存中...")
    plt.figure(figsize=(16, 12))
    
    # クラスタ数に応じたカラーパレットを用意
    n_clusters_actual = df['text_cluster'].nunique()
    palette = sns.color_palette("tab20", n_colors=max(n_clusters_actual, 20))
    
    # 散布図の描画
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='text_cluster',
        palette=palette[:n_clusters_actual],
        alpha=0.8,
        s=60,
        edgecolor='w',
        linewidth=0.5,
        legend='full'
    )
    
    plt.title(f't-SNE Projection of POI Text Embeddings (Sentence-BERT, k={n_clusters})', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    # 凡例を枠外に配置
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Text Cluster', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    plt.close()
    
    print(f"クラスタリング結果の2Dプロットを保存しました: {OUTPUT_PLOT_PATH}")

if __name__ == '__main__':
    n_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n_clusters)