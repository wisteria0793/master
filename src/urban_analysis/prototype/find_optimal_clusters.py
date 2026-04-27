# -*- coding: utf-8 -*-
"""
最適なクラスタ数 (K) の算出スクリプト
Elbow Method と Silhouette Score を用いて、景観と機能の埋め込みデータの最適な K を特定する。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import os

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
STREETCLIP_EMB = PROJECT_ROOT / 'data' / 'new' / 'streetclip_embeddings' / 'streetclip_embeddings.npy'
FACILITY_EMB = PROJECT_ROOT / 'data' / 'processed' / 'embedding' / 'facility_embeddings_final.npy'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'results' / 'cluster_optimization'

def analyze_k(data, name, max_k=30):
    print(f"分析開始: {name} (データ形状: {data.shape})")
    
    # メモリ節約のため、データが大きい場合はサンプリング（景観データ等）
    if len(data) > 5000:
        print(f"  データが大きいため 5000 サンプルにダウンサンプリングします...")
        indices = np.random.choice(len(data), 5000, replace=False)
        data_sampled = data[indices]
    else:
        data_sampled = data

    distortions = []
    silhouette_avg = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_sampled)
        distortions.append(kmeans.inertia_)
        
        # シルエット係数の計算（計算負荷が高いため注意）
        score = silhouette_score(data_sampled, kmeans.labels_)
        silhouette_avg.append(score)
        print(f"  k={k}: Distortion={kmeans.inertia_:.2f}, Silhouette={score:.4f}")

    # プロット
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Elbow)', color=color)
    ax1.plot(k_range, distortions, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, silhouette_avg, 's-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Cluster Optimization for {name}')
    fig.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR / f'optimization_{name}.png')
    print(f"結果を保存しました: {OUTPUT_DIR / f'optimization_{name}.png'}")
    plt.close()

def main():
    # 1. 景観データの分析
    if STREETCLIP_EMB.exists():
        ls_emb = np.load(STREETCLIP_EMB)
        analyze_k(ls_emb, 'landscape_streetclip', max_k=25)
    else:
        print(f"Warning: {STREETCLIP_EMB} not found.")

    # 2. 機能データの分析
    if FACILITY_EMB.exists():
        poi_emb = np.load(FACILITY_EMB)
        analyze_k(poi_emb, 'function_sentence_bert', max_k=25)
    else:
        print(f"Warning: {FACILITY_EMB} not found.")

if __name__ == "__main__":
    main()
