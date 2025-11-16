# -*- coding: utf-8 -*-
"""
2種類の減算方法で生成された施設埋め込みベクトルを、
定性的・定量的な側面から評価するためのスクリプト。

■ 評価方法
1. 定性評価（類似度検索）:
   指定した施設（プローブ）に対し、各埋め込み表現空間で類似度が高い施設を検索。
   その結果を比較し、どちらの減算方法がより「意図した特徴」を捉えているかを人間が判断する。

2. 定量評価（クラスタリング）:
   各埋め込み表現をK-Meansでクラスタリングし、シルエットスコアを計算。
   どの埋め込み空間が最も構造的に「まとまりが良い」かを数値で比較する。

■ 使い方
1. 下部の `PROBE_FACILITY_NAMES` に、類似度検索の基準としたい施設名をいくつか設定する。
2. コマンドラインで、評価したいモデルタイプを指定して実行する。
   例: python src/evaluation/evaluate_embeddings.py --model_type sentence-transformer
   例: python src/evaluation/evaluate_embeddings.py --model_type clip
"""

import json
import logging
import argparse
from pathlib import Path
import sys
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 定性評価で類似度検索を試す施設名のリスト ---
# ここに分析したい施設の名前を追加・編集してください
PROBE_FACILITY_NAMES = [
    "基坂",
    "道の駅 なないろ・ななえ",
    "函館朝市",
]

# --- 定量評価で用いるクラスタ数 ---
N_CLUSTERS = 8


# --- メイン処理 ---

def load_data(input_dir: Path, master_json_path: Path) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """埋め込みファイルとマスターJSONファイルを読み込む。"""
    logging.info(f"--- データ読み込み開始 (入力元: {input_dir}) ---")
    
    embedding_paths = {
        "オリジナル": input_dir / "facility_embeddings.npy",
        "単純減算": input_dir / "facility_embeddings_simple_sub.npy",
        "直交射影": input_dir / "facility_embeddings_projected_sub.npy",
    }

    embeddings = {}
    for name, path in embedding_paths.items():
        if not path.exists():
            logging.error(f"埋め込みファイルが見つかりません: {path}")
            sys.exit(1)
        embeddings[name] = np.load(path)
        logging.info(f"'{name}' の埋め込みをロードしました。Shape: {embeddings[name].shape}")

    if not master_json_path.exists():
        logging.error(f"マスターJSONファイルが見つかりません: {master_json_path}")
        sys.exit(1)
        
    with master_json_path.open('r', encoding='utf-8') as f:
        facilities = json.load(f)
    logging.info(f"施設マスターデータをロードしました: {len(facilities)}件")
    
    # 埋め込みと施設データの件数が一致するかチェック
    for name, emb in embeddings.items():
        if len(emb) != len(facilities):
            logging.warning(f"'{name}'の埋め込み件数({len(emb)})と施設数({len(facilities)})が一致しません。create_embedding.pyを再実行してください。")

    return embeddings, facilities


def perform_similarity_search(embeddings: np.ndarray, facilities: List[Dict], probe_name: str, top_n: int = 10):
    """指定された施設と他の全施設との類似度を計算し、トップNを表示する。"""
    facility_names = [f.get("name", "名前なし") for f in facilities]
    
    try:
        probe_index = facility_names.index(probe_name)
    except ValueError:
        logging.warning(f"プローブ施設 '{probe_name}' が見つかりません。スキップします。")
        return

    probe_vector = embeddings[probe_index].reshape(1, -1)
    
    # コサイン非類似度（距離）を計算 (0に近いほど類似)
    distances = pairwise_distances(embeddings, probe_vector, metric='cosine').flatten()
    
    # 類似度順にインデックスをソート (自分自身は除く)
    sorted_indices = np.argsort(distances)[1:top_n + 1]
    
    print(f"--- 「{probe_name}」との類似度トップ{top_n} ---")
    for i, idx in enumerate(sorted_indices):
        similarity = 1 - distances[idx]
        print(f"{i+1:2d}. {facility_names[idx]} (類似度: {similarity:.4f})")
    print("-" * 40)


def perform_clustering_evaluation(embeddings: np.ndarray, n_clusters: int) -> float:
    """K-Meansでクラスタリングし、シルエットスコアを返す。"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # クラスタ数が1以下、または全サンプルが同一クラスタの場合、スコアは計算不可
    if len(np.unique(cluster_labels)) <= 1:
        logging.warning("計算されたクラスタが1つ以下のため、シルエットスコアは計算できません。")
        return -999.0

    score = silhouette_score(embeddings, cluster_labels)
    return score


def main(model_type: str):
    """評価のメインフローを実行する。"""
    
    # --- 0. パスの設定 ---
    # create_embedding.pyからモデル名とパス構造を解決
    sys.path.append(str(Path(__file__).parent.parent.parent))
    # from src.preprocess.create_embedding import MODEL_MAP
    
    # if model_type not in MODEL_MAP:
    #     logging.error(f"モデルタイプ '{model_type}' はMODEL_MAPに定義されていません。")
    #     sys.exit(1)
        
    # model_name = MODEL_MAP[model_type]
    # sanitized_model_name = model_name.replace("/", "_")
    input_dir = Path("data/processed/embedding") / args.model_type
    master_json_path = Path("data/processed/filtered_facilities.json")

    # --- 1. データの読み込み ---
    all_embeddings, facilities = load_data(input_dir, master_json_path)
    facility_names = [f.get("name", "名前なし") for f in facilities]

    # --- 2. 定性評価（類似度検索）---
    print("\n" + "="*60)
    print(" " * 20 + "1. 定性評価：類似度検索")
    print("="*60 + "\n")
    
    for probe_name in PROBE_FACILITY_NAMES:
        print(f"\n▼▼▼ プローブ施設: {probe_name} ▼▼▼")
        for name, embeddings in all_embeddings.items():
            # オリジナルは比較の参考として表示
            print(f"\n[埋め込み: {name}]")
            perform_similarity_search(embeddings, facilities, probe_name)

    # --- 3. 定量評価（クラスタリング）---
    print("\n" + "="*60)
    print(" " * 18 + "2. 定量評価：クラスタリング")
    print("="*60 + "\n")
    
    scores = {}
    for name, embeddings in all_embeddings.items():
        logging.info(f"'{name}' の埋め込みでクラスタリングと評価を実行中...")
        score = perform_clustering_evaluation(embeddings, N_CLUSTERS)
        scores[name] = score

    print("\n--- シルエットスコアによるクラスタリング評価結果 ---")
    print(f"（クラスタ数 k={N_CLUSTERS}）\n")
    for name, score in scores.items():
        print(f"埋め込み: {name:<10s} | シルエットスコア: {score:.4f}")
    print("\n※スコアが高いほど、クラスタが密で、よく分離されていることを示す。")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成された埋め込み表現を定性的・定量的に評価するスクリプト。")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sentence-transformer", "clip"],
        help="評価対象の埋め込みを生成したモデルのタイプ。"
    )
    args = parser.parse_args()
    
    main(args.model_type)