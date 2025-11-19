# -*- coding: utf-8 -*-
"""
「静か」「景色が良い」といった抽象的な概念（コンセプト）をベクトル化し、
各埋め込み空間において、その概念に最も近い施設は何かを検索・評価するためのスクリプト。

■ 使い方
1. 下部の `PROBE_CONCEPTS` に、評価したい概念（単語や短い文章）を自由に追加・編集する。
2. コマンドラインで、評価したいモデルタイプを指定して実行する。
   例: python src/evaluation/probe_concepts.py --model_type sentence-transformer
   例: python src/evaluation/probe_concepts.py --model_type clip
"""

import json
import logging
import argparse
from pathlib import Path
import sys
from typing import List, Dict

import numpy as np
from sklearn.metrics import pairwise_distances

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 評価したい概念のリスト ---
# ここに好きな単語や短い文章を追加・編集してください
PROBE_CONCEPTS = [
    "静かな場所",
    "景色が良い",
    "歴史を感じる",
    "モダンでおしゃれ",
    "子供連れで楽しめる",
    "美味しいコーヒーが飲める",
]

# 類似度検索で表示するトップN件
TOP_N = 10


# --- メイン処理 ---

def load_data_and_model(model_type: str):
    """
    必要なデータ（埋め込み、施設情報）とモデルをロードする。
    create_embedding.pyから設定をインポートしてパスを解決する。
    """
    # 親ディレクトリをパスに追加して、他のスクリプトの関数をインポート
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.preprocess.create_embedding import MODEL_MAP, get_encoder

    if model_type not in MODEL_MAP:
        logging.error(f"モデルタイプ '{model_type}' はMODEL_MAPに定義されていません。")
        sys.exit(1)
        
    model_name = MODEL_MAP[model_type]
    # sanitized_model_name = model_name.replace("/", "_")
    input_dir = Path("data/processed/embedding") / model_type
    master_json_path = Path("data/processed/poi/filtered_facilities.json")

    logging.info(f"--- データ読み込み開始 (入力元: {input_dir}) ---")
    
    embedding_paths = {
        "オリジナル": input_dir / "combined_facility_embeddings_02611_05.npy",
        # "単純減算": input_dir / "facility_embeddings_simple_sub.npy",
        # "直交射影": input_dir / "facility_embeddings_projected_sub.npy",
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

    # エンコーダーモデルをロード
    encoder = get_encoder(model_type, model_name)

    return embeddings, facilities, encoder


def main(model_type: str):
    """評価のメインフローを実行する。"""
    
    # --- 1. データの読み込みとモデルの準備 ---
    all_embeddings, facilities, encoder = load_data_and_model(model_type)
    facility_names = [f.get("name", "名前なし") for f in facilities]

    # --- 2. 概念プローブによる類似度検索 ---
    print("\n" + "="*60)
    print(" " * 18 + "概念プローブによる類似度検索")
    print("="*60 + "\n")

    for concept in PROBE_CONCEPTS:
        print(f"\n▼▼▼ 概念プローブ: 「{concept}」 ▼▼▼")
        
        # 概念をベクトル化
        concept_vector = encoder.encode([concept], show_progress_bar=False).cpu().numpy()

        for name, embeddings in all_embeddings.items():
            print(f"\n[埋め込み: {name}]")
            
            # コサイン非類似度（距離）を計算 (0に近いほど類似)
            distances = pairwise_distances(embeddings, concept_vector, metric='cosine').flatten()
            
            # 類似度順にインデックスをソート
            sorted_indices = np.argsort(distances)[:TOP_N]
            
            print(f"--- 「{concept}」に最も近い施設トップ{TOP_N} ---")
            for i, idx in enumerate(sorted_indices):
                similarity = 1 - distances[idx]
                print(f"{i+1:2d}. {facility_names[idx]} (類似度: {similarity:.4f})")
            print("-" * 50)

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抽象的な概念（コンセプト）に基づいて、類似する施設を検索・評価するスクリプト。")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sentence-transformer", "clip"],
        help="評価対象の埋め込みを生成したモデルのタイプ。"
    )
    args = parser.parse_args()
    
    main(args.model_type)
