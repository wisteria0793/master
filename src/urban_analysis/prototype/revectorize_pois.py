import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# パス設定
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
POI_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'poi', 'filtered_facilities.json')
SAVE_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'embedding', 'sentence-transformer', 'facility_embeddings.npy')

# モデル設定 (E5は "passage: " プレフィックスを付けることで検索性能が向上する)
MODEL_NAME = 'intfloat/multilingual-e5-base'

def revectorize():
    # 1. データの読み込み
    print(f"データを読み込み中: {POI_PATH}")
    with open(POI_PATH, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    # 2. 文章の構築
    # 施設名、カテゴリ、概要を組み合わせる
    # description（詳細）がある場合はそれを優先し、なければ description_short を使う
    texts = []
    for poi in pois:
        name = poi.get('name', '')
        cats = ", ".join(poi.get('categories', []))
        
        # 説明文の選択ロジック
        full_desc_list = poi.get('description', [])
        if isinstance(full_desc_list, list) and len(full_desc_list) > 0:
            desc = " ".join(full_desc_list)
        else:
            desc = poi.get('description_short', '')
        
        # E5用のプレフィックスを付与
        # 検索対象（データベース）なので "passage: " を付ける
        input_text = f"passage: 施設名: {name}. カテゴリ: {cats}. 概要: {desc}"
        texts.append(input_text)
    
    print(f"{len(texts)} 件の文章を構築しました。")
    
    # 3. モデルのロードと推論
    # 仮想環境のライブラリを使用するため、このスクリプトは ./venv/bin/python3 で実行すること
    print(f"モデル '{MODEL_NAME}' をロードして推論を開始します（この処理には時間がかかる場合があります）...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    # 4. 保存
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.save(SAVE_PATH, embeddings)
    print(f"再ベクトル化が完了しました。保存先: {SAVE_PATH}")
    print(f"ベクトル数: {embeddings.shape[0]}, 次元数: {embeddings.shape[1]}")

if __name__ == "__main__":
    revectorize()
