# -*- coding: utf-8 -*-
"""
クリーンなPOIデータの再ベクトル化スクリプト
759地点の紹介文を用いて Sentence-Transformer ベクトルを生成する。
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# パス設定
PROJECT_ROOT = Path('/Users/atsuyakatougi/Desktop/master')
INPUT_JSON = PROJECT_ROOT / 'data' / 'processed' / 'poi' / 'filtered_facilities_final.json'
OUTPUT_NPY = PROJECT_ROOT / 'data' / 'processed' / 'embedding' / 'facility_embeddings_final.npy'

def main():
    print("クリーンなPOIデータを読み込み中...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 紹介文を抽出し、リスト形式の場合は結合する
    # description がない場合は description_short を代わりに使用
    descriptions = []
    for f in data:
        desc = f.get('description', [])
        if not desc or len(desc) == 0:
            desc = f.get('description_short', '')
            
        if isinstance(desc, list):
            desc = " ".join([str(d) for d in desc])
        descriptions.append(str(desc))
    
    print(f"対象地点数: {len(descriptions)}")
    
    # モデルのロード (以前の解析と整合性をとるため標準的なモデルを使用)
    print("Sentence-Transformerモデルをロード中...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print("ベクトル化を実行中 (GPUがあれば自動で使用されます)...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    # 保存
    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_NPY, embeddings)
    
    print(f"再ベクトル化が完了しました: {OUTPUT_NPY}")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
