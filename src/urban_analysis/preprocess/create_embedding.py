
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from tqdm.auto import tqdm

# 共通設定とコンポーネントのインポート
from urban_analysis.config import PROCESSED_DATA_DIR
from urban_analysis.preprocess.embeddings import get_encoder, Encoder

# --- 設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_MAP = {
    "sentence-transformer": "cl-tohoku/bert-base-japanese-whole-word-masking",
    "clip": "openai/clip-vit-base-patch32"
}

CONFIG = {
    "input_json_path": PROCESSED_DATA_DIR / "poi/filtered_facilities.json",
    "model": "sentence-transformer",
    "model_name": MODEL_MAP["sentence-transformer"],
    "type_extraction_key": "categories",
    "output_dir": None,
    "desc_embeddings_filename": "facility_embeddings.npy",
    "simple_subtracted_filename": "facility_embeddings_simple_sub.npy",
    "projected_subtracted_filename": "facility_embeddings_projected_sub.npy",
}

def load_facilities(file_path: Path) -> List[Dict[str, Any]]:
    """JSONファイルから施設データを読み込む"""
    logging.info(f"'{file_path}' から施設データを読み込みます...")
    if not file_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {file_path}")
    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"{len(data)}件の施設データを読み込みました。")
    return data

def extract_data(facilities: List[Dict[str, Any]], type_key: Any) -> Tuple[List[str], List[List[str]], List[str]]:
    """データ抽出"""
    descriptions, types_list, facility_names = [], [], []
    for item in tqdm(facilities, desc="データ抽出中"):
        desc_text = " ".join(item['description']) if isinstance(item.get('description'), list) else item.get('description', '') or item.get('description_short', '')
        descriptions.append(desc_text)
        
        temp_item = item
        if isinstance(type_key, list):
            for key in type_key: temp_item = temp_item.get(key, {})
            types = temp_item if isinstance(temp_item, list) else []
        else:
            types = item.get(type_key, [])
        types_list.append(types)
        facility_names.append(item.get('name', 'Unnamed Facility'))
    return descriptions, types_list, facility_names

def create_averaged_type_embeddings(encoder: Encoder, types_list: List[List[str]]) -> torch.Tensor:
    """カテゴリ埋め込みの平均化"""
    averaged_embeddings = []
    embedding_dim = encoder.get_embedding_dimension()

    for types in tqdm(types_list, desc="カテゴリ埋め込みの平均化"):
        if types:
            type_vectors = encoder.encode(types, show_progress_bar=False)
            averaged_vector = torch.mean(type_vectors, dim=0)
            averaged_embeddings.append(averaged_vector)
        else:
            averaged_embeddings.append(torch.zeros(embedding_dim, device=encoder.device))
            
    return torch.stack(averaged_embeddings)

def subtract_orthogonally(vectors_a: torch.Tensor, vectors_b: torch.Tensor) -> torch.Tensor:
    """直交射影による減算"""
    dot_product = torch.sum(vectors_a * vectors_b, dim=1, keepdim=True)
    b_norm_sq = torch.sum(vectors_b * vectors_b, dim=1, keepdim=True)
    projection_scale = dot_product / (b_norm_sq + 1e-8)
    projection = projection_scale * vectors_b
    return vectors_a - projection

def save_embeddings(embeddings: np.ndarray, file_path: Path):
    """保存"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, embeddings)
    logging.info(f"埋め込みを '{file_path}' に保存しました。")

def main():
    try:
        facilities = load_facilities(CONFIG["input_json_path"])
        descriptions, types_list, facility_names = extract_data(facilities, CONFIG["type_extraction_key"])

        # 共通モジュールからエンコーダーを取得
        encoder = get_encoder(CONFIG["model"], CONFIG["model_name"], modality="text")

        logging.info("説明文の埋め込みを生成しています...")
        description_embeddings = encoder.encode(descriptions)

        logging.info("カテゴリの埋め込みを生成・平均化しています...")
        type_embeddings = create_averaged_type_embeddings(encoder, types_list)

        logging.info("減算処理を実行...")
        simple_sub_embeddings = description_embeddings - type_embeddings
        projected_sub_embeddings = subtract_orthogonally(description_embeddings, type_embeddings)

        output_dir = CONFIG["output_dir"]
        save_embeddings(description_embeddings.cpu().numpy(), output_dir / CONFIG["desc_embeddings_filename"])
        save_embeddings(simple_sub_embeddings.cpu().numpy(), output_dir / CONFIG["simple_subtracted_filename"])
        save_embeddings(projected_sub_embeddings.cpu().numpy(), output_dir / CONFIG["projected_subtracted_filename"])

        logging.info("処理完了。")

    except Exception as e:
        logging.error(f"エラー発生: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=CONFIG["model"], choices=MODEL_MAP.keys())
    args = parser.parse_args()

    CONFIG["model"] = args.model
    CONFIG["model_name"] = MODEL_MAP[args.model]
    
    sanitized_model_name = CONFIG["model_name"].replace("/", "_")
    CONFIG["output_dir"] = PROCESSED_DATA_DIR / "embedding" / sanitized_model_name
    
    main()
