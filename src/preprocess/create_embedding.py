# -*- coding: utf-8 -*-
"""
施設データから説明文とカテゴリの埋め込み表現を生成し、
施設の機能的側面を取り除いた特徴ベクトルを生成するスクリプト。

■ 変更点
- コマンドライン引数を `--model` に簡略化。
- 減算ロジックを2種類実装:
  1. 単純減算 (A - B)
  2. 直交射影による減算 (AからBの方向成分を削除)
- モデルの抽象化:
  引数に応じてSentenceTransformerとCLIPのテキストエンコーダーを切り替え可能。
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Protocol

import numpy as np
import torch
from tqdm.auto import tqdm

# --- オプショナルなライブラリのインポート ---
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    AutoProcessor, AutoModel = None, None


# --- 設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# モデルタイプと具体的なモデル名の対応表
MODEL_MAP = {
    "sentence-transformer": "cl-tohoku/bert-base-japanese-whole-word-masking",
    "clip": "openai/clip-vit-base-patch32"
}

# デフォルト設定
CONFIG = {
    "input_json_path": "data/processed/filtered_facilities.json",
    # "output_dir": "data/processed/embedding/sentence-bert",      # デフォルトの保存フォルダ
    "model": "sentence-transformer",    # デフォルトのモデルタイプ
    "model_name": MODEL_MAP["sentence-transformer"], # デフォルトのモデル名
    "type_extraction_key": "categories",
    "output_dir": None, # 実行時に動的に設定
    
    # --- 出力ファイル名 ---
    "desc_embeddings_filename": "facility_embeddings.npy",
    "simple_subtracted_filename": "facility_embeddings_simple_sub.npy",
    "projected_subtracted_filename": "facility_embeddings_projected_sub.npy",
}


# --- モデル処理の抽象化 ---

class Encoder(Protocol):
    """エンコーダーが持つべきメソッドを定義するプロトコル。"""
    device: torch.device
    def encode(self, texts: List[str], batch_size: int, show_progress_bar: bool) -> torch.Tensor: ...
    def get_embedding_dimension(self) -> int: ...

class SentenceTransformerEncoder:
    """SentenceTransformerモデルをラップするクラス。"""
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformersがインストールされていません。`pip install sentence-transformers`を実行してください。")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        logging.info(f"SentenceTransformerモデル '{model_name}' をデバイス '{self.device}' にロードしました。")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class CLIPTextEncoder:
    """CLIPのテキストエンコーダーをラップするクラス。"""
    def __init__(self, model_name: str):
        if AutoProcessor is None or AutoModel is None:
            raise ImportError("Transformersがインストールされていません。`pip install transformers`を実行してください。")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embedding_dim = self.model.config.text_config.hidden_size
        logging.info(f"CLIPモデル '{model_name}' をデバイス '{self.device}' にロードしました。")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> torch.Tensor:
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="CLIPエンコード中")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                all_embeddings.append(text_features.cpu())
        
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

def get_encoder(model_type: str, model_name: str) -> Encoder:
    """設定に応じて適切なエンコーダークラスのインスタンスを返すファクトリ関数。"""
    if model_type == "sentence-transformer":
        return SentenceTransformerEncoder(model_name)
    elif model_type == "clip":
        # output_dir = "data/processed/embedding/clip"
        return CLIPTextEncoder(model_name)
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")


# --- コアロジック関数 ---

def load_facilities(file_path: Path) -> List[Dict[str, Any]]:
    """JSONファイルから施設データのリストを読み込む。"""
    logging.info(f"'{file_path}' から施設データを読み込みます...")
    if not file_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {file_path}")
    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"{len(data)}件の施設データを読み込みました。")
    return data

def extract_data(facilities: List[Dict[str, Any]], type_key: Any) -> Tuple[List[str], List[List[str]], List[str]]:
    """施設のリストから説明文、カテゴリ、名前を抽出する。"""
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

    for i, name in enumerate(facility_names[:5]):
        print(name, types_list[i], descriptions[i])
    return descriptions, types_list, facility_names

def create_averaged_type_embeddings(encoder: Encoder, types_list: List[List[str]]) -> torch.Tensor:
    """各施設のカテゴリリストから埋め込みを生成し、平均化する。"""
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
    """ベクトルAから、ベクトルBの方向成分を直交射影を用いて取り除く。"""
    dot_product = torch.sum(vectors_a * vectors_b, dim=1, keepdim=True)
    b_norm_sq = torch.sum(vectors_b * vectors_b, dim=1, keepdim=True)
    projection_scale = dot_product / (b_norm_sq + 1e-8)
    projection = projection_scale * vectors_b
    return vectors_a - projection

def save_embeddings(embeddings: np.ndarray, file_path: Path):
    """NumPy配列をファイルに保存する。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, embeddings)
    logging.info(f"埋め込みを '{file_path}' に保存しました。")


def main():
    """メインの実行関数。"""
    try:
        # --- 1. データ読み込みと抽出 ---
        facilities = load_facilities(Path(CONFIG["input_json_path"]))
        descriptions, types_list, facility_names = extract_data(facilities, CONFIG["type_extraction_key"])

        # --- 2. モデルのロード ---
        encoder = get_encoder(CONFIG["model"], CONFIG["model_name"])

        # --- 3. 埋め込みの生成 ---
        logging.info("説明文の埋め込みを生成しています...")
        description_embeddings = encoder.encode(descriptions)

        logging.info("カテゴリの埋め込みを生成・平均化しています...")
        type_embeddings = create_averaged_type_embeddings(encoder, types_list)

        # --- 4. 2種類の方法で減算 ---
        logging.info("2種類の方法で減算を実行します...")
        simple_sub_embeddings = description_embeddings - type_embeddings
        projected_sub_embeddings = subtract_orthogonally(description_embeddings, type_embeddings)
        logging.info("減算完了。")

        # --- 5. 結果の保存 ---
        output_dir = CONFIG["output_dir"]
        save_embeddings(description_embeddings.cpu().numpy(), output_dir / CONFIG["desc_embeddings_filename"])
        save_embeddings(simple_sub_embeddings.cpu().numpy(), output_dir / CONFIG["simple_subtracted_filename"])
        save_embeddings(projected_sub_embeddings.cpu().numpy(), output_dir / CONFIG["projected_subtracted_filename"])

        # --- 6. 結果の検証表示 ---
        logging.info("--- 結果の検証（最初の施設）---")
        print(f"施設名: {facility_names[0]}")
        print(f"元の説明文埋め込み (先頭5次元): {description_embeddings[0, :5].cpu().numpy()}")
        print(f"カテゴリ埋め込み (先頭5次元): {type_embeddings[0, :5].cpu().numpy()}")
        print(f"単純減算後の埋め込み (先頭5次元): {simple_sub_embeddings[0, :5].cpu().numpy()}")
        print(f"直交射影後の埋め込み (先頭5次元): {projected_sub_embeddings[0, :5].cpu().numpy()}")

    except (FileNotFoundError, ValueError, ImportError) as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)


if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="施設データから特徴ベクトルを生成するスクリプト。")
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["model"],
        choices=MODEL_MAP.keys(),
        help=f"使用するモデルの種類。選択肢: {list(MODEL_MAP.keys())}。デフォルト: {CONFIG['model']}"
    )
    args = parser.parse_args()

    # 引数に基づいてCONFIGを更新
    CONFIG["model"] = args.model
    CONFIG["model_name"] = MODEL_MAP[args.model]

    # モデル名に基づいて出力先ディレクトリを動的に設定
    sanitized_model_name = CONFIG["model_name"].replace("/", "_")
    CONFIG["output_dir"] = Path("data/processed/embedding") / sanitized_model_name
    
    logging.info(f"コマンドライン引数により、モデルタイプ: '{CONFIG['model']}', モデル名: '{CONFIG['model_name']}' で実行します。")
    logging.info(f"出力先ディレクトリ: '{CONFIG['output_dir']}'")
    
    main()

"""
実行コマンド例
python src/preprocess/create_embedding.py --model clip
"""