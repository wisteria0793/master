import torch
import logging
from typing import List, Protocol, Union
import numpy as np
from tqdm.auto import tqdm

# オプショナルインポート（インストールされていない場合の対策）
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
except ImportError:
    AutoProcessor, AutoModel = None, None

class Encoder(Protocol):
    """エンコーダーのインターフェース定義"""
    device: torch.device
    def encode(self, inputs: Union[List[str], List['PIL.Image.Image']], batch_size: int, show_progress_bar: bool) -> torch.Tensor: ...
    def get_embedding_dimension(self) -> int: ...

class SentenceTransformerEncoder:
    """テキスト用: SentenceTransformerのラッパー"""
    def __init__(self, model_name: str, device: str = None):
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformers is not installed.")
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = SentenceTransformer(model_name, device=self.device)
        logging.info(f"Loaded SentenceTransformer: {model_name} on {self.device}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class CLIPTextEncoder:
    """テキスト用: CLIPのテキストエンコーダーラッパー"""
    def __init__(self, model_name: str, device: str = None):
        if AutoProcessor is None:
            raise ImportError("Transformers is not installed.")
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embedding_dim = self.model.config.text_config.hidden_size
        logging.info(f"Loaded CLIP Text Model: {model_name} on {self.device}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = True) -> torch.Tensor:
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="CLIP Text Encoding")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                all_embeddings.append(text_features.cpu())
        
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

class CLIPImageEncoder:
    """画像用: CLIPの画像エンコーダーラッパー"""
    def __init__(self, model_name: str, device: str = None):
        if AutoProcessor is None:
            raise ImportError("Transformers is not installed.")
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        except Exception:
            # Fallback for generic Auto classes if specific CLIP classes fail
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
        self.embedding_dim = self.model.config.vision_config.hidden_size
        logging.info(f"Loaded CLIP Image Model: {model_name} on {self.device}")

    def encode(self, images: List['PIL.Image.Image'], batch_size: int = 32, show_progress_bar: bool = True) -> torch.Tensor:
        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="CLIP Image Encoding")

        with torch.no_grad():
            for i in iterator:
                batch_imgs = images[i:i+batch_size]
                inputs = self.processor(images=batch_imgs, return_tensors="pt", padding=True).to(self.device)
                image_features = self.model.get_image_features(**inputs)
                all_embeddings.append(image_features.cpu())
        
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

def get_encoder(model_type: str, model_name: str, modality: str = "text") -> Encoder:
    """ファクトリ関数"""
    if model_type == "sentence-transformer" and modality == "text":
        return SentenceTransformerEncoder(model_name)
    elif model_type == "clip":
        if modality == "text":
            return CLIPTextEncoder(model_name)
        elif modality == "image":
            return CLIPImageEncoder(model_name)
    
    raise ValueError(f"Unsupported model_type: {model_type} or modality: {modality}")
