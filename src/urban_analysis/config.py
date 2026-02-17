
import os
from pathlib import Path

# プロジェクトのルートディレクトリを自動取得（このファイルがある場所の親ディレクトリ）
# src/urban_analysis/config.py -> src/urban_analysis -> src -> PROJECT_ROOT
SRC_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_ROOT.parent.parent

# データディレクトリの設定
DATA_DIR = os.getenv("DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = Path(DATA_DIR) / "raw"
PROCESSED_DATA_DIR = Path(DATA_DIR) / "new"

# 具体的なパス設定
OSM_XML_PATH = RAW_DATA_DIR / "osm_hakodate" / "Hakodate.osm.xml"
STREET_VIEW_IMAGES_DIR = RAW_DATA_DIR / "street_view_images_50m_optimized"

# 埋め込みデータのパスなど
EMBEDDING_DIR = PROCESSED_DATA_DIR / "embedding"

def get_data_path(relative_path: str) -> Path:
    """データディレクトリからの相対パスを絶対パスに変換"""
    return Path(DATA_DIR) / relative_path
