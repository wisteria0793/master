# Urban Analysis Library for Master Thesis

都市解析、グラフニューラルネットワーク (GNN)、クラスタリング分析などの手法を用いた研究用コードベースです。
Pythonパッケージとして構造化されており、再利用可能なコンポーネントを提供します。

## 📦 ディレクトリ構成

`src/urban_analysis/` がメインパッケージです。

| ディレクトリ | 役割 | 主な機能 |
|---|---|---|
| **`gnn/`** | グラフニューラルネットワーク | GAEモデル (`models.py`), OSMグラフ構築 (`graph_builder.py`) |
| **`data_analysis/`** | データ分析・可視化 | クラスタリング (`clustering.py`), セグメンテーション (`segmentation.py`), 指標計算 (`metrics.py`) |
| **`preprocess/`** | 前処理 | 埋め込み生成 (`embeddings.py`), データ整形 |
| **`collect_data/`** | データ収集 | Google Street View, OSMデータの取得など |
| `config.py` | 設定 | パスや定数の一元管理 |

## 🚀 セットアップ

### 1. 仮想環境の作成とインストール
依存関係の競合を防ぐため、仮想環境の使用を推奨します。

```bash
# 仮想環境の作成
python3 -m venv .venv

# 仮想環境の有効化 (Mac/Linux)
source .venv/bin/activate

# ライブラリのインストール (Editableモード)
pip install -e .
```

### 2. データセットの準備
デフォルトでは、データは `data/` ディレクトリに配置されることを想定しています。
パスを変更する場合は、環境変数 `DATA_DIR` を設定するか、`src/urban_analysis/config.py` を編集してください。

## 📖 使い方 (Examples)

### 1. グラフニューラルネットワーク (GNN) の学習

OSM道路網データと画像特徴量を使って、Graph Autoencoder (GAE) を学習します。

```bash
# GNN学習スクリプトの実行
python -m urban_analysis.gnn.train_graph_autoencoder
```

学習済みエンベディングは `data/processed/gnn_embeddings/` に保存されます。

### 2. 階層的クラスタリングの実行

任意の埋め込みベクトル（.npyファイル）を入力として階層的クラスタリングを実行し、クラスタラベルを出力します。

```bash
# 埋め込みファイルを指定して実行
python -m urban_analysis.data_analysis.hierarchical_clustering \
    data/processed/embedding/sentence-transformer/facility_embeddings.npy \
    --num_clusters 10 \
    --output clusters_10.npy \
    --plot
```

### 3. 画像セグメンテーション (SegFormer)

特定の画像に対してセマンティックセグメンテーションを行い、緑地率などを計算します。
（スクリプト内のパスを書き換えるか、独自のスクリプトから呼び出してください）

```python
from urban_analysis.data_analysis.segmentation import SegmentationModel, calculate_class_ratios

model = SegmentationModel()
image, mask = model.segment("path/to/image.jpg")
ratios = calculate_class_ratios(mask, model.id2label)
print(ratios)
```

### 4. クラスタリングの定量評価 (シルエットスコア)

生成された埋め込みベクトルとクラスタラベルを用いて、シルエットスコアを計算します。
スコアは -1 から 1 の範囲で、1に近いほどクラスタが適切に分離されていることを示します。

```bash
# .npy ファイルを使用して計算
python -m urban_analysis.evaluation.calculate_silhouette \
    --embeddings data/processed/embedding/sentence-transformer/facility_embeddings.npy \
    --labels data/processed/hierarchical_clustering/vector_sentence_bert_10.npy

# CSVファイル（列名 'cluster'）からラベルを読み込む場合
python -m urban_analysis.evaluation.calculate_silhouette \
    --embeddings data/processed/embedding/sentence-transformer/facility_embeddings.npy \
    --labels data/processed/hierarchical_clustering/locations_with_clusters_hc_10_with_address.csv \
    --label_col cluster
```

### 5. ネットワークカーネル密度推定 (NKDE)

道路ネットワーク上でのイベント（景観クラスタなど）の密度を計算し、ホットスポットを分析します。

```python
import osmnx as ox
from urban_analysis.data_analysis.network_kde import network_kernel_density

# 1. 道路ネットワークの読み込み
G = ox.graph_from_xml("path/to/hakodate.osm.xml", simplify=True)

# 2. イベントデータ（地点と最近傍OSMノード）の準備
# event_points: pd.DataFrame (columns=['osm_node', ...])
event_points = get_your_event_data() 

# 3. NKDEの計算
bandwidth = 500  # バンド幅（メートル）
edge_densities = network_kernel_density(G, event_points, bandwidth)

# edge_densitiesは {(u, v, k): density, ...} の辞書形式で返されます
```

## 🛠 コンポーネント開発ガイド

### 新しいモデルの追加
`src/urban_analysis/gnn/models.py` に新しいPyTorchモジュールクラスを追加してください。

### 新しい分析手法の追加
`src/urban_analysis/data_analysis/` に新しいモジュールを作成し、`__init__.py` で公開してください。

---
Author: Atsuya Katougi
Based on master thesis project.
