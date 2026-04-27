# POI GNN埋め込み再生成の実装計画

以前実施されていた、施設紹介文のテキスト情報とPOI間の地理的関係をGNN（Graph Neural Network）で統合し、各POIのノード埋め込みを導き出すプロセスを再現します。

## ユーザーレビューが必要な事項

> [!IMPORTANT]
> - 施設間のエッジ（接続）を定義する際の距離しきい値（例: 300m〜500m）や、近傍ノード数 $k$ の設定について、以前の実験設定があればそれに合わせる必要があります。今回は標準的な設定（500m以内）で構築する予定です。
> - 使用するモデルは、既存のストリートビュー画像のGNN学習でも利用されている **GAE (Graph AutoEncoder)** を想定しています。

## 提案される変更

### 1. 基盤となるテキスト埋め込みの復旧
既存のスクリプトを実行し、GNNの入力となるテキストベクトルを生成します。

#### [RUN] [revectorize_pois.py](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/prototype/revectorize_pois.py)
- `intfloat/multilingual-e5-base` モデルを使用して、`filtered_facilities.json` から `facility_embeddings.npy` を生成します。

### 2. POI GNN学習スクリプトの新規作成
POIをノード、地理的接近性をエッジ、テキストベクトルを特徴量とするGNN学習パイプラインを構築します。

#### [NEW] [train_poi_gnn.py](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/gnn/train_poi_gnn.py)
- `filtered_facilities.json` から各POIの座標を取得。
- `facility_embeddings.npy` をノード特徴量としてロード。
- `scipy.spatial.KDTree` を使用して、一定距離内の施設間にエッジを張る。
- PyTorch Geometric (PyG) の `GAE` モデルを構成。
- 学習を実行し、空間的な近接性と意味的な類似性の両方を考慮した低次元（例: 64次元）の埋め込みを抽出・保存する。

## オープンな質問

> [!NOTE]
> 以前の `facility_embeddings.npy` は「テキストのみ」のベクトルでしたか、それとも「GNN後の統合ベクトル」でしたか？
> ユーザーの依頼文では「GNNで地理的な関係やテキスト間の関連を導き出した」結果を再現したいとのことですので、GNN後の出力を `poi_gnn_embeddings.npy` のような名前で保存し、テキストのみのベクトルと区別することを推奨します。

## 検証計画

### 自動テスト / 実行検証
- `train_poi_gnn.py` の実行により、損失関数が減少していることを確認。
- 生成された埋め込みファイル（`.npy`）の形状が POI数 × 指定次元数 になっていることを確認。
- 類似する地点（地理的に近く、内容も似ている）が埋め込み空間上で近くなっています。

### 手動検証
- クラスタリング結果の視覚化（既存の `visualize_gnn_clusters.py` のロジックを流用可能）。
