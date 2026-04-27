# 函館市内 POI 限定 GNN クラスタリングの実装計画

施設データから「函館市」に所在するもののみを抽出し、それらを対象とした GNN 学習およびクラスタリング可視化を再実行します。

## ユーザーレビューが必要な事項

> [!IMPORTANT]
> - 対象 POI セットが変更（821件 → 702件）になるため、GNN のグラフ構造（エッジ）も変化します。そのため、GNN 学習 (`train_poi_gnn.py`) から再実行する必要があります。
> - 出力ファイル名には `hakodate_` プレフィックスを付与し、既存の全体データと区別します。

## 提案される変更

### 1. GNN 学習スクリプトの修正と実行
フィルタリング機能を備えた学習スクリプトに更新し、函館市内限定の埋め込みを生成します。

#### [MODIFY] [train_poi_gnn.py](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/gnn/train_poi_gnn.py)
- `load_poi_data` 関数に、住所に「函館市」が含まれるかどうかのフィルタリング処理を追加。
- 出力ファイル名を `hakodate_poi_gnn_embeddings.npy` などに変更。

### 2. クラスタ分析・可視化スクリプトの修正と実行
フィルタリングされた埋め込みに対応させ、出力を函館市限定版として生成します。

#### [MODIFY] [analyze_poi_gnn_clusters.py](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/gnn/analyze_poi_gnn_clusters.py)
- 読み込み対象を `hakodate_poi_gnn_embeddings.npy` に変更。
- プロット図および HTML マップの保存名を `hakodate_poi_gnn_cluster_map.html` 等に変更。

## オープンな質問

> [!NOTE]
> - 今回、クラスタ数は再度シルエットスコアで最適値を算出しますが、以前の全体データ ($k=9$) から大きく変わる可能性があります。算出された最適値をそのまま採用する方針でよろしいでしょうか。

## 検証計画

### 自動テスト / 実行検証
- スクリプト実行ログにおいて、処理対象の POI 数が 702 件（または住所一致件数）であることを確認。
- `hakodate_` プレフィックスの付いたファイルが正しく生成されることを確認。

### 手動検証
- 生成された [hakodate_poi_gnn_cluster_map.html](file:///Users/atsuyakatougi/Desktop/master/docs/results/hakodate_poi_gnn_cluster_map.html) を開き、市外（北斗市や七飯町など）の POI が含まれていないことを確認。
