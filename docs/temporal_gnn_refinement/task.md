# GNN時間的特徴反映強化 タスクリスト

## フェーズ1: 時間的特徴量のスケーリング（Temporal Scaling）の実装と評価
- [x] `src/urban_analysis/gnn/train_poi_gnn.py` に `TEMPORAL_WEIGHT` パラメータを導入
- [x] 特徴量結合時にスケーリングを適用するよう修正
- [x] スケーリング有効版での学習実行
- [x] `analyze_poi_gnn_clusters.py` での結果可視化と「朝市 vs 居酒屋」の分離確認

## フェーズ2: 時間的エッジフィルタリング（Temporal Edge Filtering）の実装と評価
- [x] `src/urban_analysis/gnn/graph_builder.py` または `train_poi_gnn.py` のグラフ構築ロジックを特定
- [x] 営業時間ビットセットを用いた同時性（Jaccard係数等）の計算機能を追加
- [x] 同時性が低いペアのエッジを削除するロジックを実装
- [x] エッジフィルタリング有効版での学習実行
- [x] 可視化と評価

## フェーズ3: 比較分析と最終統合
- [x] 2つの手法の効果を比較し、最適なパラメータ/手法を特定
- [x] 最終的な結果を `README.md` に追記
- [x] ドキュメント化（Walkthrough作成）
