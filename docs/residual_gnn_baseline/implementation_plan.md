# 実験18.11: 統合学習(18.3)におけるPOI事前処理の有効性検証

## 目的
実験18.3（景観保存型統合GNN）において、入力となるPOIデータに「単体GNNによる事前空間平滑化」を施すことが、最終的な統合地区の質にどのような影響を与えるかを検証する。

## 比較対象の定義
- **Proposed (18.3 Pre-processed)**:
    - POI入力: `train_poi_gnn.py` によって空間平滑化された64次元埋め込み。
    - 景観入力: StreetView生特徴量 (768d)
    - 統合法: GAE (Residual GAT) による統合学習
- **Baseline (18.3 Raw-input)**:
    - POI入力: `poi_features.npy` (799d) ※生属性データ
    - 景観入力: StreetView生特徴量 (768d)
    - 統合法: GAE (Residual GAT) による統合学習（※現状の18.3の実装）

## 追加・修正ファイル

### [NEW] [train_unified_gnn_residual_proposed.py](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/gnn/train_unified_gnn_residual_proposed.py)
事前GNN済みのPOI埋め込みを入力として受け取れるように調整した統合学習スクリプト。

### [NEW] [compare_residual_learning.py](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/gnn/compare_residual_learning.py)
ProposedとBaselineの学習済み埋め込みを読み込み、同一条件でクラスタリング・地図化して比較するスクリプト。

## 期待される効果
POI単体で事前に空間的な関係性を学習しておくことで、景観と統合した際に「機能的なノイズ（周囲と無関係な突出した属性）」が抑えられ、より地理的・文脈的に一貫性のある地区（District）が形成されることを期待する。
