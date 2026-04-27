# 修正内容の確認 (Walkthrough): 実験18.11 統合学習対照実験

## 実施内容
実験18.3（景観保存型統合GNN）の質を検証するため、POIの事前処理（事前GNNによる空間平滑化）の有無による対照実験を実施しました。

### 1. 比較パターンの構築
同一の統合学習モデル（Multimodal Residual GAT）を用い、入力となるPOIデータの性質のみを変化させました。

- **Proposed (事前処理あり)**: 
    - `train_poi_gnn.py` で生成された、空間文脈を含む64次元埋め込みを入力。
- **Baseline (事前処理なし)**: 
    - 紹介文ベクトルと時間情報を直接クラスタリングした10次元シードを入力し、空間的な前処理を一切排除。

### 2. 学習と地図化
- 両パターンについて、200エポックの学習を実行し、最終的な埋め込みを抽出。
- 地球統計学的な連続性を比較するため、同一のクラスタ数（k=20）で地図化。

## 検証結果 (比較用地図)
| パターン | 地図へのリンク |
| :--- | :--- |
| **Baseline (Raw)** | [map_18_3_baseline_raw_seeds.html](../../docs/results/residual_learning_comparison/map_18_3_baseline_raw_seeds.html) |
| **Proposed (Pre-GNN)** | [map_18_3_proposed_pre_gnn.html](../../docs/results/residual_learning_comparison/map_18_3_proposed_pre_gnn.html) |

## 技術的改善点
- **モデルの柔軟性向上**: `MultimodalResidualGATEncoder` の入力次元が固定（799/768）されていた問題を修正し、任意のシード次元（例: 今回のBaselineの10次元）を受け入れられるように `models.py` をアップデートしました。
- **データアライメント**: フィルタリングによって地点数が異なるPOIデータセット（702地点 vs 374地点）を、`original_idx` に基づいて動的に整合させるロジックを実装しました。
