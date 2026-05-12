# Phase 4 実装タスクリスト

- `[x]` `phase4_recommender.py` の作成と実装
  - `phase3_recommender.py` をコピー
  - クラス名を `Phase4Recommender` 等に変更
  - `recommend` メソッドで「景観クラスタのみの一致」に変更
  - 「同一機能クラスタの上限（例：最大2件）」を設ける多様性フィルター（Diversity Filter）を実装
- `[x]` `run_phase4.py` の作成と実装
  - `run_phase3.py` をコピー
  - `Phase4Recommender` を利用するように変更
  - ログ出力やマップ出力ファイル名を `phase4_` 等に変更
- `[x]` `README.md` の更新
  - Phase 4 の説明（多様性考慮・景観ベース推薦）を追加
- `[x]` 実行テスト・検証
  - コマンドを実行し、多様な機能の施設が選出されているか、マップが出力されるか確認
