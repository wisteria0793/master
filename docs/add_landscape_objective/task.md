# タスクリスト: Phase 5への景観目的関数の追加

- [x] `phase5_recommender.py` の修正
  - [x] 候補POIに対する景観密度スコア (`ls_density`) の計算処理追加
  - [x] スコアの正規化処理追加
  - [x] `_objective_function` を4次元に変更
  - [x] `MultiObjProblem` を `n_obj=4` に変更
  - [x] デフォルトの重み設定、参照点の生成 (`n_dim=4`) の更新
  - [x] 最終解の選択処理と返り値の更新
- [x] `run_phase5.py` の動作検証
  - [x] スクリプトの実行
  - [x] 期待通りの出力（マップ生成、ログ）の確認
- [x] `walkthrough.md` の作成
