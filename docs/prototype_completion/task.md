# 観光ルート推薦プロトタイプ統合 タスクリスト

## フェーズ1: データローダーの刷新と最新データの統合
- [ ] `src/urban_analysis/prototype/data_loader.py` の修正
    - 最新のGNN埋め込み（w50.0_f0.3）の読み込み
    - POIに対する営業時間（24次元ベクトル）の紐付け
    - 景観クラスタ（StreetCLIP）データの整合性確認

## フェーズ2: GNNベースの関連POI推薦機能の実装
- [ ] `src/urban_analysis/prototype/recommender.py` の修正
    - GNN埋め込み（64次元）によるコサイン類似度計算の実装
    - 現在時刻・曜日による「営業中」フィルタリングの実装
    - 同じGNNクラスタIDを持つ「関連エリア」POIの抽出ロジック追加

## フェーズ3: 景観クラスタ連動型ルート生成の実装
- [ ] `src/urban_analysis/prototype/router.py` の修正
    - 起点POIの景観コンテキスト（StreetCLIPクラスタ）の特定
    - 特定された景観クラスタを優先するNKDEコスト関数の調整
    - 複数POIを巡るTSP（巡回セールスマン問題）ソルバーとの整合性確認

## フェーズ4: プロトタイプ全体の結合と検証
- [ ] `src/urban_analysis/prototype/run_prototype.py` の更新
    - ユーザー手順（起点POI選択 → 推薦リスト提示 → 複数選択 → ルート生成）のフロー実装
- [ ] 函館駅前エリアおよび元町エリアでの動作検証
- [ ] 結果の地図出力（HTML）と精度評価
