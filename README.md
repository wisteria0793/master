# 修士論文の研究取り組み

## 研究テーマ
**地区特性を考慮した観光ルート推薦システムの構築**

本研究は、機械学習モデルを用いて、観光地特有の雰囲気や情緒的な特性を定量化し、それを活用した新しい観光ルート推薦システムの構築を目指す

---


## 過去の取り組み

これまでの実験（実験1〜18.11）については、以下のアーカイブを参照してください。
- [過去の実験アーカイブ](docs/history/experiments_archive.md)

---

## 現在の取り組み (プロトタイプ開発)

### 都市地区統合分析：景観と機能によるマルチモーダル都市記述
都市の「外見（StreetView画像から得られる景観特性）」と「中身（POIの紹介文から得られる機能特性）」を統合し、実社会における都市の性格（District Identity）を多角的に定量化するプロトタイプの開発を進めています。

この分析により、従来の機能分類だけでは捉えきれなかった「歴史的な佇まいを持つビジネス街」や「生活感のある観光スポット」といった、都市の重層的な特性を抽出することを目指します。

#### フェーズ1：空間相関に基づくシンプル統合（完了・分析済み）
- **アルゴリズム詳細 (Detailed Algorithm)**:
    1. **局所座標への射影変換 (Coordinate Projection)**: 
       緯度経度 $(lat, lng)$ を正確な距離計算のためにメートル単位の局所座標 $(x, y)$ へ近似変換。函館周辺の特性を考慮し、緯度1度 $\approx 111,000m$、経度1度 $\approx 82,000m$ として線形射影。
    2. **近傍景観の集約 (Spatial Context Aggregation)**:
       各POI $P_i$ に対し、半径 $R=150m$ 以内の全景観地点 $S_j$ を $KDTree$ で探索。周辺景観セット $\{C_{ls, j} \mid dist(P_i, S_j) < R\}$ から、**最頻値 (Mode)** を算出。これを当該POIの空間景観コンテキスト $L_{context, i}$ と定義。
    3. **統計的「典型的ペア」の定義 (Co-occurrence Typicality)**:
       全POIサンプル $(C_{text}, L_{context})$ の共生行列を算出、各機能クラスター $k$ に対して以下を典型的景観 $T(k)$ として定義：
       $$T(k) = \text{argmax}_c \sum_{i} \mathbb{1}(C_{text, i} = k \land L_{context, i} = c)$$
    4. **都市の乖離判定 (Urban Dissonance Detection)**:
       各POI地点 $i$ において、実測された周辺景観 $L_{context, i}$ と統計的期待値 $T(C_{text, i})$ を比較し、不一致の場合を「乖離（Dissonance）」として抽出：
       $$Dissonance_i = \begin{cases} 1 & \text{if } L_{context, i} \neq T(C_{text, i}) \\ 0 & \text{otherwise} \end{cases}$$
- **分析結果 (Results)**:
    - **対象地点数**: 408地点（空間紐付け成功分）
    - **クラスタ数**: 景観（20クラスター） / 機能（9クラスター）
    - **統計**: 典型的スポット 392地点 / **意外なスポット 16地点 (全体の約4%)**
- **主要な成果物 (Outputs)**:
    - **統合データ**: `data/processed/integrated_poi_landscape_clusters.csv`
    - **可視化マップ**: `docs/results/integrated_urban_map.html`
    - **実行スクリプト**: 
        - `src/urban_analysis/prototype/simple_spatial_integration.py`
        - `src/urban_analysis/prototype/visualize_integrated_map.py`

#### フェーズ2：クラスターラベルに基づくGNN空間統合（次ステップ）
フェーズ1の単純な紐付けから進み、クラスター化された抽象情報をGraph Neural Networks (GNN) に統合することで、空間的な一貫性と文脈を考慮した都市記述を生成します。

- **1. データクレンジングと前処理の紆余曲折 (Data Denoising Process)**:
    - **初期段階**: 全POI（793地点）を対象にフェーズ1を実施。
    - **ブラッシュアップ1**: 施設紹介文が欠損している34地点を除去（759地点）。紹介文を再結合・再ベクトル化し、意味的な純度を向上。
    - **ブラッシュアップ2（重要）**: 「景観情報がないPOIが母集団に混ざると地区の定義が歪む」という課題に対処。半径150m以内にStreetView点が存在しないPOIを事前に除外。
    - **最終データセット**: **厳選された374地点**（機能的文脈 ＋ 景観的文脈が共に存在する地点）を確定。
- **2. 最適なクラスタ数（K）の客観的算出 (Results)**:
    - 374地点に絞り込んだ最新特徴量に対し、エルボー法・シルエット法を再実行。
    - **景観（Landscape）**: 統計的変曲点である **k=12**。
    - **機能（Function）**: 孤立地点の除去により特性が鮮明化し、**k=8**（以前のk=13から最適化）を特定。
- **3. クラスターラベルの特徴量化とGNN学習**:
    - 20次元（POI:8, LS:12）の純粋な空間的ネットワークとしてGNN（GAE/GAT）を再学習。
- **4. 特徴的な可視化による最終統合**:
    - `visualize_gnn_clusters.py` との互換性を持つ `tab20` カラーパレットを採用。
    - 不透明度100%のソリッドな描画により、景観と機能が融合した12の統合地区を明快に抽出。

- **5. 単体GNNによる比較検証 (Ablation Study / Baseline)**:
    - 統合による効果を測定するため、景観・機能を個別に空間平滑化した比較用マップを生成。
    - **景観単体平滑化**: `docs/results/smoothed_landscape_map_k12.html`
    - **機能単体平滑化**: `docs/results/smoothed_function_map_k8.html`
    - **（参考）統合前の純粋クラスタリング**: `docs/results/individual_*.html`

- **6. 同時統合 vs 段階的統合の比較検証 (Early vs Late Integration)**:
    - **同時統合（本手法）**: 景観と機能を初期段階で結合し、単一のGNNで学習。相互依存的な「地区の文脈」を抽出可能。
    - **段階的統合（比較用）**: 個別のGNNで磨き上げた特徴を後から空間集計して結合。`docs/results/late_integration_urban_districts_map.html`
    - **考察**: 同時統合の方が境界が滑らかで、景観と活動が深く結びついた実態に近い地区形成が行われることを確認。

#### フェーズ3：観光ルート推薦エンジンの実装（進行中）
解析された「統合地区」のデータを活用し、実用的な観光体験を提供するエンジンを構築。

- **1. 地区文脈ベースのPOI推薦 (Recommendation Engine v1) [実装済み]**:
    - **概要**: ユーザーが起点として選んだPOIの「統合地区ID」を特定し、その地区の文脈を最大限に活かした推薦を行う。
    - **推薦ロジック**: 
        - 同一クラスター内のPOIを母集団とし、Sentence-BERTによる意味的類似度でランキング。
        - 地区を構成する主要な景観クラスターを分析し、「この地区は〇〇な雰囲気である」という言語的な推薦理由を自動提示。
    - **実装スクリプト**: [`src/urban_analysis/prototype/district_recommendation_engine.py`](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/prototype/district_recommendation_engine.py)

- **2. 景観重視の経路生成 (Landscape-aware Pathfinding) [開発予定]**:
    - 単なる最短経路ではなく、質の高い景観クラスター（GNNで平滑化された景観エリア）を優先的に通過する「景観スコアリング」を用いたルート生成アルゴリズム。
    - 観光客に対し、移動そのものをコンテンツ化する「歩きたくなる道」の提示。

- **3. インタラクティブ・プロトタイプ構築 [開発予定]**:
    - ユーザーが地点を選択すると、即座に推薦POIと景観優先ルートが地図上に描画されるアプリケーション。

---

詳細は [docs/urban_district_integration/](file:///Users/atsuyakatougi/Desktop/master/docs/urban_district_integration/) 以下のドキュメントを参照してください。

