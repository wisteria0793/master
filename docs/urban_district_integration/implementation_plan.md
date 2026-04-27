# 景観クラスタとPOIクラスタの統合分析 実装計画 (指定データ活用版)

## 概要
指定された最新の特徴量（StreetCLIP）およびPOI紹介文ベクトルを用い、既存の地区抽出結果をベースとした統合分析を行います。

## 使用データ
1. **景観データ (Landscape)**:
    - 特徴量: `data/new/streetclip_embeddings/streetclip_embeddings.npy`
    - メタデータ: `data/new/streetclip_embeddings/streetclip_metadata.csv`
    - 既存クラスタ: `data/processed/approach_b/sv_landscape_only_clusters.csv`
2. **POIデータ (Function)**:
    - 元データ: `data/processed/poi/filtered_facilities.json`
    - 特徴量: 紹介文をベクトル化したもの（対応する `.npy` または既存の `poi_text_clusters.csv` を活用）
    - 既存クラスタ: `data/processed/approach_b/poi_text_only_clusters.csv`

## 統合アプローチ
「すでに行われた地区抽出」を正解ラベルとし、それらを空間的に統合します。

### 1. 最新景観特徴量と地区ラベルの同期
- `data/new/` のStreetCLIP特徴量に対し、`approach_b/` の既存クラスターIDをマッピングします。
- 必要に応じて、最新のポイント群に対して最近傍補間によりクラスターIDを全地点に付与します。

### 2. POI地点への景観コンテキストの集約
- 各POI（374地点）に対し、その周辺にある景観地点のクラスター構成を算出します。
- 例：「このレストランの周囲200mは『歴史的景観クラスター(ID-3)』が80%を占める」といった属性をPOIに付与。

### 3. 多次元地区プロファイリング
- **機能（Text）× 外見（Landscape）のクロス集計**:
    - 特定のPOIクラスター（例：レトロカフェ）がどのような景観クラスターに出現しやすいかを定量化。
- **乖離（Dissonance）分析**:
    - 機能と景観が一般的な組み合わせではない地点を「都市の意外なスポット」として抽出。

### 4. 統合可視化
- 指定されたPOI（filtered_facilities）を地図上にプロット。
- 景観クラスターに応じた「背景色」と、POIクラスターに応じた「アイコン」を組み合わせた統合マップを作成。

## 実施タスク
- [ ] 1. StreetCLIP最新データへの既存クラスターIDのマッピング
- [ ] 2. POI地点への空間ジョイン（近傍景観属性の付与）
- [ ] 3. 景観×機能の相関マトリクスの作成
- [ ] 4. 統合可視化マップ（Folium）の生成
