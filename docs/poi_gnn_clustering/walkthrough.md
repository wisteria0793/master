# POI GNNクラスタリング・地図プロット 修正内容の確認

GNN 統合埋め込みを用いた POI のクラスタリング分析と、地図上への可視化が完了しました。

## 実施内容

### 1. 最適なクラスタ数 ($k$) の分析
[analyze_poi_gnn_clusters.py](file:///Users/atsuyakatougi/Desktop/master/src/urban_analysis/gnn/analyze_poi_gnn_clusters.py) を実行し、クラスタ数 $k=5$ から $40$ の範囲でシルエットスコアを算出しました。
- **結果**: $k=9$ が最大スコア (0.6075) を記録し、最も自然なクラスタ構造であると判断されました。
- **グラフ**: [poi_gnn_silhouette_scores.png](file:///Users/atsuyakatougi/Desktop/master/docs/results/poi_gnn_silhouette_scores.png)

### 2. クラスタリングの実行
特定された $k=9$ を用い、Ward法によって POI を 9 つのグループに分類しました。
- テキストの意味的類似性と地理的距離が GNN によって既に統合されているため、空間的なまとまりと施設種別のまとまりの両方を反映したクラスタが得られています。
- **CSV結果**: [poi_gnn_clusters.csv](file:///Users/atsuyakatougi/Desktop/master/data/processed/gnn_embeddings/poi_gnn_clusters.csv)

### 3. 地図上へのプロット
Folium を使用して、クラスタごとに色分けされたインタラクティブマップを生成しました。
- 各 POI をクリックすることで、施設名、クラスタ番号、カテゴリを確認できます。
- **HTMLマップ**: [poi_gnn_cluster_map.html](file:///Users/atsuyakatougi/Desktop/master/docs/results/poi_gnn_cluster_map.html)

## 完了したタスク
- [x] クラスタ分析・可視化スクリプト (`analyze_poi_gnn_clusters.py`) の作成
- [x] 最適クラスタ数 ($k$) の算出 (シルエットスコアによる最適化)
- [x] クラスタリングの実行と結果の保存
- [x] Foliumによる地図プロット (HTMLマップ) の作成
- [x] 結果の確認と可視化プロットの検証
