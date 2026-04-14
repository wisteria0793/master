import os
import sys
import folium

# プロジェクトルートパス
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(BASE_DIR)

from src.urban_analysis.prototype.recommender import RouteRecommender
from src.urban_analysis.prototype.router import RouteGenerator
from src.urban_analysis.prototype.data_loader import load_and_cluster_embeddings

OUTPUT_MAP_DIR = os.path.join(BASE_DIR, 'docs', 'prototype_route_system')

def main(target_poi_name):
    print("====================================")
    print(f"観光ルート推薦プロトタイプ実行開始")
    print(f"基準POI: {target_poi_name}")
    print("====================================\n")
    
    # 1. 推薦エンジンの初期化と実行
    print("1. データ読み込み中...")
    recommender = RouteRecommender()
    
    print("2. 推薦施設を抽出中...")
    # プロトタイプの推薦ロジック（同じ景観クラスタ内、距離制約内、コサイン類似度TopN）
    recommended_df = recommender.recommend(target_poi_name)
    
    if recommended_df.empty:
        print("条件に合致する推薦POIが見つかりませんでした。別のPOIをお試しください。")
        return

    # 基準POI情報の取得
    target_rows = recommender.poi_df[recommender.poi_df['name'].str.contains(target_poi_name, na=False)]
    target_poi = target_rows.iloc[0]
    
    print("\n【推薦結果】")
    for idx, row in recommended_df.iterrows():
        print(f" - {row['name']} (距離: {row['distance_m']:.0f}m, 類似度: {row['similarity_score']:.4f}, クラスタ: {row['cluster']})")
    
    # 2. ルート生成の実行
    print("\n3. 道路ネットワークグラフを取得＆ルート探索中...")
    all_cluster_points_df = load_and_cluster_embeddings()
    router = RouteGenerator()
    result = router.generate_route(target_poi, recommended_df, all_cluster_points_df)
    
    if not result or not result[0]:
        print("経路の生成に失敗しました。")
        return
        
    route_geom, best_order = result
        
    # 3. Foliumによる可視化
    print("\n4. マップ（Folium）を生成中...")
    start_lat = target_poi['lat']
    start_lng = target_poi['lng']
    
    # 地図の初期化 (起点となるPOIを中心に)
    # 起点から推薦ポイント全てが収まるように調整
    m = folium.Map(location=[start_lat, start_lng], zoom_start=15)
    
    # 起点 (ターゲット) のプロット (赤・星)
    folium.Marker(
        location=[start_lat, start_lng],
        popup=f"【起点】\n{target_poi['name']}\nクラスタ: {target_poi['cluster']}",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m)
    
    # TSPの訪問順序をマッピング
    # best_order は [0, 3, 1, 2, 0] のようなリスト (0は起点、1〜Nは推薦POIのインデックス+1)
    visit_order_map = {}
    visit_num = 1
    for node_idx in best_order:
        if node_idx == 0:
            continue
        df_idx = node_idx - 1  # recommended_df の 0-based インデックス
        if df_idx not in visit_order_map:
            visit_order_map[df_idx] = visit_num
            visit_num += 1
    
    # 推薦POIのプロット (青・情報アイコン)
    for i in range(len(recommended_df)):
        row = recommended_df.iloc[i]
        order = visit_order_map.get(i, "?")
        popup_text = f"【目的地 {order}】\n{row['name']}\n距離: {row['distance_m']:.0f}m\n類似度: {row['similarity_score']:.4f}"
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=popup_text,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
    # 経路ポリラインの描画 (太めのオレンジ)
    folium.PolyLine(
        route_geom,
        color='darkorange',
        weight=6,
        opacity=0.8,
        tooltip="推薦ルート"
    ).add_to(m)
    
    # 保存ディレクトリの確保
    os.makedirs(OUTPUT_MAP_DIR, exist_ok=True)
    
    # ファイル名に対象POI名を付与 (スラッシュやスペースはアンダースコアに置換)
    safe_poi_name = target_poi['name'].replace('/', '_').replace(' ', '_')
    output_path = os.path.join(OUTPUT_MAP_DIR, f'recommended_route_{safe_poi_name}.html')
    
    m.save(output_path)
    
    print(f"\n完了！ルートマップを保存しました: {output_path}")

if __name__ == "__main__":
    # 対象POIをコマンドライン引数で受け取るか、デフォルト値を設定
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
    else:
        # 函館を代表する観光地をデフォルトで設定
        target_name = "函館山" 
        
    try:
        main(target_name)
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
