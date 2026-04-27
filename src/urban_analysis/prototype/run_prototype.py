import folium
import os
import sys
import pandas as pd
from datetime import datetime

# プロジェクトルートパス
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(BASE_DIR)

from src.urban_analysis.prototype.data_loader import get_merged_poi_data
from src.urban_analysis.prototype.recommender import RouteRecommender
from src.urban_analysis.prototype.router import RouteGenerator

# 出力設定
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, 'docs', 'results'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(target_poi_name="函館朝市"):
    print("=== 函館観光ルート推薦プロトタイプ (GNN + StreetCLIP 統合版) ===")
    
    # 1. データの読み込み
    print("データを準備中...")
    poi_df, street_df = get_merged_poi_data()
    recommender = RouteRecommender(poi_df=poi_df)
    router = RouteGenerator()
    
    # 2. 起点POIの選択
    print(f"\n[1] 起点POIを選択しました: {target_poi_name}")
    
    try:
        # 現在時刻での推薦 (デモ用に9:00に設定)
        sim_time = datetime.now().replace(hour=9, minute=0)
        print(f"シミュレート時刻: {sim_time.strftime('%Y-%m-%d %H:%M')}")
        
        # 3. 関連エリア・施設の特定（GNN時空間埋め込み + 営業中フィルタ）
        print("\n[2] 関連の深いエリアと近隣POIを特定中...")
        recommended_df = recommender.recommend(target_poi_name, target_time=sim_time, top_n=5)
        
        if recommended_df.empty:
            print("条件に合う推薦地点が見つかりませんでした。")
            return

        print(f"--- 推薦されたPOI (Top {len(recommended_df)}) ---")
        for idx, row in recommended_df.iterrows():
            print(f"- {row['name']} (GNNクラスタ: {row['cluster']}, 距離: {row['distance_m']:.0f}m)")
            
        # 起点POI情報の取得
        target_poi = poi_df[poi_df['name'].str.contains(target_poi_name)].iloc[0]
        
        # 4. 景観重視ルートの生成 (StreetCLIPクラスタ連動)
        print("\n[3] 景観特性を考慮したルートを探索中...")
        route_coords, visit_order = router.generate_route(target_poi, recommended_df, street_df)
        
        # 5. 可視化
        print("\n[4] マップを生成中...")
        m = folium.Map(location=[target_poi['lat'], target_poi['lng']], zoom_start=15)
        
        # POIのプロット
        # 起点
        folium.Marker(
            [target_poi['lat'], target_poi['lng']], 
            popup=f"START: {target_poi['name']}", 
            icon=folium.Icon(color='red', icon='play')
        ).add_to(m)
        
        # 推薦POI
        for i, idx in enumerate(visit_order[1:]): # 起点以外
            row = recommended_df.iloc[idx-1]
            folium.Marker(
                [row['lat'], row['lng']], 
                popup=f"STOP {i+1}: {row['name']}", 
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
            
        # ルートの描画
        if route_coords:
            folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)
            
        output_path = os.path.join(OUTPUT_DIR, f"prototype_route_map_{target_poi_name}.html")
        m.save(output_path)
        print(f"\n=== 完了 ===")
        print(f"生成されたルートマップ: {output_path}")

    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
    else:
        target_name = "函館朝市" 
        
    main(target_name)
