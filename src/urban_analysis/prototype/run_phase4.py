import folium
import os
import sys
import pandas as pd
from datetime import datetime

# プロジェクトルートパス
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(BASE_DIR)

from src.urban_analysis.prototype.phase4_recommender import Phase4Recommender
from src.urban_analysis.prototype.router import RouteGenerator
from pathlib import Path

# 出力設定
OUTPUT_DIR = Path(os.path.abspath(os.path.join(BASE_DIR, 'docs', 'results')))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(target_poi_name="函館朝市", start_time_str="10:00"):
    print("=== 函館観光ルート推薦 Phase 4 (多様性＆景観ベースアプローチ) ===")
    print("データを準備中...")
    
    # 時間のパース (HH:MM -> minutes)
    try:
        parts = start_time_str.split(':')
        start_time_min = int(parts[0]) * 60 + int(parts[1])
    except:
        print("開始時刻のパースに失敗しました。10:00をデフォルトとします。")
        start_time_min = 600
    
    # 1. データの読み込み
    recommender = Phase4Recommender()
    router = RouteGenerator()
    
    # 2. 起点POIの選択
    print(f"\n[1] 起点POIを選択しました: {target_poi_name} (出発時刻: {start_time_str})")
    
    try:
        # 3. 関連エリア・施設の特定（同景観クラスタで多様性を確保）
        print("\n[2] 同景観クラスタのPOIを抽出し、機能の多様性を確保中...")
        recommended_df, target_poi, candidates_df = recommender.recommend(target_poi_name, top_n=10)
        
        if recommended_df.empty:
            print("条件に合う推薦地点が見つかりませんでした。終了します。")
            return
            
        print(f"   該当POI一覧: {', '.join(candidates_df['name'].tolist())}")

        def format_time(mins):
            h = int(mins) // 60
            m = int(mins) % 60
            return f"{h:02d}:{m:02d}"

        print(f"\n--- 推薦された候補POI (全 {len(candidates_df)} 件中、近傍 Top {len(recommended_df)}件) ---")
        for idx, row in recommended_df.iterrows():
            op_str = format_time(row.get('open_time', 0))
            cl_str = format_time(row.get('close_time', 1440))
            print(f"- {row['name']} (距離: {row['distance_m']:.0f}m) [営業時間: {op_str}〜{cl_str}]")
            
        # 4. 景観重視ルートの生成 (Network KDE)
        # router.generate_route は target_poi を辞書やSeriesとして受け取る
        # street_df には ls_valid_df (lat, lng, clusterを持つ) を渡す
        print("\n[3] 景観特性と営業時間を考慮したルートを探索中...")
        street_df = recommender.ls_valid_df
        route_coords, visit_order, schedules = router.generate_route(
            target_poi, recommended_df, street_df, start_time_min=start_time_min
        )
        
        # スケジュール結果の出力
        print("\n--- 推薦ルート スケジュール ---")

        total_time = 0
        visited_count = len(visit_order) - 1 # 最後の帰還を除く
        visited_names = []
            
        for i, idx in enumerate(visit_order):
            if idx == 0:
                name = target_poi['name']
            else:
                name = recommended_df.iloc[idx-1]['name']
                visited_names.append(name)
                
            sched = schedules[i]
            arr = format_time(sched['arrival'])
            dep = format_time(sched.get('departure', sched['arrival']))
            wait = sched.get('wait', 0)
            wait_str = f" (待機: {wait:.0f}分)" if wait > 0 else ""
            print(f"{i+1:2d}. [{arr} -> {dep}] {name}{wait_str}")
            
        if visited_count < len(recommended_df) + 1:
            candidate_names = recommended_df['name'].tolist()
            skipped_names = [n for n in candidate_names if n not in visited_names]
            print(f"\n※ 時間制約（長時間の待機が必要、または閉店済み）により、以下の {len(skipped_names)} 件の施設はスキップされました:")
            for sn in skipped_names:
                print(f"  - {sn}")
        
        # 5. 可視化
        print("\n[4] マップを生成中...")
        m = folium.Map(location=[target_poi['lat'], target_poi['lng']], zoom_start=15)
        
        # 起点のプロット
        folium.Marker(
            [target_poi['lat'], target_poi['lng']], 
            popup=f"START: {target_poi['name']}", 
            icon=folium.Icon(color='red', icon='play')
        ).add_to(m)
        
        # 推薦POIのプロット
        for i, idx in enumerate(visit_order[1:-1]): # 起点と最後の帰還を除外
            row = recommended_df.iloc[idx-1]
            sched = schedules[i+1] # 起点が0番目なのでi+1
            arr = format_time(sched['arrival'])
            dep = format_time(sched['departure'])
            popup_html = f"<b>STOP {i+1}: {row['name']}</b><br>到着: {arr}<br>出発: {dep}"
            if sched.get('wait', 0) > 0:
                popup_html += f"<br>開店待ち: {sched['wait']:.0f}分"
                
            folium.Marker(
                [row['lat'], row['lng']], 
                popup=folium.Popup(popup_html, max_width=300), 
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
            
        # ルートの描画
        if route_coords:
            folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)
            
        # 既存のPhase4結果ファイルを確認して連番を付与
        counter = 1
        output_path = OUTPUT_DIR / f"phase4_route_map_{target_poi_name}_{counter}.html"
        while output_path.exists():
            counter += 1
            output_path = OUTPUT_DIR / f"phase4_route_map_{target_poi_name}_{counter}.html"
            
        m.save(output_path)
        print(f"\n=== 完了 ===")
        print(f"生成されたルートマップ: {output_path}")

    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="観光ルート推薦 (Phase 4)")
    parser.add_argument("target_poi", nargs="?", default="函館朝市", help="起点となるPOI名")
    parser.add_argument("--time", default="10:00", help="出発時刻 (HH:MM)")
    args = parser.parse_args()
        
    main(args.target_poi, start_time_str=args.time)
