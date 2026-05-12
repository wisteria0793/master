# run_phase5.py
"""Execute Phase 5 multi‑objective tourist route recommendation.
It extends the Phase 4 runner by invoking the Phase5Recommender, which
optimizes distance, Google rating and review count via NSGA‑III.
"""

import folium
import os
import sys
from pathlib import Path

# Project root resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(BASE_DIR)

from src.urban_analysis.prototype.phase5_recommender import Phase5Recommender
from src.urban_analysis.prototype.router import RouteGenerator

# Output directory for maps and logs
OUTPUT_DIR = Path(os.path.abspath(os.path.join(BASE_DIR, 'docs', 'results')))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_time(mins) -> str:
    mins = int(mins)
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def main(target_poi_name: str = "函館朝市", start_time_str: str = "10:00", top_n: int = 10, config_path: str = None):
    print("=== 函館観光ルート推薦 Phase 5 (多目的最適化) ===")
    print("データを準備中…")

    # Parse start time
    try:
        h, m = map(int, start_time_str.split(':'))
        start_time_min = h * 60 + m
    except Exception:
        print("開始時刻のパースに失敗しました。10:00 をデフォルトとします。")
        start_time_min = 600

    # Initialise recommender and router
    recommender = Phase5Recommender(start_poi_name=target_poi_name, config_path=config_path)
    router = RouteGenerator()

    # Get recommendations
    try:
        result = recommender.recommend(top_n=top_n)
        selected_pois = result["selected_pois"]
        # Convert list of dicts to DataFrame for compatibility with router
        import pandas as pd
        recommended_df = pd.DataFrame(selected_pois)
        # The original target POI info is stored in recommender.target_poi (in Phase5) – reuse it
        target_poi = recommender.target_poi
        # Candidates may be useful for logging; retrieve from recommender.candidates
        candidates_df = recommender.candidates
    except Exception as e:
        print(f"推薦取得中にエラーが発生しました: {e}")
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)

    if recommended_df.empty:
        print("条件に合う推薦地点が見つかりませんでした。終了します。")
        return

    print(f"\n--- 推薦された POI ({len(recommended_df)} 件) ---")
    for idx, row in recommended_df.iterrows():
        # distance may be missing in Phase5 output; compute if present
        dist = row.get('distance_m', None)
        dist_str = f" (距離: {dist:.0f}m)" if isinstance(dist, (int, float)) else ""
        print(f"- {row.get('name', row.get('poi_id'))}{dist_str}")

    # Generate route using the same router implementation as Phase4
    print("\n[3] 景観特性と営業時間を考慮したルートを探索中…")
    street_df = recommender.ls_valid_df  # inherited from Phase4
    try:
        route_coords, visit_order, schedules = router.generate_route(
            target_poi, recommended_df, street_df, start_time_min=start_time_min
        )
    except Exception as e:
        print(f"ルート生成中にエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Output schedule
    print("\n--- 推薦ルート スケジュール ---")
    visited_names = []
    for i, idx in enumerate(visit_order):
        name = target_poi['name'] if idx == 0 else recommended_df.iloc[idx - 1]['name']
        if idx != 0:
            visited_names.append(name)
        sched = schedules[i]
        arr = format_time(sched['arrival'])
        dep = format_time(sched.get('departure', sched['arrival']))
        wait = sched.get('wait', 0)
        wait_str = f" (待機: {int(wait)}分)" if wait > 0 else ""
        print(f"{i+1:2d}. [{arr} -> {dep}] {name}{wait_str}")

    # Report skipped POIs if any (same logic as Phase4)
    if len(visited_names) < len(recommended_df):
        skipped = [n for n in recommended_df['name'].tolist() if n not in visited_names]
        print(f"\n※ 時間制約により以下の {len(skipped)} 件がスキップされました:")
        for sn in skipped:
            print(f"  - {sn}")

    # Visualisation with Folium
    print("\n[4] マップを生成中…")
    m = folium.Map(location=[target_poi['lat'], target_poi['lng']], zoom_start=15)
    # 起点
    folium.Marker(
        [target_poi['lat'], target_poi['lng']],
        popup=f"START: {target_poi['name']}",
        icon=folium.Icon(color='red', icon='play')
    ).add_to(m)
    # POI markers
    for i, idx in enumerate(visit_order[1:-1]):  # exclude start and final return
        row = recommended_df.iloc[idx - 1]
        sched = schedules[i + 1]
        popup_html = f"<b>STOP {i+1}: {row.get('name', row.get('poi_id'))}</b><br>到着: {format_time(sched['arrival'])}<br>出発: {format_time(sched['departure'])}"
        if sched.get('wait', 0) > 0:
            popup_html += f"<br>開店待ち: {int(sched['wait'])}分"
        folium.Marker(
            [row['lat'], row['lng']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    # Route line
    if route_coords:
        folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)

    # Save map with incremental filename
    counter = 1
    safe_name = target_poi_name.replace(' ', '_')
    output_path = OUTPUT_DIR / f"phase5_route_map_{safe_name}_{counter}.html"
    while output_path.exists():
        counter += 1
        output_path = OUTPUT_DIR / f"phase5_route_map_{safe_name}_{counter}.html"
    m.save(output_path)
    print("\n=== 完了 ===")
    print(f"生成されたルートマップ: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase5 多目的観光ルート推薦")
    parser.add_argument("target_poi", nargs="?", default="函館朝市", help="起点となる POI 名")
    parser.add_argument("--time", default="10:00", help="出発時刻 (HH:MM)")
    parser.add_argument("--top_n", type=int, default=10, help="1 回の候補解で選択する POI 数")
    parser.add_argument("--config", default=None, help="Phase5 設定ファイルへのパス (yaml)")
    args = parser.parse_args()
    main(args.target_poi, start_time_str=args.time, top_n=args.top_n, config_path=args.config)
