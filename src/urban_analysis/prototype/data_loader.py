import pandas as pd
import numpy as np
import json
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path
from urban_analysis.config import PROJECT_ROOT, PROCESSED_DATA_DIR, RAW_DATA_DIR, OSM_XML_PATH
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

def extract_temporal_features(poi):
    """(既知のロジック: prepare_unified_features.pyと同一)"""
    hours_vec = np.zeros(24)
    days_vec = np.zeros(7)
    oh = poi.get('google_places_data', {}).get('details', {}).get('opening_hours', {})
    periods = oh.get('periods', [])
    if not periods:
        cats = poi.get('categories', [])
        if any(c in str(cats) for c in ['居酒屋', 'バー', '夜']):
            hours_vec[18:24], hours_vec[0:2] = 1, 1
        elif any(c in str(cats) for c in ['朝市', '市場']):
            hours_vec[5:12] = 1
        else: hours_vec[9:18] = 1
        days_vec[:] = 1
        return np.concatenate([hours_vec, days_vec])
    for p in periods:
        if 'open' in p:
            d = p['open'].get('day')
            if d is not None: days_vec[d % 7] = 1
            if 'time' in p['open'] and 'close' in p.get('close', {}):
                try:
                    start_h = int(p['open']['time'][:2])
                    end_h = int(p['close']['time'][:2])
                    if end_h < start_h:
                        hours_vec[start_h:24] = 1
                        hours_vec[0:end_h] = 1
                    else: hours_vec[start_h:end_h] = 1
                except: pass
    return np.concatenate([hours_vec, days_vec])

def load_poi_data():
    """filtered_facilities.jsonをロードする"""
    poi_json_path = PROCESSED_DATA_DIR / 'poi' / 'filtered_facilities.json'
    with open(poi_json_path, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    df_list = []
    for poi in pois:
        loc = poi.get("geometry", {}).get("location", {})
        if not loc:
            loc = poi.get("google_places_data", {}).get("details", {}).get("geometry", {}).get("location", {})
        
        df_list.append({
            "name": poi["name"],
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
            "temp_feat": extract_temporal_features(poi)
        })
    return pd.DataFrame(df_list)

def load_landscape_data(n_clusters=20):
    """StreetCLIPの埋め込みデータをロードし、クラスタリングを行う"""
    emb_path = PROJECT_ROOT / 'data' / 'new' / 'streetclip_embeddings' / 'streetclip_embeddings.npy'
    meta_path = PROJECT_ROOT / 'data' / 'new' / 'streetclip_embeddings' / 'streetclip_metadata.csv'
    pano_meta_path = RAW_DATA_DIR / 'street_view_images_50m_optimized' / 'pano_metadata.json'

    embeddings = np.load(emb_path)
    meta_df = pd.read_csv(meta_path)
    
    with open(pano_meta_path, 'r') as f:
        pano_meta = json.load(f)
    pano_coords = {p["pano_id"]: (p["original_lat"], p["original_lon"]) for p in pano_meta}

    meta_df['index'] = meta_df.index
    unique_points = meta_df.groupby('point_id')['index'].apply(list).to_dict()
    
    point_data = []
    for pid, indices in unique_points.items():
        if pid not in pano_coords: continue
        mean_feat = np.mean(embeddings[indices], axis=0)
        lat, lng = pano_coords[pid]
        point_data.append({
            "point_id": pid,
            "lat": lat,
            "lng": lng,
            "feature": mean_feat
        })
    
    embedding_df = pd.DataFrame(point_data)
    features = np.stack(embedding_df['feature'].values)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    linked = linkage(features_scaled, method='ward')
    clusters = fcluster(linked, n_clusters, criterion='maxclust')
    embedding_df['cluster'] = clusters - 1
    return embedding_df

def get_merged_poi_data():
    """実験18.3（景観保存型）で生成した統合埋め込みを読み込む"""
    UNIFIED_CSV = PROJECT_ROOT / 'data' / 'processed' / 'gnn_unified_residual' / 'residual_embeddings_clustered.csv'
    
    if not UNIFIED_CSV.exists():
        raise FileNotFoundError(f"統合埋め込みデータが見つかりません: {UNIFIED_CSV}")
        
    df = pd.read_csv(UNIFIED_CSV)
    dim_cols = [c for c in df.columns if c.startswith('dim_')]
    df['gnn_embedding'] = df[dim_cols].values.tolist()
    df.rename(columns={'cluster_id': 'cluster'}, inplace=True)
    
    poi_df = df[df['type'] == 'poi'].copy()
    street_df = df[df['type'] == 'sv'].copy()
    
    # 時間情報の再付与
    with open(PROJECT_ROOT / 'data' / 'processed' / 'poi' / 'filtered_facilities.json', 'r') as f:
        pois = json.load(f)
    
    # 住所からターゲット地域に属するPOIのみを特定し、その順序でtemp_featを生成
    # ※ prepare_filtered_unified_features.py のロジックと同期させる必要がある
    TARGET_TOWNS = [
        "末広町", "若松町", "東雲町", "新川町", "千歳町", "海岸町", "松川町", "上新川町", 
        "大森町", "松風町", "旭町", "栄町", "宝来町", "元町", "谷地頭町", "青柳町", 
        "住吉町", "弥生町", "大町", "弁天町", "入舟町", "豊川町", "大手町"
    ]
    
    poi_temp_map = {}
    for poi in pois:
        addr = poi.get('address', '')
        if any(town in addr for town in TARGET_TOWNS):
            poi_temp_map[poi["name"]] = extract_temporal_features(poi)
            
    poi_df['temp_feat'] = poi_df['name'].map(poi_temp_map)
    
    return poi_df, street_df
