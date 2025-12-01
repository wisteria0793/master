import os
import json
import overpy
import time
from dotenv import load_dotenv

load_dotenv('.env')

# --- Constants and Configuration ---
# Using the same bounding box as get_flickr.py for consistency
HAKODATE_BBOX_STR = '41.7,140.5,41.9,141.0' # S,W,N,E format for Overpass API
DOWNLOAD_DIR = './data/raw/osm_hakodate'
RATE_LIMIT_DELAY = 1.05 # Delay between API calls if needed

# --- Folder Preparation ---
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    print(f"Created directory: {DOWNLOAD_DIR}")

print(f"--- Starting OpenStreetMap data collection for Hakodate (bbox: {HAKODATE_BBOX_STR}) ---")

def fetch_osm_data(bbox_str):
    """
    Fetches OpenStreetMap data (nodes, ways, relations) within a given bounding box
    using Overpass API via overpy.
    """
    api = overpy.Overpass()
    
    # Overpass query to get all nodes, ways, and relations within the bounding box
    # This is a very broad query; it can be refined later for specific POIs.
    query = f"""
    (
      node["amenity"~"^(restaurant|cafe|museum|attraction|park|library)$"]({bbox_str});
      way["amenity"~"^(restaurant|cafe|museum|attraction|park|library)$"]({bbox_str});
      relation["amenity"~"^(restaurant|cafe|museum|attraction|park|library)$"]({bbox_str});
      
      node["tourism"~"^(information)$"]({bbox_str});
      way["tourism"~"^(information)$"]({bbox_str});
      relation["tourism"~"^(information)$"]({bbox_str});
      
      node["shop"~"^(supermarket)$"]({bbox_str});
      way["shop"~"^(supermarket)$"]({bbox_str});
      relation["shop"~"^(supermarket)$"]({bbox_str});

      node["public_transport"~"^(station)$"]({bbox_str});
      way["public_transport"~"^(station)$"]({bbox_str});
      relation["public_transport"~"^(station)$"]({bbox_str});

      way["highway"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    
    print(f"Executing Overpass query for bbox: {bbox_str}")
    
    try:
        result = api.query(query)
        print(f"Successfully fetched OSM data. Nodes: {len(result.nodes)}, Ways: {len(result.ways)}, Relations: {len(result.relations)}")
        return result
    except Exception as e:
        print(f"Error fetching OSM data: {e}")
        return None

def save_osm_data_as_json(osm_result, filename="osm_data.json"):
    """
    Saves the fetched overpy result object into a JSON file.
    Note: overpy result object is not directly JSON serializable.
    We need to extract relevant data.
    """
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    
    data_to_save = {
        "nodes": [],
        "ways": [],
        "relations": []
    }

    for node in osm_result.nodes:
        data_to_save["nodes"].append({
            "id": node.id,
            "lat": float(node.lat),
            "lon": float(node.lon),
            "tags": node.tags
        })
    
    for way in osm_result.ways:
        data_to_save["ways"].append({
            "id": way.id,
            "nodes": [n.id for n in way.nodes],
            "tags": way.tags
        })

        for relation in osm_result.relations:
            members = []
            for member in relation.members:
                # Infer type from class name if direct 'type' attribute is missing or causing issues
                member_type = member.__class__.__name__.replace('Relation', '').lower()
                if member_type in ['node', 'way', 'relation']: # Ensure it's a valid OSM element type
                    members.append({
                        "type": member_type,
                        "ref": member.ref,
                        "role": member.role,
                    })
                else:
                    print(f"Warning: Unknown relation member type encountered: {member.__class__.__name__}")
            data_to_save["relations"].append({
                "id": relation.id,
                "members": members,
                "tags": relation.tags
            })

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved OSM data to {output_path}")
    except Exception as e:
        print(f"Error saving OSM data to JSON: {e}")

if __name__ == "__main__":
    osm_data = fetch_osm_data(HAKODATE_BBOX_STR)
    if osm_data:
        save_osm_data_as_json(osm_data, "hakodate_osm_raw.json")
    print("--- OpenStreetMap data collection finished ---")
