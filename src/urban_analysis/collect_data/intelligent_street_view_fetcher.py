import os
import json
import requests
import time
import sys
import xml.etree.ElementTree as ET
from geopy.distance import great_circle, Point
from dotenv import load_dotenv

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# CONFIGURATION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
INPUT_OSM_FILE = './data/raw/osm_hakodate/Hakodate.osm.xml'
OUTPUT_DIR = './data/raw/street_view_images_50m_intelligent'
LOG_FILE = './docs/results/intelligent_fetch_log_50m.json'

INTERVAL_METERS = 50  # The target interval for sampling along roads
MICRO_SEARCH_STEPS = [0, 10, -10, 20, -20, 30, -30] # Search steps in meters if the primary point fails
COORDINATE_PRECISION = 6 # Number of decimal places for coordinates

IMAGE_SIZE = "640x640"
FOV = 90
REQUEST_DELAY = 0.05 # Delay between API calls. Be respectful to the API.

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# HELPER FUNCTIONS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def build_node_cache(osm_file):
    """
    Parses the OSM XML file once to build a dictionary of node IDs to their coordinates.
    Uses iterparse for memory efficiency.
    """
    print(f"Building node cache from {osm_file}...")
    nodes = {}
    context = ET.iterparse(osm_file, events=("start",))
    for _, elem in context:
        if elem.tag == 'node':
            try:
                node_id = elem.attrib['id']
                lat = float(elem.attrib['lat'])
                lon = float(elem.attrib['lon'])
                nodes[node_id] = (lat, lon)
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed node. Error: {e}", file=sys.stderr)
        # Clear the element to free memory
        elem.clear()
    print(f"Node cache built with {len(nodes)} nodes.")
    return nodes

def get_way_path_and_length(way_elem, node_cache):
    """Gets the coordinate path for a way and calculates its total length."""
    node_ids = [nd.attrib['ref'] for nd in way_elem.findall('nd')]
    path = [node_cache[nid] for nid in node_ids if nid in node_cache]
    
    length = 0.0
    for i in range(len(path) - 1):
        length += great_circle(path[i], path[i+1]).meters
        
    return path, length

def get_points_on_path(path, interval):
    """Generates candidate points along a coordinate path at a given interval."""
    if not path or len(path) < 2:
        return [path[0]] if path else []

    points = [path[0]]
    dist_to_next_marker = interval
    
    for i in range(len(path) - 1):
        start_node = Point(path[i])
        end_node = Point(path[i+1])
        segment_dist = great_circle(start_node, end_node).meters
        
        while dist_to_next_marker <= segment_dist:
            fraction = dist_to_next_marker / segment_dist
            lat = start_node.latitude + fraction * (end_node.latitude - start_node.latitude)
            lon = start_node.longitude + fraction * (end_node.longitude - start_node.longitude)
            points.append((lat, lon))
            
            segment_dist -= dist_to_next_marker
            dist_to_next_marker = interval
        
        dist_to_next_marker -= segment_dist
        
    return points

def get_point_at_distance(path, distance_meters):
    """Finds a single point at a specific distance along a path."""
    if not path: return None
    
    cumulative_dist = 0
    for i in range(len(path) - 1):
        start_node = Point(path[i])
        end_node = Point(path[i+1])
        segment_dist = great_circle(start_node, end_node).meters
        
        if cumulative_dist + segment_dist >= distance_meters:
            dist_into_segment = distance_meters - cumulative_dist
            fraction = dist_into_segment / segment_dist if segment_dist > 0 else 0
            lat = start_node.latitude + fraction * (end_node.latitude - start_node.latitude)
            lon = start_node.longitude + fraction * (end_node.longitude - start_node.longitude)
            return (lat, lon)
            
        cumulative_dist += segment_dist
    return None

def check_metadata(point, api_key):
    """Uses the Metadata API to check if an image exists at a location."""
    time.sleep(REQUEST_DELAY)
    lat, lon = point
    params = {'location': f"{lat},{lon}", 'key': api_key, 'source': 'outdoor'}
    try:
        response = requests.get("https://maps.googleapis.com/maps/api/streetview/metadata", params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('status') == 'OK'
    except requests.exceptions.RequestException as e:
        print(f"  - Metadata check failed: {e}", file=sys.stderr)
    return False

def fetch_image(point, api_key):
    """Fetches and returns the image content for a valid point."""
    time.sleep(REQUEST_DELAY)
    lat, lon = point
    params = {'size': IMAGE_SIZE, 'location': f"{lat},{lon}", 'fov': FOV, 'key': api_key, 'source': 'outdoor'}
    try:
        response = requests.get("https://maps.googleapis.com/maps/api/streetview", params=params, timeout=15)
        if response.status_code == 200 and len(response.content) > 20000:
             return response.content
    except requests.exceptions.RequestException as e:
        print(f"  - Image fetch failed: {e}", file=sys.stderr)
    return None

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# MAIN EXECUTION
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def main():
    print("--- Starting Intelligent Street View Fetcher ---")
    
    # --- Preparations ---
    load_dotenv('.env')
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: 'GOOGLE_API_KEY' not found in .env file.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    node_cache = build_node_cache(INPUT_OSM_FILE)
    
    log_data = []
    processed_ways = 0
    saved_images = 0

    print("\n--- Starting to process ways from OSM file ---")
    context = ET.iterparse(INPUT_OSM_FILE, events=("start",))
    
    for _, elem in context:
        if elem.tag == 'way':
            is_road = any(tag.attrib.get('k') == 'highway' for tag in elem.findall('tag'))
            if not is_road:
                elem.clear()
                continue
            
            processed_ways += 1
            way_id = elem.attrib['id']
            way_log = {'way_id': way_id, 'status': 'no_image_found', 'saved_points': []}
            
            path, length = get_way_path_and_length(elem, node_cache)
            if not path:
                log_data.append({**way_log, 'status': 'path_not_found'})
                elem.clear()
                continue
            
            # 1. Generate primary candidate points
            candidate_points = get_points_on_path(path, INTERVAL_METERS)
            
            print(f"\nProcessing Way #{way_id} (Length: {length:.2f}m, Candidates: {len(candidate_points)})")

            # 2. For each candidate, find a valid image via micro-search
            for primary_point in candidate_points:
                found_valid_point = None
                
                # 3. Micro-search loop
                for offset in MICRO_SEARCH_STEPS:
                    dist_along_path = great_circle(path[0], primary_point).meters + offset
                    
                    # Ensure the search point is within the bounds of the path
                    if 0 <= dist_along_path <= length:
                        search_point = get_point_at_distance(path, dist_along_path)
                        if search_point:
                            rounded_point = (round(search_point[0], COORDINATE_PRECISION), 
                                             round(search_point[1], COORDINATE_PRECISION))
                            
                            print(f"  - Checking point: {rounded_point} (Offset: {offset}m)")
                            if check_metadata(rounded_point, api_key):
                                found_valid_point = rounded_point
                                break # Found a valid point, stop micro-search
                
                # 4. If a valid point was found, fetch and save the image
                if found_valid_point:
                    print(f"  - SUCCESS: Valid location found at {found_valid_point}. Fetching image...")
                    image_content = fetch_image(found_valid_point, api_key)
                    if image_content:
                        lat, lon = found_valid_point
                        filename = f"way_{way_id}_loc_{lat}_{lon}.jpg"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        with open(filepath, 'wb') as f:
                            f.write(image_content)
                        
                        saved_images += 1
                        way_log['status'] = 'success'
                        way_log['saved_points'].append({'coord': found_valid_point, 'filename': filename})
                        print(f"  - SAVE: Image saved to {filename}")
                    else:
                        print("  - FAILED: Found metadata but failed to fetch image.")
                else:
                    print("  - FAILED: No valid image found for this interval point after micro-search.")
            
            log_data.append(way_log)
            # Clear the element to free memory
            elem.clear()

    # --- Finalization ---
    print(f"\n--- --- --- --- --- --- --- --- ---")
    print(f"Processing complete.")
    print(f"Total ways processed: {processed_ways}")
    print(f"Total images saved: {saved_images}")
    
    print(f"Saving log file to {LOG_FILE}...")
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print("--- Intelligent Street View Fetcher ---")
    print("This script will parse the OSM XML file and attempt to fetch one image")
    print(f"for each {INTERVAL_METERS}m interval on all roads, performing a micro-search if needed.")
    print("\n!!! IMPORTANT: This script will make API calls that may incur costs.")
    print("!!! You are responsible for monitoring your own API usage and billing.")
    user_consent = input("!!! Do you understand and wish to proceed? (yes/no): ")
    
    if user_consent.lower() == 'yes':
        main()
    else:
        print("Operation cancelled by user.")
        sys.exit(0)
