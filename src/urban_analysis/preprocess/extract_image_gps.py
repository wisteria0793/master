import os
import json
import glob
from tqdm import tqdm

def extract_gps_from_json_metadata(image_dir, output_path):
    """
    Extracts GPS latitude and longitude from companion JSON files for each image.
    """
    # Find all JSON metadata files
    json_paths = sorted(glob.glob(os.path.join(image_dir, '*.json')))

    if not json_paths:
        print(f"No JSON metadata files found in directory: {image_dir}")
        return

    gps_data = {}
    files_with_gps = 0
    files_without_gps = 0

    print(f"Found {len(json_paths)} metadata files to process...")

    for path in tqdm(json_paths, desc="Extracting GPS from JSON"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # The image filename is the id + '_' + secret + '.jpg'
            image_id = metadata.get("id")
            secret = metadata.get("secret")
            if not (image_id and secret):
                files_without_gps += 1
                continue
            
            # Construct the corresponding jpg filename based on the pattern found
            image_filename = f"{image_id}_{secret}.jpg"

            location = metadata.get('location')
            if location and 'latitude' in location and 'longitude' in location:
                # Convert to float, as they might be strings
                lat = float(location['latitude'])
                lon = float(location['longitude'])
                
                # Skip if lat/lon are 0.0, which often indicates no real data
                if lat == 0.0 and lon == 0.0:
                    gps_data[image_filename] = None
                    files_without_gps += 1
                    continue

                gps_data[image_filename] = {'lat': lat, 'lon': lon}
                files_with_gps += 1
            else:
                gps_data[image_filename] = None
                files_without_gps += 1
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            filename = os.path.basename(path)
            print(f"Could not process {filename}: {e}")
            # Try to construct a key for the error case anyway
            image_filename_base = os.path.splitext(filename)[0]
            # This is a guess; might need adjustment if filenames don't match
            possible_jpg_filename = f"{image_filename_base}.jpg" 
            gps_data[possible_jpg_filename] = None
            files_without_gps += 1
            continue

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the GPS data to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gps_data, f, indent=4)

    print("\n--- GPS Extraction Summary ---")
    print(f"Total metadata files processed: {len(json_paths)}")
    print(f"Images with GPS data found: {files_with_gps}")
    print(f"Images without GPS data: {files_without_gps}")
    print(f"GPS data saved to: {output_path}")

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    image_directory = os.path.join(project_root, 'data', 'raw', 'hakodate_all_photos_bbox')
    
    # Use the user-specified output path
    output_file_path = os.path.join(project_root, 'data', 'processed', 'images', 'image_gps_data.json')
    
    extract_gps_from_json_metadata(image_directory, output_file_path)