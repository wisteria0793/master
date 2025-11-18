import os
import json
import glob
from PIL import Image
import piexif
from tqdm import tqdm

def dms_to_decimal(dms, ref):
    """Converts GPS coordinates from DMS (Degrees, Minutes, Seconds) to decimal degrees."""
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60.0
    seconds = dms[2][0] / dms[2][1] / 3600.0
    
    decimal = degrees + minutes + seconds
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps_data(image_dir, output_path):
    """
    Extracts GPS latitude and longitude from all images in a directory
    and saves the data to a JSON file.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg'))) + \
                  sorted(glob.glob(os.path.join(image_dir, '*.jpeg')))

    if not image_paths:
        print(f"No JPG/JPEG images found in directory: {image_dir}")
        return

    gps_data = {}
    images_with_gps = 0
    images_without_gps = 0

    print(f"Found {len(image_paths)} images to process...")

    for path in tqdm(image_paths, desc="Extracting GPS data"):
        filename = os.path.basename(path)
        try:
            img = Image.open(path)
            exif_dict = piexif.load(img.info.get('exif', b''))
            gps_ifd = exif_dict.get('GPS')

            if gps_ifd:
                lat_dms = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
                lon_dms = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
                lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b'N').decode('utf-8')
                lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b'E').decode('utf-8')

                if lat_dms and lon_dms:
                    lat = dms_to_decimal(lat_dms, lat_ref)
                    lon = dms_to_decimal(lon_dms, lon_ref)
                    gps_data[filename] = {'lat': lat, 'lon': lon}
                    images_with_gps += 1
                else:
                    gps_data[filename] = None
                    images_without_gps += 1
            else:
                gps_data[filename] = None
                images_without_gps += 1
        except Exception as e:
            print(f"Could not process {filename}: {e}")
            gps_data[filename] = None
            images_without_gps += 1

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the GPS data to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gps_data, f, indent=4)

    print("\n--- GPS Extraction Summary ---")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Images with GPS data: {images_with_gps}")
    print(f"Images without GPS data: {images_without_gps}")
    print(f"GPS data saved to: {output_path}")

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    image_directory = os.path.join(project_root, 'data', 'raw', 'hakodate_all_photos_bbox')
    output_file_path = os.path.join(project_root, 'data', 'processed', 'embedding', 'clip', 'image_gps_data.json')
    
    # Check for required library
    try:
        import piexif
    except ImportError:
        print("Error: 'piexif' library not found.")
        print("Please install it using: pip install piexif")
    else:
        extract_gps_data(image_directory, output_file_path)
