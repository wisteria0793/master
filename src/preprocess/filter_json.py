"""
output_with_google_places_jp.json
This script filters a JSON file to retain only those items where the 'description' or 'description_short' fields are not null or empty.
"""

import json

def filter_and_save_json():
    """
    Reads a JSON file, filters it to keep items where 'description' or 
    'description_short' is not null or empty, and saves the result to a new file.
    """

    input_file = '../../data/raw/output_with_google_places_jp.json'
    output_file = '../../data/processed/filtered_facilities.json'

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return

    filtered_data = [
        item for item in data
        if (item.get('description') is not None and item.get('description')) or \
           (item.get('description_short') is not None and item.get('description_short'))
    ]

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully filtered {len(data)} items down to {len(filtered_data)}.")
        print(f"Filtered data saved to: {output_file}")
    except IOError:
        print(f"Error: Could not write to output file at {output_file}")

if __name__ == '__main__':
    
    filter_and_save_json()

