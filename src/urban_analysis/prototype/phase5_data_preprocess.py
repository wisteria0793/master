# phase5_data_preprocess.py
"""Data preprocessing for Phase 5.
Extracts `poi_id`, `rating`, `review_count` from the filtered facilities JSON
and writes them to a CSV file used by the Phase 5 recommender.
"""
import json
import csv
import hashlib
import os
from pathlib import Path

INPUT_JSON = Path(__file__).parents[3] / "data" / "processed" / "poi" / "filtered_facilities_final.json"
OUTPUT_CSV = Path(__file__).parents[3] / "data" / "processed" / "poi" / "poi_google_info.csv"
LOG_FILE = Path(__file__).parents[3] / "logs" / "phase5_preprocess.log"

def compute_file_hash(path: Path) -> str:
    """Return SHA256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    os.makedirs(LOG_FILE.parent, exist_ok=True)
    # Compute input hash for reproducibility
    input_hash = compute_file_hash(INPUT_JSON)
    # Load JSON and extract fields
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for poi in data:
        # Expected keys: "id", "rating", "review_count"
        pid = poi.get("id") or poi.get("poi_id")
        rating = poi.get("rating")
        review = poi.get("review_count")
        if pid is None:
            continue
        rows.append([pid, rating if rating is not None else "", review if review is not None else ""])
    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["poi_id", "rating", "review_count"])
        writer.writerows(rows)
    # Log
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Preprocess completed. Input hash: {input_hash}. Rows written: {len(rows)}\n")

if __name__ == "__main__":
    main()
