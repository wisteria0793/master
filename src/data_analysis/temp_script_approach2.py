import pandas as pd
import re

# Load the CSV file
file_path = 'data/processed/segmentation_results_50m/segmentation_ratios.csv'
df = pd.read_csv(file_path)

# It's good practice to fill NaNs in the source data before aggregation
df.fillna(0, inplace=True)

# Extract point_id from filename
def extract_point_id(filename):
    match = re.search(r'pano_([a-zA-Z0-9_-]+)_h\d+\.jpg', filename)
    if match:
        return match.group(1)
    return None

df['point_id'] = df['filename'].apply(extract_point_id)

# Drop rows where info extraction failed
df.dropna(subset=['point_id'], inplace=True)

# Identify label columns (all columns except filename and point_id)
label_columns = df.columns.drop(['filename', 'point_id'])

# Group by point_id and aggregate
grouped = df.groupby('point_id')[label_columns]
aggregated_df = grouped.agg(['max', 'std'])

# Flatten the multi-level column names
aggregated_df.columns = [f'{col[0]}_{col[1]}' for col in aggregated_df.columns]

# The std for a single value (or if all values are the same) is NaN or 0.
# We'll fill any resulting NaNs with 0 for consistency.
aggregated_df.fillna(0, inplace=True)

# Reset index to make point_id a column again
aggregated_df.reset_index(inplace=True)

# Save the result
output_path = 'data/processed/segmentation_results_50m/feature_engineered_vectors.csv'
aggregated_df.to_csv(output_path, index=False)

print(f"Successfully created feature engineered vectors at: {output_path}")
print("\nFirst 5 rows of the new file:")
print(aggregated_df.head())
