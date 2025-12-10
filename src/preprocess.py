import os
import json
import pandas as pd
from io import StringIO

# Update the JSON file path to 2017q1.json
RAW_PATH = "data/raw/financials/2017q1.json"
OUTPUT_DIR = "data/cleaned/reports"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" Loading JSON file: {RAW_PATH}")
with open(RAW_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert each section (NUM, PRE, SUB, TAG) stored as text into DataFrame
dfs = {}
for key, content in data.items():
    if key.endswith(".txt"):
        print(f" Converting {key} to DataFrame...")
        dfs[key] = pd.read_csv(StringIO(content), sep="\t")

# Save each DataFrame to CSV
for key, df in dfs.items():
    name = key.replace(".txt", "")
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f" Saved {path} ({df.shape[0]} rows, {df.shape[1]} columns)")

print("\n All financial tables extracted and saved successfully!")
