import os
import json
import pandas as pd
from io import StringIO

RAW_PATH = "data/raw/financials/2015q3.json"
OUTPUT_DIR = "data/cleaned/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 Loading JSON file: {RAW_PATH}")
with open(RAW_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convertir chaque section (num, pre, sub, tag) en DataFrame
dfs = {}
for key, content in data.items():
    if key.endswith(".txt"):
        print(f"➡️ Converting {key} to DataFrame...")
        dfs[key] = pd.read_csv(StringIO(content), sep="\t")

# Sauvegarder chaque DataFrame
for key, df in dfs.items():
    name = key.replace(".txt", "")
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"✅ Saved {path} ({df.shape[0]} rows, {df.shape[1]} columns)")

print("\n All financial tables extracted and saved successfully!")
