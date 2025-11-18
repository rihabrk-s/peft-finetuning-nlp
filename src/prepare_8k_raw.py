import pandas as pd
import os
import re
from pathlib import Path

INPUT_CSV = "data/raw/8k_filings_raw_text_2024.csv"
OUTPUT_CSV = "data/cleaned/training_data_8k_2024.csv"

CHUNK_SIZE = 3000          # safe for low RAM
MIN_TEXT_LENGTH = 300      # filter out very short filings
SUMMARY_SENTENCES = 2      # weak extractive summary (2 first sentences)

def clean_text(t):
    t = str(t)
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

def weak_summary(text, max_sent=SUMMARY_SENTENCES):
    """Simple extractive summary: first 1–2 sentences."""
    sentences = re.split(r'[.?!]\s+', text)
    if len(sentences) == 0:
        return ""
    summary = ". ".join(sentences[:max_sent]) + "."
    return summary

def process_8k_dataset(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV, chunksize=CHUNK_SIZE):
    os.makedirs(Path(output_csv).parent, exist_ok=True)

    # remove old file if exists
    if Path(output_csv).exists():
        os.remove(output_csv)

    total_rows = 0
    chunk_iter = pd.read_csv(input_csv, chunksize=chunksize)

    for idx, chunk in enumerate(chunk_iter):
        print(f"\nProcessing chunk {idx+1}...")

        # ensure column exists
        if "raw_text" not in chunk.columns:
            raise ValueError("Column 'raw_text' not found in the dataset")

        # clean text
        chunk["text"] = chunk["raw_text"].apply(clean_text)

        # filter very short filings
        chunk = chunk[chunk["text"].str.len() > MIN_TEXT_LENGTH]

        # generate weak summaries
        chunk["summary"] = chunk["text"].apply(weak_summary)

        # keep only useful columns
        chunk = chunk[["text", "summary"]]

        # append to output CSV
        chunk.to_csv(output_csv, mode="a", header=(idx==0), index=False)

        total_rows += len(chunk)
        print(f"  Saved {len(chunk)} rows this chunk | Total: {total_rows}")

    print(f"\n\n DONE! Created summarization dataset with {total_rows} samples.")
    print(f"➡ Output file: {output_csv}")

if __name__ == "__main__":
    process_8k_dataset()
