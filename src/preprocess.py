# preprocess_local.py
from datasets import load_dataset
from transformers import AutoTokenizer
import json

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH = "data/cleaned/training_data_8k_2024_sft.jsonl"
MAX_LENGTH = 2048
SAMPLE_OUTPUT = "tokenized_sample.json"

# ---------------------------
# Load tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Phi-3-mini may not have pad token

# ---------------------------
# Load dataset
# ---------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print("Raw dataset example:")
print(dataset[0])

# ---------------------------
# Format prompt function
# ---------------------------
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

# ---------------------------
# Tokenization function
# ---------------------------
def tokenize_fn(example):
    tokens = tokenizer(
        format_prompt(example),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    # LoRA / Trainer requires labels
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# ---------------------------
# Tokenize dataset
# ---------------------------
print("\nTokenizing one sample for sanity check...")
tok_sample = tokenize_fn(dataset[0])
for k, v in tok_sample.items():
    print(k, type(v), len(v))

# ---------------------------
# Map over entire dataset (optional, can be large)
# ---------------------------
# tokenized_ds = dataset.map(
#     tokenize_fn,
#     remove_columns=dataset.column_names
# )
# print(tokenized_ds[0])

# ---------------------------
# Save a small tokenized sample for checking
# ---------------------------
with open(SAMPLE_OUTPUT, "w") as f:
    json.dump(tok_sample, f, indent=2)

print(f"\n✅ Tokenized sample saved to {SAMPLE_OUTPUT}")
