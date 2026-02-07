#!/usr/bin/env python3
"""
QLoRA fine-tuning script (Kaggle GPU - NVIDIA T4)
Optimized to finish in reasonable time.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# ======================
# PATHS & CONFIG
# ======================

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH = "/kaggle/input/training-data-8k-2024-sft-jsonl/training_data_8k_2024_sft.jsonl"
OUTPUT_DIR = "models/phi3-qlora-kaggle"

MAX_LENGTH = 512                # 🚀 BIG speedup
NUM_PROC = 4
TRAIN_SIZE = 20000              # 🚀 enough for LoRA

# ======================
# QLoRA (4-bit) CONFIG
# ======================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ======================
# TOKENIZER
# ======================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ======================
# MODEL
# ======================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"   # 🚫 flash-attn drama
)

model = prepare_model_for_kbit_training(model)

# 🔥 Disable checkpointing = faster (fits on T4 in 4-bit)
model.gradient_checkpointing_disable()

# ======================
# LoRA CONFIG
# ======================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# DATASET
# ======================

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# Shuffle + subset (LoRA does NOT need full dataset)
dataset = dataset.shuffle(seed=42).select(range(TRAIN_SIZE))

# ---- FORMAT PROMPT ----

def format_prompt(example):
    if example["input"].strip():
        user_text = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_text = example["instruction"]

    return {
        "text": (
            "<|user|>\n"
            f"{user_text}\n"
            "<|assistant|>\n"
            f"{example['output']}"
        )
    }

dataset = dataset.map(
    format_prompt,
    remove_columns=dataset.column_names,
    num_proc=NUM_PROC
)

# ---- TOKENIZATION ----

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
    num_proc=NUM_PROC
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ======================
# TRAINING
# ======================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # 🚀 faster
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=25,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

# ======================
# SAVE ADAPTER
# ======================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ QLoRA training finished successfully")
