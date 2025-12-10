#!/usr/bin/env python3
"""
src/train.py
LoRA training script for instruction SFT JSONL using PEFT + Transformers Trainer.

Expect input JSONL lines with fields: {"prompt": "...", "response": "..."}

Usage:
    python src/train.py \
      --model_name microsoft/Phi-3-mini-4k-instruct \
      --train_file data/cleaned/training_data_8k_2024_sft.jsonl \
      --output_dir models/phi3-mini-lora \
      --per_device_train_batch_size 4 \
      --num_train_epochs 3
"""
import os
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


@dataclass
class Example:
    prompt: str
    response: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="models/phi3-mini-lora")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--cutoff_len", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    return p.parse_args()


def load_jsonl_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")


def make_prompt(example: Dict) -> str:
    # This concatenates instruction -> response pattern. Adjust if your dataset uses different format.
    prompt = example.get("prompt") or ""
    response = example.get("response") or ""
    # We'll format as: "### Instruction:\n{prompt}\n\n### Response:\n{response}"
    return {"input": f"### Instruction:\n{prompt}\n\n### Response:\n", "target": response}


def tokenize_batch(batch, tokenizer, cutoff_len):
    inputs = [b["input"] for b in batch]
    targets = [b["target"] for b in batch]
    model_inputs = tokenizer(inputs, truncation=True, max_length=cutoff_len, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=cutoff_len, padding=False)

    # For causal LM, concatenate input+label into one sequence for training with attention mask.
    # Many SFT scripts instead create labels that mask the input portion. We'll implement a simple approach:
    input_ids = model_inputs["input_ids"]
    label_ids = labels["input_ids"]

    # Flatten to single sequence: input + label (but we will set labels to -100 for input portion)
    combined_input_ids = []
    combined_attention_mask = []
    combined_labels = []
    for a, b in zip(input_ids, label_ids):
        combined = a + b
        attn = [1] * len(combined)
        # labels: -100 for instruction tokens (so loss computed only on the response)
        lbl = [-100] * len(a) + b
        # truncate if needed
        if len(combined) > cutoff_len:
            combined = combined[-cutoff_len:]
            attn = attn[-cutoff_len:]
            lbl = lbl[-cutoff_len:]
        combined_input_ids.append(combined)
        combined_attention_mask.append(attn)
        combined_labels.append(lbl)

    return {"input_ids": combined_input_ids, "attention_mask": combined_attention_mask, "labels": combined_labels}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer and model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    # ensure tokenizer has pad token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare for LoRA/k-bit training (optional: keep as standard FP16 if you have sufficient memory)
    model = prepare_model_for_kbit_training(model)

    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"] if "phi" in args.model_name.lower() else None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    print("PEFT model created. Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Load dataset
    ds = load_jsonl_dataset(args.train_file)
    print("Raw dataset example:", ds[0])

    # Map to prompt/target
    ds = ds.map(lambda x: make_prompt(x), remove_columns=ds.column_names)

    # Tokenize
    def tok_func(batch):
        return tokenize_batch(batch, tokenizer, args.cutoff_len)
    tokenized = ds.map(tok_func, batched=True, batch_size=8, remove_columns=ds.column_names)

    # Data collator will pad to max in batch
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None, return_tensors="pt")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="no",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished. Saving PEFT adapters to:", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved.")

if __name__ == "__main__":
    main()
