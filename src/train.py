#!/usr/bin/env python3
"""
src/train.py
LoRA training script for instruction SFT JSONL using PEFT + Transformers Trainer.

Usage:
    python src/train.py \
      --model_name microsoft/Phi-3-mini-4k-instruct \
      --train_file data/cleaned/training_data_8k_2024_sft.jsonl \
      --output_dir models/phi3-mini-lora \
      --per_device_train_batch_size 2 \
      --num_train_epochs 3
"""
import os
import argparse
from dataclasses import dataclass
from typing import Dict

import torch
import logging
import sys
import traceback
import faulthandler
from datasets import load_dataset

# Configure logging to get more info from transformers/huggingface hub
logging.basicConfig(level=logging.DEBUG)
from transformers import logging as tr_logging
tr_logging.set_verbosity_debug()
# Enable faulthandler to capture C-level crashes / segfaults
faulthandler.enable(all_threads=True)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

@dataclass
class Example:
    prompt: str
    response: str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/phi3-mini-lora")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)

    # Device / precision / optimization options
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device selection: 'auto' lets transformers decide, 'cpu', or 'cuda'")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 (mixed precision) where supported")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 where supported")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config json (optional)")
    parser.add_argument("--test_model", type=str, default=None, help="Quick test model name (e.g., 'gpt2') to validate the pipeline")

    return parser.parse_args()

def load_jsonl_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")

def make_prompt(example: Dict) -> Dict:
    prompt = example.get("prompt") or ""
    response = example.get("response") or ""
    return {"input": f"### Instruction:\n{prompt}\n\n### Response:\n", "target": response}

def tokenize_batch(batch, tokenizer, cutoff_len):
    inputs = [b["input"] for b in batch]
    targets = [b["target"] for b in batch]

    model_inputs = tokenizer(inputs, truncation=True, max_length=cutoff_len, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=cutoff_len, padding=False)

    combined_input_ids = []
    combined_attention_mask = []
    combined_labels = []

    for input_ids, label_ids in zip(model_inputs["input_ids"], labels["input_ids"]):
        combined = input_ids + label_ids
        attn = [1] * len(combined)
        lbl = [-100] * len(input_ids) + label_ids
        if len(combined) > cutoff_len:
            combined = combined[-cutoff_len:]
            attn = attn[-cutoff_len:]
            lbl = lbl[-cutoff_len:]
        combined_input_ids.append(combined)
        combined_attention_mask.append(attn)
        combined_labels.append(lbl)

    return {
        "input_ids": combined_input_ids,
        "attention_mask": combined_attention_mask,
        "labels": combined_labels,
    }

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Tokenizer loaded.")

    print("Preparing to load model...")
    # Patch config to disable sliding window (prevents flash-attention window_size errors)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if hasattr(cfg, "sliding_window"):
        cfg.sliding_window = None
    print("Patched config for CPU eager attention (sliding_window=None)")

    # Try to load model with eager attention implementation (CPU-friendly)
    # Validate test model override
    if args.test_model:
        print(f"Overriding model to test model: {args.test_model}")
        args.model_name = args.test_model

    print(f"Loading model with device={args.device}, load_in_8bit={args.load_in_8bit}, fp16={args.fp16}, bf16={args.bf16}")

    # Validate 8-bit requirement
    if args.load_in_8bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
        except Exception:
            print("Error: bitsandbytes is required for 8-bit loading but it's not installed or not compatible on this platform.")
            print("Install with: pip install bitsandbytes (Windows support is limited; WSL2/Linux recommended)")
            sys.exit(1)
        if args.device == "cpu":
            print("8-bit loading requires a CUDA device; please set --device cuda or auto and ensure CUDA PyTorch is installed.")
            sys.exit(1)

    # Device detection and selection with helpful diagnostics
    print("Torch version:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available, "device count:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))

    # Resolve requested device
    if args.device == "auto":
        resolved_device = "cuda" if cuda_available else "cpu"
    else:
        resolved_device = args.device

    if resolved_device == "cuda" and not cuda_available:
        print("Error: --device cuda requested but no CUDA device is available.\n"
              "If you don't have CUDA, run with --device cpu or --device auto to use CPU.\n"
              "If you expect a GPU to be present, ensure CUDA-enabled PyTorch is installed and CUDA drivers are available.")
        sys.exit(1)

    print(f"Resolved device: {resolved_device}")

    if resolved_device == "cpu":
        device_map = {"": "cpu"}
    elif resolved_device == "cuda":
        # On Windows, avoid distributed/auto mapping when a single GPU is available
        if torch.cuda.device_count() == 1:
            device_map = {"": "cuda:0"}
        else:
            device_map = "auto"
    else:
        device_map = "auto"

    # Choose dtype when not using 8-bit (for 8-bit we don't set dtype)
    if args.load_in_8bit:
        dtype = None
    elif args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    load_kwargs = {
        "config": cfg,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["dtype"] = dtype

    # Use eager attention on CPU to avoid unsupported window_size issues
    if args.device == "cpu":
        load_kwargs["attn_implementation"] = "eager"

    try:
        print("Calling AutoModelForCausalLM.from_pretrained() with parameters:", {k: load_kwargs.get(k) for k in ("device_map", "load_in_8bit", "dtype", "attn_implementation") if k in load_kwargs})
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
        print("from_pretrained returned successfully.")
    except TypeError:
        # Some transformer versions may not accept attn_implementation; fallback to config-only load
        print("Warning: attn_implementation kwarg not supported, retrying without it")
        load_kwargs.pop("attn_implementation", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
            print("from_pretrained fallback returned successfully.")
        except Exception as e:
            print("Error loading model in fallback:", e)
            traceback.print_exc()
            with open("model_load_error.log", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            sys.exit(1)
    except BaseException as e:
        # Catch BaseException so we can log things like SystemExit or KeyboardInterrupt if they occur unexpectedly
        print("BaseException during model.from_pretrained() ->", repr(e))
        traceback.print_exc()
        with open("model_load_error.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise
    print("Model loaded.")

    model.resize_token_embeddings(len(tokenizer))

    # LoRA config
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
    ds = ds.map(lambda x: make_prompt(x), remove_columns=ds.column_names)

    tokenized = ds.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.cutoff_len),
        batched=True,
        batch_size=4,
        remove_columns=ds.column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None, return_tensors="pt")

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="no",
        remove_unused_columns=False,
        report_to="none",
    )

    # Precision and distributed options
    if args.fp16:
        training_args_kwargs["fp16"] = True
    if args.bf16:
        training_args_kwargs["bf16"] = True
    if args.deepspeed:
        training_args_kwargs["deepspeed"] = args.deepspeed

    training_args = TrainingArguments(**training_args_kwargs)

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
