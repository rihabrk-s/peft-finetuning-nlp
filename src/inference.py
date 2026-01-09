#!/usr/bin/env python3
"""
src/inference.py

Quick utility to load the base model + LoRA adapters and run generation.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--adapter_dir", type=str, required=True, help="Path to saved PEFT adapters (models/...)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def load_model(base_model, adapter_dir, device):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Loading base model (may take a while)...")
    if device.startswith("cuda"):
        base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
    else:
        # CPU fallback: patch config and force eager attention
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        if hasattr(cfg, "sliding_window"):
            cfg.sliding_window = None
        try:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=cfg,
                trust_remote_code=True,
                device_map=None,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
        except TypeError:
            print("Warning: attn_implementation kwarg not supported on this transformers version; loading with patched config only")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=cfg,
                trust_remote_code=True,
                device_map=None,
                torch_dtype=torch.float32,
            )

    # Wrap with PEFT adapters
    print("Loading adapters from:", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir, device_map=("auto" if device.startswith("cuda") else {"": "cpu"}))
    return tokenizer, model

def generate(tokenizer, model, instruction, max_new_tokens=256, temperature=0.2):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    gen_config = GenerationConfig(
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.02,
    )
    out = model.generate(**inputs, generation_config=gen_config)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    args = parse_args()
    tokenizer, model = load_model(args.base_model, args.adapter_dir, args.device)
    while True:
        instr = input("Instruction (or 'exit'): ")
        if instr.strip().lower() in ["exit", "quit"]:
            break
        print("Generating...")
        print(generate(tokenizer, model, instr))

if __name__ == "__main__":
    main()
