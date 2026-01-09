#!/usr/bin/env python3
"""
FastAPI + Gradio app to serve the model.

Run:
    python -m src.web.app
This will:
 - load model + adapters
 - start a FastAPI server (port 8000)
 - start a Gradio UI (port 7860) in a background thread

Endpoints:
 - POST /api/summarize  { "text": "..." }
 - POST /api/classify   { "text": "..." }
 - POST /api/extract    { "text": "..." }
"""
import os
import threading
import time
from typing import Dict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import gradio as gr

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "models/phi3-mini-lora")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Smart Corporate Doc Assistant")

class TextIn(BaseModel):
    text: str
    max_tokens: int = 256

# Load model once at startup
print("Loading tokenizer and model (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if DEVICE == "cuda":
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, device_map="auto")
else:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if hasattr(cfg, "sliding_window"):
        cfg.sliding_window = None
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            config=cfg,
            trust_remote_code=True,
            device_map=None,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    except TypeError:
        print("Warning: attn_implementation kwarg not supported; loading with patched config only")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            config=cfg,
            trust_remote_code=True,
            device_map=None,
            torch_dtype=torch.float32,
        )
model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map=("auto" if DEVICE=="cuda" else {"": "cpu"}))
model.eval()
print("Model loaded on", DEVICE)

def gen_from_instruction(instruction: str, max_tokens=256, temperature=0.2):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    gen_config = GenerationConfig(
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        do_sample=False,
        max_new_tokens=max_tokens,
    )
    out = model.generate(**inputs, generation_config=gen_config)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the response part (strip instruction)
    if "### Response:" in text:
        return text.split("### Response:")[-1].strip()
    return text.strip()

@app.post("/api/summarize")
def summarize(payload: TextIn):
    instruction = f"Summarize the following document in a concise professional summary:\n\n{payload.text}"
    return {"summary": gen_from_instruction(instruction, max_tokens=payload.max_tokens)}

@app.post("/api/classify")
def classify(payload: TextIn):
    instruction = f"Classify the following document. Return one word: 'email', 'contract', 'report', 'invoice', 'memo', or 'other'. Text:\n\n{payload.text}"
    return {"label": gen_from_instruction(instruction, max_tokens=32)}

@app.post("/api/extract")
def extract(payload: TextIn):
    instruction = f"Extract the key fields from the document (dates, amounts, parties, deadlines) and output as short bullet points:\n\n{payload.text}"
    return {"extraction": gen_from_instruction(instruction, max_tokens=256)}

# --- Gradio UI using the same model instance ---
def create_gradio_interface():
    def summarize_fn(txt):
        return gen_from_instruction(f"Summarize the following document in a concise professional summary:\n\n{txt}", max_tokens=256)

    def extract_fn(txt):
        return gen_from_instruction(f"Extract the key fields from the document (dates, amounts, parties, deadlines) and output as short bullet points:\n\n{txt}", max_tokens=256)

    with gr.Blocks() as demo:
        gr.Markdown("# Smart Corporate Document Assistant")
        with gr.Tabs():
            with gr.TabItem("Summarize"):
                inp = gr.Textbox(lines=8, placeholder="Paste document text here...")
                out = gr.Textbox(lines=8)
                btn = gr.Button("Summarize")
                btn.click(summarize_fn, inp, out)
            with gr.TabItem("Extract"):
                inp2 = gr.Textbox(lines=8, placeholder="Paste document text here...")
                out2 = gr.Textbox(lines=8)
                btn2 = gr.Button("Extract")
                btn2.click(extract_fn, inp2, out2)
        gr.Markdown("API endpoints: POST /api/summarize /api/extract /api/classify")
    return demo

def run_gradio():
    demo = create_gradio_interface()
    # run in same process
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False)

if __name__ == "__main__":
    # Launch Gradio in a thread
    t = threading.Thread(target=run_gradio, daemon=True)
    t.start()
    print("Gradio UI launched on port 7860")
    # Launch FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
