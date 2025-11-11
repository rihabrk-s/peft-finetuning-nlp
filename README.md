# Smart Corporate Document Assistant

## About the Project
This project is all about making corporate document processing smarter and faster. Using **Parameter-Efficient Fine-Tuning (PEFT)**, I fine-tune a language model to understand and summarize professional documents like emails, financial reports, and contracts.  

The goal is simple: save time and help people quickly get insights from long and complex documents without reading everything line by line.  

## Key Features
- **Clean & Preprocess Documents**: Extract text from raw files (PDF, DOCX) and clean it for training.  
- **PEFT Fine-Tuning**: Adapt a pre-trained language model (GPT2/T5/BERT) efficiently to the corporate domain.  
- **Summarization**: Generate concise summaries of reports and emails.  
- **Information Extraction & Classification**: Identify key data like dates, names, amounts, or classify documents by type.  
- **Interactive Demo**: Run a web interface with Docker (FastAPI + Gradio) to test the model easily.  

## Project Structure
peft-finetuning-nlp/
├─ data/ # raw and cleaned datasets
├─ src/ # Python scripts (preprocessing, training, inference)
├─ models/ # fine-tuned models
├─ results/ # training logs, metrics, plots
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md



## How to Use
1. **Clone the repo**
```bash
git clone https://github.com/rihabrk-s/peft-finetuning-nlp.git
cd peft-finetuning-nlp
Install dependencies


pip install -r requirements.txt
Run Fine-Tuning


python src/train.py
Run Inference / Summarization


python src/inference.py
Run with Docker


docker-compose up --build

This project shows how to make a language model work smarter, not harder, for a real corporate use case. The idea is to combine AI efficiency with practical business impact, and I’m excited to keep building on it.