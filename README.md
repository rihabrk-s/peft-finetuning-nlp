#  PEFT Fine-Tuning for NLP Tasks

This project demonstrates **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA** to adapt large language models for NLP tasks efficiently.  
Instead of fine-tuning all model parameters, this approach fine-tunes only a small set  reducing computational cost and memory usage.

---

##  Project Structure

peft-finetuning-nlp/
│
├── data/ # Datasets (IMDB, SST2, etc.)
├── notebooks/ # Jupyter notebooks for experimentation
├── results/ # Metrics, logs, sample outputs
├── src/ # Source code (training, evaluation)
├── Dockerfile # Container setup
├── requirements.txt # Python dependencies