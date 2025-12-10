# Use Python slim base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# copy project files
COPY . /app

# Install system deps
RUN apt-get update && apt-get install -y git build-essential libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python deps (use your requirements.txt augmented)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# default command: start the web app
CMD ["python", "-m", "src.web.app"]
