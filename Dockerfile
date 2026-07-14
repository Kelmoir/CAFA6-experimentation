FROM python:3.10-slim

# Installiere Abhängigkeiten
RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

# Installiere Python-Pakete für Colab CLI und Hugging Face
RUN pip install --upgrade pip && \
    pip install google-colab huggingface_hub transformers datasets

# Erstelle Arbeitsverzeichnis im Container
RUN mkdir -p /workspace/input /workspace/output
WORKDIR /workspace

# Kopiere Skripte (falls nötig)
# COPY trainingsskript.py /workspace/input/