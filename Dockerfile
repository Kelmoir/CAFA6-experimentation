# CAFA 6 Protein Challenge - Podman Sandbox
# Lightweight container for colab CLI and Google Colab integration
# 
# Design Notes:
# - This container provides the colab CLI tool for managing Google Colab resources
# - Actual model training runs IN Google Colab (which uses PyTorch)
# - The container itself only needs basic Python + colab CLI
# - Heavy ML packages (torch, transformers) are installed automatically by Colab

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Create workspace structure
RUN mkdir -p /workspace/input 
RUN mkdir -p /workspace/output 
RUN mkdir -p /workspace/data 
RUN mkdir -p /workspace/models 
RUN mkdir -p /secrets
RUN mkdir -p /home/sandboxuser/.config 
RUN mkdir -p /home/sandboxuser/.cache/colab-cli
WORKDIR /workspace

# Install colab CLI and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir google-colab-cli huggingface-hub requests

# Create non-root user for security
RUN useradd -m -u 1000 sandboxuser && \
    chown -R sandboxuser:sandboxuser /workspace /home/sandboxuser

# Switch to non-root user
USER sandboxuser
WORKDIR /workspace

# Environment variables for authentication
ENV HUGGINGFACE_TOKEN_FILE="/secrets/huggingface_token" \
    GOOGLE_APPLICATION_CREDENTIALS="/secrets/google_credentials.json" \
    GOOGLE_AUTH_OAUTHLIB_FLOW="true" \
    HF_HOME="/home/sandboxuser/.cache/huggingface" \
    GIT_CONFIG_GLOBAL="/etc/gitconfig" \
    COLAB_CLI_CONFIG="/home/sandboxuser/.config/colab-cli" \
    OAUTHLIB_RELAX_TOKEN_SCOPE=1

# Default command - keep container running for interactive work
CMD ["tail", "-f", "/dev/null"]