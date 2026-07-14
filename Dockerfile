# CAFA 6 Protein Challenge - Podman Sandbox
# Multi-stage build for optimized container

# Base stage with Python and system dependencies
FROM python:3.10-slim as base

# Install system dependencies for Python packages and OAuth
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        ca-certificates \
        libssl-dev \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create workspace structure
RUN mkdir -p /workspace/input /workspace/output /workspace/data /workspace/models
WORKDIR /workspace

# Final stage
FROM base as final

# Create a non-root user for security
RUN useradd -m -u 1000 sandboxuser && \
    mkdir -p /home/sandboxuser/.config && \
    chown -R sandboxuser:sandboxuser /workspace /home/sandboxuser

# Switch to non-root user
USER sandboxuser
WORKDIR /workspace

# Environment variables for token locations
ENV HUGGINGFACE_TOKEN_FILE="/secrets/huggingface_token" \
    GOOGLE_COLAB_TOKEN_FILE="/secrets/colab_token" \
    HF_HOME="/home/sandboxuser/.cache/huggingface" \
    GIT_CONFIG_GLOBAL="/etc/gitconfig"

# Create secrets directory (will be mounted from host)
RUN mkdir -p /secrets

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; print('TensorFlow OK')" || exit 1

# Default command - keep container running for interactive work
CMD ["tail", "-f", "/dev/null"]