#!/bin/bash

# CAFA 6 Protein Challenge - Podman Sandbox Initialization Script
# This script sets up the initial podman environment and creates template files

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

echo "=== CAFA 6 Protein Challenge Podman Sandbox Initialization ==="
echo "Project directory: ${PROJECT_DIR}"

# Create required directories
echo "Creating required directories..."
mkdir -p "${PROJECT_DIR}/.podman/secrets"
mkdir -p "${PROJECT_DIR}/.podman/config"
mkdir -p "${PROJECT_DIR}/.podman/colab-config"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${PROJECT_DIR}/models"

# Create .gitignore for podman directory
if [ ! -f "${PROJECT_DIR}/.podman/.gitignore" ]; then
    echo "Creating .podman/.gitignore..."
    cat > "${PROJECT_DIR}/.podman/.gitignore" << 'EOF'
# Podman sandbox files
.podman/
*.token
*.secret
EOF
fi

# Create secrets directory .gitignore
if [ ! -f "${PROJECT_DIR}/.podman/secrets/.gitignore" ]; then
    echo "Creating .podman/secrets/.gitignore..."
    cat > "${PROJECT_DIR}/.podman/secrets/.gitignore" << 'EOF'
# Ignore all token files
*
EOF
fi

# Create example configuration files if they don't exist
if [ ! -f "${PROJECT_DIR}/.podman/config/container.config" ]; then
    echo "Creating example container configuration..."
    cat > "${PROJECT_DIR}/.podman/config/container.config" << 'EOF'
# Podman Container Configuration for CAFA 6 Protein Challenge

# Container name
CONTAINER_NAME="cafa6-sandbox"

# Image name and tag
IMAGE_NAME="cafa6-protein-challenge"
IMAGE_TAG="latest"

# Port mappings (uncomment if needed)
# PORTS=(
#     "8888:8888"  # Jupyter notebook
#     "6006:6006"  # TensorBoard
# )

# Volume mounts
VOLUMES=(
    "${PROJECT_DIR}:/workspace:Z"
    "${PROJECT_DIR}/.podman/secrets:/secrets:ro,Z"
    "${PROJECT_DIR}/data:/workspace/data:Z"
    "${PROJECT_DIR}/models:/workspace/models:Z"
)

# Environment variables
ENV_VARS=(
    "HUGGINGFACE_TOKEN_FILE=/secrets/huggingface_token"
    "GOOGLE_COLAB_TOKEN_FILE=/secrets/colab_token"
    "HF_HOME=/home/sandboxuser/.cache/huggingface"
    "PYTHONUNBUFFERED=1"
)

# Network configuration
NETWORK_MODE="bridge"

# Resource limits (adjust based on your system)
MEMORY_LIMIT="8g"
CPU_LIMIT="4"

# Security settings
SECURITY_OPTS=(
    "no-new-privileges"
    "label=disable"
)

DROP_CAPABILITIES=(
    "ALL"
)

ADD_CAPABILITIES=(
    "NET_BIND_SERVICE"
    "SYS_TIME"
    "DAC_OVERRIDE"
)
EOF
fi

# Create README for secrets directory
if [ ! -f "${PROJECT_DIR}/.podman/secrets/README.md" ]; then
    echo "Creating secrets README..."
    cat > "${PROJECT_DIR}/.podman/secrets/README.md" << 'EOF'
# Token Management for Podman Sandbox

## Token Files

This directory contains authentication tokens for external services.
**NEVER COMMIT THESE FILES TO VERSION CONTROL.**

### Required Token Files

1. **huggingface_token** - Hugging Face authentication token
   - Get from: https://huggingface.co/settings/tokens
   - Required for: Model downloads, dataset access, pushing models

2. **colab_token** - Google Colab authentication token
   - Get from: Google Cloud Console or Colab settings
   - Required for: Colab API access

## Token File Format

Each token file should contain **only the token string**, no additional text, quotes, or formatting.

Example `huggingface_token`:
```
hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
```

Example `colab_token`:
```
1/ABCdefGHIjklMNopQRSTuvWXYz1234567890abcdef-ghIJKLMNOPQRSTUVwxyz
```

## Creating Token Files

```bash
# For Hugging Face
echo "your_huggingface_token_here" > .podman/secrets/huggingface_token
chmod 600 .podman/secrets/huggingface_token

# For Google Colab
echo "your_colab_token_here" > .podman/secrets/colab_token
chmod 600 .podman/secrets/colab_token
```
EOF
fi

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo "WARNING: podman is not installed. Please install podman to use this sandbox."
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install podman"
    echo "  Fedora/RHEL: sudo dnf install podman"
    echo "  macOS: brew install podman"
else
    echo "Podman is installed: $(podman --version)"
fi

# Create a .env.example file for environment variables
if [ ! -f "${PROJECT_DIR}/.env.example" ]; then
    echo "Creating .env.example..."
    cat > "${PROJECT_DIR}/.env.example" << 'EOF'
# CAFA 6 Protein Challenge Environment Variables
# Copy this file to .env and modify the values for your environment

# Container configuration
CONTAINER_NAME=cafa6-sandbox
IMAGE_NAME=cafa6-protein-challenge
IMAGE_TAG=latest

# Resource limits
MEMORY_LIMIT=8g
CPU_LIMIT=4

# Network ports (uncomment if needed)
# JUPYTER_PORT=8888
# TENSORBOARD_PORT=6006

# Token file locations (relative to project directory)
HUGGINGFACE_TOKEN_FILE=.podman/secrets/huggingface_token
GOOGLE_COLAB_TOKEN_FILE=.podman/secrets/colab_token
EOF
fi

# Set permissions for scripts
chmod +x "${PROJECT_DIR}/podman-start.sh" 2>/dev/null || true
chmod +x "${PROJECT_DIR}/podman-stop.sh" 2>/dev/null || true
chmod +x "${PROJECT_DIR}/podman-tokens.sh" 2>/dev/null || true
chmod +x "${PROJECT_DIR}/podman-colab-setup.sh" 2>/dev/null || true

echo ""
echo "=== Initialization Complete ==="
echo ""
echo "Next steps:"
echo "1. Create authentication files in .podman/secrets/"
echo "   - huggingface_token (for Hugging Face access)"
echo "   - google_credentials.json (optional, for offline OAuth)"
echo ""
echo "2. Customize container configuration in .podman/config/container.config"
echo ""
echo "3. Start the sandbox:"
echo "   ./podman-start.sh"
echo ""
echo "4. Set up Colab CLI OAuth:"
echo "   podman exec -it cafa6-sandbox ./podman-colab-setup.sh setup"
echo ""
echo "5. Stop the sandbox:"
echo "   ./podman-stop.sh"
echo ""
echo "6. Connect to running container:"
echo "   podman exec -it ${CONTAINER_NAME} bash"
echo ""