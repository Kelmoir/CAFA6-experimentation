# CAFA 6 Protein Challenge - Podman Sandbox

## Overview

This podman sandbox provides a secure, isolated environment for running the CAFA 6 Protein Challenge project. It includes all necessary dependencies, Python packages, and secure token management for external services like Hugging Face and Google Colab.

## Features

- **Containerized Environment**: Isolated from host system
- **Non-Root User**: Runs as unprivileged user for security
- **Secure Token Management**: Separate volume for authentication tokens
- **Resource Limits**: Configurable CPU and memory constraints
- **Network Access**: Full web access for Colab and Hugging Face APIs
- **Automation Ready**: Start/stop scripts for CI/CD workflows

## Directory Structure

```
CAFA_6_protein_challenge/
├── .podman/
│   ├── secrets/              # Token files (excluded from git)
│   │   ├── huggingface_token
│   │   └── colab_token
│   └── config/
│       └── container.config # Container configuration
├── Dockerfile               # Container build instructions
├── podman-start.sh         # Start script
├── podman-stop.sh          # Stop script  
├── podman-init.sh          # Initialization script
├── podman-tokens.sh        # Token management script
└── PODMAN_SANDBOX.md       # This file
```

## Prerequisites

- **Podman** installed on host system
- **Git** for version control access
- **Internet connection** for package downloads and API access

### Podman Installation

```bash
# Ubuntu/Debian
sudo apt-get install podman

# Fedora/RHEL
sudo dnf install podman

# macOS
brew install podman
```

## Setup

### 1. Initialize the Sandbox

```bash
./podman-init.sh
```

This will:
- Create required directories
- Set up configuration files
- Create .gitignore files
- Verify podman installation

### 2. Add Authentication Tokens

#### Hugging Face Token

1. Get your token from: https://huggingface.co/settings/tokens
2. Create token file:

```bash
# Method 1: Using the token script
echo "your_huggingface_token_here" | ./podman-tokens.sh create huggingface_token

# Method 2: Manual creation
echo "your_huggingface_token_here" > .podman/secrets/huggingface_token
chmod 600 .podman/secrets/huggingface_token
```

#### Google Colab OAuth (QAuth)

**Important:** Google Colab CLI uses OAuth2 rather than static tokens.

1. Start the podman sandbox:
```bash
./podman-start.sh
```

2. Enter the container and run the colab setup:
```bash
podman exec -it cafa6-sandbox bash
./podman-colab-setup.sh setup
```

This will start the OAuth flow and open a browser window for authentication.

For offline authentication (optional):
```bash
# Copy your Google service account JSON to secrets
cp /path/to/your-service-account.json .podman/secrets/google_credentials.json
chmod 600 .podman/secrets/google_credentials.json
```

#### Validate Tokens

```bash
# List all token files
./podman-tokens.sh list

# Validate specific token
./podman-tokens.sh validate huggingface_token

# Check token format
./podman-tokens.sh check huggingface_token

# Test connectivity (requires curl)
./podman-tokens.sh test huggingface_token
```

## Usage

### Start the Sandbox

```bash
# Normal start (uses existing image if available)
./podman-start.sh

# Force rebuild and restart
./podman-start.sh --force
./podman-start.sh -f

# Build only (don't start)
./podman-start.sh --build
./podman-start.sh -b
```

### Connect to Running Container

```bash
# Interactive shell
podman exec -it cafa6-sandbox bash

# Run specific command
podman exec cafa6-sandbox python training_run.py
```

### Stop the Sandbox

```bash
# Stop and remove container
./podman-stop.sh

# Stop, remove, and cleanup
./podman-stop.sh --cleanup
./podman-stop.sh -c

# Show help
./podman-stop.sh --help
./podman-stop.sh -h
```

## Configuration

### Container Configuration

Edit `.podman/config/container.config` to customize:

```bash
# Container name and image
CONTAINER_NAME="cafa6-sandbox"
IMAGE_NAME="cafa6-protein-challenge"
IMAGE_TAG="latest"

# Resource limits (adjust based on your system)
MEMORY_LIMIT="8g"     # Memory limit
CPU_LIMIT="4"         # CPU core limit

# Port mappings (uncomment if needed)
PORTS=(
    "8888:8888"   # Jupyter notebook
    "6006:6006"   # TensorBoard
)

# Volume mounts
VOLUMES=(
    "${PROJECT_DIR}:/workspace:Z"                    # Project files
    "${PROJECT_DIR}/.podman/secrets:/secrets:ro,Z"   # Tokens (read-only)
    "${PROJECT_DIR}/data:/workspace/data:Z"          # Data files
    "${PROJECT_DIR}/models:/workspace/models:Z"      # Model files
)
```

### Token and OAuth Management

Authentication files are stored in `.podman/secrets/` and mounted as read-only volumes in the container. The container accesses authentication through environment variables:

- `HUGGINGFACE_TOKEN_FILE=/secrets/huggingface_token`
- `GOOGLE_APPLICATION_CREDENTIALS=/secrets/google_credentials.json`

In your Python scripts, you can access authentication like:

```python
import os

# Read Hugging Face token
with open(os.environ.get('HUGGINGFACE_TOKEN_FILE', '/secrets/huggingface_token'), 'r') as f:
    hf_token = f.read().strip()

# Use with Hugging Face Hub
from huggingface_hub import login
login(token=hf_token)

# For Google Colab - use the colab CLI directly
# The OAuth credentials are managed by the colab CLI automatically
import subprocess
result = subprocess.run(['colab', 'auth', 'status'], capture_output=True, text=True)
```

For Google Colab operations, use the `colab` CLI tool directly or through the colab skill wrapper.

## Security Features

### Container Security

- **Non-root user**: Container runs as `sandboxuser` (UID 1000)
- **Capability dropping**: All capabilities dropped except essential ones
- **No new privileges**: Prevents privilege escalation
- **SELinux context**: Proper labeling with `:Z` volume mounts
- **Read-only tokens**: Token files mounted as read-only

### Network Security

- **Default bridge network**: Isolated network stack
- **Customizable network**: Can be configured for specific use cases
- **DNS isolation**: Uses container-specific DNS resolution

### Token Security

- **Secure permissions**: Token files created with 600 permissions
- **Git exclusion**: `.podman/` directory excluded from version control
- **Read-only mounting**: Tokens are read-only in the container
- **Separate volume**: Tokens mounted from separate directory

## Dockerfile

The `Dockerfile` is optimized for the CAFA 6 project with:

- **Multi-stage build**: Reduces final image size
- **Python 3.10 slim base**: Minimal Python environment
- **All requirements installed**: Full dependency installation
- **Workspace setup**: Pre-configured directories
- **Health checks**: Verifies TensorFlow availability

## Automation

The sandbox is designed for automation workflows:

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Setup Podman Sandbox
  run: |
    ./podman-init.sh
    ./podman-start.sh --build

- name: Run Training
  run: |
    podman exec cafa6-sandbox python training_run.py

- name: Stop Sandbox
  run: ./podman-stop.sh --cleanup
```

### Scheduled Jobs

```bash
# Start sandbox for scheduled job
./podman-start.sh

# Run job
podman exec cafa6-sandbox python generate_dataset.py

# Stop sandbox when done
./podman-stop.sh --cleanup
```

## Troubleshooting

### Common Issues

1. **Podman not found**: Install podman as shown in prerequisites
2. **Permission denied**: Run scripts with `chmod +x` and ensure proper permissions
3. **Port conflicts**: Check if ports in configuration are available
4. **Token validation fails**: Verify token format and file permissions

### Debug Commands

```bash
# Show running containers
podman ps

# Show all containers (including stopped)
podman ps -a

# View container logs
podman logs cafa6-sandbox

# Inspect container
podman inspect cafa6-sandbox

# Build with verbose output
podman build --verbose .
```

## Cleanup

### Remove Container and Image

```bash
# Stop and remove container
./podman-stop.sh --cleanup

# Remove image (manual)
podman rmi cafa6-protein-challenge:latest

# Remove all podman resources
podman system prune -a
```

### Remove Local Files

```bash
# Remove podman directory
rm -rf .podman/

# Remove container cache
podman system prune
```

## Best Practices

1. **Always use tokens**: Never hardcode credentials in scripts
2. **Validate tokens**: Use `podman-tokens.sh` to validate token files
3. **Secure permissions**: Ensure token files have 600 permissions
4. **Git ignore**: Never commit token files or `.podman/secrets/`
5. **Resource limits**: Set appropriate resource limits for your system
6. **Regular updates**: Update base image and dependencies regularly

## Support

For issues with:
- **Podman**: https://github.com/containers/podman
- **Hugging Face**: https://huggingface.co/docs
- **Google Colab**: https://colab.research.google.com/