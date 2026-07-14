# Google Colab CLI Guide for CAFA 6 Protein Challenge

## Overview

This guide explains how to use the Google Colab CLI tool (`colab`) within the podman sandbox for the CAFA 6 Protein Challenge project.

## Prerequisites

- Podman sandbox must be running: `./podman-start.sh`
- `google-colab-cli` package is installed in the container (automatically via requirements.txt)
- OAuth authentication must be completed

## Quick Start

### 1. Start the Podman Sandbox

```bash
./podman-start.sh
```

### 2. Set Up Colab CLI Authentication

```bash
# Enter the container
podman exec -it cafa6-sandbox bash

# Run the colab setup script (starts OAuth flow)
./podman-colab-setup.sh setup
```

The OAuth flow will open a browser window. Sign in with your Google account and authorize the colab CLI application.

### 3. Verify Authentication

```bash
# Check authentication status
colab auth status

# List available notebooks
colab notebook list
```

## Colab CLI Commands

The `colab` CLI provides several commands for managing Google Colab notebooks:

### Authentication Commands

```bash
# Login (starts OAuth flow)
colab auth login

# Logout
colab auth logout

# Check authentication status
colab auth status
```

### Notebook Commands

```bash
# List notebooks
colab notebook list
colab notebook list --limit 10
colab notebook list --query "CAFA6"

# Create a new notebook
colab notebook create --name "CAFA6 Training" --runtime python3

# Clone a notebook
colab notebook clone --source "notebook-id" --name "My Copy"

# Run a notebook
colab notebook run --name "CAFA6 Training" --command "python training_run.py"

# Upload a file to Colab
colab notebook upload --file "/workspace/training_run.py" --name "Uploaded Training Script"

# Get notebook details
colab notebook info --name "CAFA6 Training"

# Delete a notebook
colab notebook delete --name "CAFA6 Training"
```

### Advanced Commands

```bash
# Execute Python code directly
colab notebook execute --code "print('Hello from Colab!')"

# Upload and run a script
colab notebook upload --file "/workspace/training_run.py" --name "Training"
colab notebook run --name "Training"

# Monitor running notebooks
colab notebook list --status running

# Get logs from a running notebook
colab notebook logs --name "CAFA6 Training"
```

## Using Colab CLI in Python Scripts

You can call the colab CLI from your Python scripts:

```python
import subprocess
import json

# Check authentication
result = subprocess.run(['colab', 'auth', 'status'], 
                       capture_output=True, text=True)
print(f"Auth status: {result.stdout}")

# List notebooks
result = subprocess.run(['colab', 'notebook', 'list', '--limit', '5'], 
                       capture_output=True, text=True)
notebooks = json.loads(result.stdout)
for notebook in notebooks:
    print(f"Notebook: {notebook['name']} (ID: {notebook['id']})")

# Run a notebook
result = subprocess.run(['colab', 'notebook', 'run', 
                       '--name', 'CAFA6 Training',
                       '--command', 'python training_run.py'], 
                       capture_output=True, text=True)
print(f"Run result: {result.stdout}")

# Upload and run
subprocess.run(['colab', 'notebook', 'upload', 
                '--file', '/workspace/generate_dataset.py',
                '--name', 'Dataset Generation'])
subprocess.run(['colab', 'notebook', 'run', '--name', 'Dataset Generation'])
```

## Colab Skill Setup

For automated workflows, set up the colab skill:

```bash
# Inside the container
./podman-colab-setup.sh skill-setup
```

This creates a `colab-skill` wrapper that provides a standardized interface for colab operations.

### Using the Colab Skill

```bash
# List notebooks
colab-skill list

# Check status
colab-skill status

# Upload a file
colab-skill upload /workspace/data/processed.csv

# Run a notebook
colab-skill run "CAFA6 Training"
```

## Configuration

### Environment Variables

The following environment variables are set in the container:

- `GOOGLE_APPLICATION_CREDENTIALS=/secrets/google_credentials.json`
- `GOOGLE_AUTH_OAUTHLIB_FLOW=true`
- `COLAB_CLI_CONFIG=/home/sandboxuser/.config/colab-cli`

### Configuration Files

Colab CLI configuration is stored in:
- `~/.config/colab-cli/config.json` - CLI configuration
- `~/.config/colab-cli/credentials.json` - OAuth credentials (auto-generated)

### Custom Configuration

To customize colab CLI behavior, edit the configuration file:

```bash
# Edit the configuration
nano ~/.config/colab-cli/config.json
```

Example configuration:
```json
{
    "auth": {
        "flow": "browser",
        "scopes": [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/colab"
        ]
    },
    "api": {
        "endpoint": "https://colab.research.google.com",
        "timeout": 3600
    },
    "settings": {
        "default_runtime": "python3",
        "output_dir": "/workspace/output",
        "data_dir": "/workspace/data"
    }
}
```

## Offline Authentication

For environments without browser access, you can use service account credentials:

1. Create a Google Cloud service account with appropriate permissions
2. Download the JSON key file
3. Copy it to the secrets directory:

```bash
# On the host
cp /path/to/your-service-account.json .podman/secrets/google_credentials.json
chmod 600 .podman/secrets/google_credentials.json

# Then start the container (it will use the service account)
./podman-start.sh --force
```

## Troubleshooting

### Common Issues

1. **OAuth flow doesn't open browser**
   - Ensure you have a working browser on the host system
   - Try running `colab auth login` with `--browser=firefox` or `--browser=chrome`
   - For headless systems, use service account authentication

2. **Authentication expires**
   - Run `colab auth login` again to refresh credentials
   - OAuth tokens typically expire after 1 hour (refresh tokens may last longer)

3. **colab command not found**
   - Ensure the container is running: `podman ps`
   - Check if colab CLI is installed: `podman exec cafa6-sandbox pip list | grep colab`
   - Rebuild the container: `./podman-start.sh --force`

4. **Permission denied**
   - Ensure the container has network access
   - Check if you're running as the correct user inside the container

### Debug Commands

```bash
# Check colab CLI version
colab --version

# Verbose output
colab --verbose notebook list

# Check authentication details
colab auth status --verbose

# Test connectivity to Colab API
curl -v https://colab.research.google.com
```

## Best Practices

1. **Authentication Lifecycle**: OAuth tokens expire. Plan for token refresh in long-running workflows.

2. **Resource Management**: Colab notebooks consume Google Cloud resources. Clean up unused notebooks.

3. **Data Management**: Use Colab's persistent storage or mount your project directories.

4. **Error Handling**: Always check the exit codes of colab CLI commands in scripts.

5. **Security**: Keep your OAuth credentials secure. Don't share the container with untrusted users.

## Integration with Workflows

### GitHub Actions Example

```yaml
- name: Run Training on Colab
  run: |
    ./podman-start.sh
    podman exec cafa6-sandbox bash -c '
      ./podman-colab-setup.sh auth &&
      colab notebook upload --file /workspace/training_run.py --name "CAFA6 Training" &&
      colab notebook run --name "CAFA6 Training"
    '
    ./podman-stop.sh --cleanup
```

### Local Automation

```bash
#!/bin/bash
# Automated training workflow

./podman-start.sh

# Wait for container to be ready
sleep 10

# Upload and run training
podman exec cafa6-sandbox bash -c '
  ./podman-colab-setup.sh auth &&
  colab notebook upload --file /workspace/training_run.py --name "Training" &&
  colab notebook run --name "Training" --command "python training_run.py --epochs 10"
'

# Cleanup
./podman-stop.sh --cleanup
```

## Support

- **colab CLI Documentation**: https://github.com/google/colab-cli
- **Google Colab**: https://colab.research.google.com/
- **OAuth2 Documentation**: https://developers.google.com/identity/protocols/oauth2