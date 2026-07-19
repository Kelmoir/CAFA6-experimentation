#!/bin/bash

# CAFA 6 Protein Challenge - Google Colab CLI Setup Script
# This script sets up the colab CLI tool with OAuth authentication

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
COLAB_CONFIG_DIR="${PROJECT_DIR}/.podman/colab-config"
SECRETS_DIR="${PROJECT_DIR}/.podman/secrets"

echo "=== Google Colab CLI Setup ==="
echo "Project directory: ${PROJECT_DIR}"

# Create required directories
mkdir -p "${COLAB_CONFIG_DIR}"
mkdir -p "${SECRETS_DIR}"

usage() {
    echo "Usage: ./podman-colab-setup.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup            Interactive OAuth setup for colab CLI"
    echo "  auth             Authenticate colab CLI"
    echo "  config           Configure colab CLI settings"
    echo "  test             Test colab CLI connection"
    echo "  status           Show current colab CLI status"
    echo "  cleanup          Remove colab CLI configuration"
    echo "  help, -h, --help  Show this help message"
    echo ""
    echo "Note: This script is designed to be run inside the podman container"
    echo "or on the host system to prepare configuration files."
}

# Function to check if we're running inside a container
is_in_container() {
    if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
        return 0
    else
        # Check if podman is running
        if podman ps >/dev/null 2>&1; then
            return 1
        else
            return 0
        fi
    fi
}

# Function to set up OAuth for colab CLI
setup_oauth() {
    echo "Setting up OAuth for Google Colab CLI..."
    
    if is_in_container; then
        echo "Running inside container - performing container-specific setup"
        
        # Ensure required directories exist within the sandbox
        echo "Creating required directories within sandbox..."
        mkdir -p "${HOME}/.config/colab-cli"
        mkdir -p "${HOME}/.cache/colab-cli"
        mkdir -p "/workspace/.colab-logs"
        
        # Set environment variables for colab CLI to use container directories
        export COLAB_CLI_CONFIG="${HOME}/.config/colab-cli"
        export COLAB_CLI_LOG_DIR="/workspace/.colab-logs"
        export COLAB_CLI_CACHE_DIR="${HOME}/.cache/colab-cli"
        
        # Check if colab CLI is installed
        if ! command -v colab &> /dev/null; then
            echo "colab CLI not found. Installing..."
            pip install --user google-colab-cli
        fi
        
        # Run OAuth flow
        echo "Starting OAuth flow for Google Colab CLI..."
        echo ""
        echo "For authentication, you have two options:"
        echo ""
        echo "Option 1: Use Google OAuth2 (interactive browser required)"
        echo "  - Run this OUTSIDE the container with X11 forwarding"
        echo "  - Or copy the OAuth URL from the container to your host browser"
        echo ""
        echo "Option 2: Use Application Default Credentials (recommended for automation)"
        echo "  - Provide google_credentials.json in .podman/secrets/"
        echo "  - Use: colab --auth adc <command>"
        echo ""
        
        # Check if we have ADC credentials
        if [ -f "/secrets/google_credentials.json" ]; then
            echo "Found Google credentials file. Using Application Default Credentials..."
            export GOOGLE_APPLICATION_CREDENTIALS="/secrets/google_credentials.json"
            
            # Test with ADC
            if colab --auth adc sessions 2>/workspace/.colab-logs/auth_error.log; then
                echo "ADC authentication successful!"
                echo "You can now use colab CLI commands with --auth adc"
            else
                echo "ADC authentication failed. Check /workspace/.colab-logs/auth_error.log"
                cat /workspace/.colab-logs/auth_error.log 2>/dev/null || true
            fi
        else
            echo "No Google credentials file found."
            echo "To use OAuth2, run this command from a terminal with browser access:"
            echo ""
            echo "  colab --auth oauth2 new"
            echo ""
            echo "Or provide google_credentials.json in .podman/secrets/ and use --auth adc"
            
            # Try OAuth2 (will fail in non-interactive mode, but show the URL)
            echo "Attempting OAuth2 flow (will show URL)..."
            colab --auth oauth2 sessions 2>&1 | grep -E "(https://|accounts.google.com|authorization code)" | head -5 || true
        fi
    else
        echo "Running on host system - preparing configuration for container"
        
        # Create instructions for manual setup
        echo "To set up colab CLI OAuth authentication:"
        echo ""
        echo "1. Start the podman sandbox:"
        echo "   ./podman-start.sh"
        echo ""
        echo "2. Enter the container:"
        echo "   podman exec -it cafa6-sandbox bash"
        echo ""
        echo "3. Run this script inside the container:"
        echo "   ./podman-colab-setup.sh setup"
        echo ""
        echo "4. Follow the OAuth flow in your browser"
        echo ""
        echo "Alternative: Set up OAuth on host and copy config to container"
        echo ""
    fi
}

# Function to authenticate colab CLI
auth_colab_cli() {
    echo "Authenticating colab CLI..."
    
    if ! command -v colab &> /dev/null; then
        echo "colab CLI not found. Please install it first:"
        echo "pip install --user google-colab-cli"
        return 1
    fi
    
    # Ensure directories exist
    mkdir -p "${HOME}/.config/colab-cli" "${HOME}/.cache/colab-cli" "/workspace/.colab-logs"
    
    # Check if we have ADC credentials
    if [ -f "/secrets/google_credentials.json" ]; then
        echo "Using Application Default Credentials..."
        export GOOGLE_APPLICATION_CREDENTIALS="/secrets/google_credentials.json"
        
        if colab --auth adc sessions >/dev/null 2>&1; then
            echo "Already authenticated with ADC"
            colab --auth adc sessions
            return 0
        fi
        
        echo "ADC credentials found but not authenticated. Try:"
        echo "  colab --auth adc new"
        return 1
    else
        echo "No ADC credentials found. OAuth2 flow requires a browser."
        echo "Please run this command from a terminal with browser access:"
        echo "  colab --auth oauth2 new"
        echo ""
        echo "Or provide google_credentials.json in .podman/secrets/ and use --auth adc"
        return 1
    fi
}

# Function to test colab CLI connection
test_colab_connection() {
    echo "Testing colab CLI connection..."
    
    if ! command -v colab &> /dev/null; then
        echo "colab CLI not found"
        return 1
    fi
    
    # Test authentication status
    echo "Checking authentication status..."
    colab auth status 2>/dev/null || echo "Not authenticated or colab CLI issue"
    
    # Test basic colab command
    echo "Testing basic colab command..."
    colab --help >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "colab CLI is working correctly"
        return 0
    else
        echo "colab CLI command failed"
        return 1
    fi
}

# Function to show colab CLI status
show_status() {
    echo "=== Colab CLI Status ==="
    
    if command -v colab &> /dev/null; then
        echo "colab CLI: Installed ($(colab --version 2>/dev/null || echo 'unknown version'))"
        
        if [ -f "${HOME}/.config/colab-cli/credentials.json" ] || [ -f "/home/sandboxuser/.config/colab-cli/credentials.json" ]; then
            echo "Credentials: Found"
        else
            echo "Credentials: Not found"
        fi
        
        # Try to get auth status
        if colab auth status >/dev/null 2>&1; then
            echo "Authentication: Active"
            colab auth status
        else
            echo "Authentication: Not active"
        fi
    else
        echo "colab CLI: Not installed"
        echo "Install with: pip install google-colab-cli"
    fi
    
    echo ""
    echo "Configuration directory: ${COLAB_CONFIG_DIR}"
    echo "Secrets directory: ${SECRETS_DIR}"
}

# Function to configure colab CLI
configure_colab_cli() {
    echo "Configuring colab CLI..."
    
    # Create config directory
    mkdir -p "${COLAB_CONFIG_DIR}"
    
    # Create default configuration
    if [ ! -f "${COLAB_CONFIG_DIR}/config.json" ]; then
        echo "Creating default colab CLI configuration..."
        cat > "${COLAB_CONFIG_DIR}/config.json" << 'EOF'
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
        "endpoint": "https://colab.research.google.com"
    },
    "settings": {
        "default_runtime": "python3",
        "timeout": 3600,
        "output_dir": "/workspace/output",
        "data_dir": "/workspace/data"
    }
}
EOF
        echo "Configuration created at: ${COLAB_CONFIG_DIR}/config.json"
    fi
    
    # Create .gitignore for colab config
    if [ ! -f "${COLAB_CONFIG_DIR}/.gitignore" ]; then
        echo "Creating .gitignore for colab config..."
        cat > "${COLAB_CONFIG_DIR}/.gitignore" << 'EOF'
# Colab CLI configuration and credentials
*.json
credentials*
token*
oauth*
EOF
    fi
}

# Function to set up colab skill for future use
setup_colab_skill() {
    echo "Setting up colab skill for future use..."
    
    if is_in_container; then
        # Inside container - we can set up the skill directly
        if command -v colab &> /dev/null; then
            echo "colab CLI is available in container"
            
            # Check if we have active authentication
            if colab auth status >/dev/null 2>&1; then
                echo "Authentication is active"
                
                # Create a skill configuration
                echo "Creating colab skill configuration..."
                
                # The colab CLI should be available as a skill
                # This might involve creating wrapper scripts or configuration
                if [ ! -f "/usr/local/bin/colab-skill" ]; then
                    echo "Creating colab skill wrapper..."
                    cat > /tmp/colab-skill << 'EOF'
#!/bin/bash
# Colab CLI skill wrapper
# This provides a standardized interface for colab operations

set -euo pipefail

case "$1" in
    "auth")
        colab auth "${@:2}"
        ;;
    "list")
        colab notebook list "${@:2}"
        ;;
    "create")
        colab notebook create "${@:2}"
        ;;
    "run")
        colab notebook run "${@:2}"
        ;;
    "clone")
        colab notebook clone "${@:2}"
        ;;
    "upload")
        colab notebook upload "${@:2}"
        ;;
    "status")
        colab auth status
        colab notebook list --limit 5
        ;;
    "help"|--help|-h)
        echo "Colab Skill - Google Colab CLI wrapper"
        echo ""
        echo "Usage: colab-skill [COMMAND] [ARGS]"
        echo ""
        echo "Commands:"
        echo "  auth [login|logout|status]  Authentication management"
        echo "  list [options]              List available notebooks"
        echo "  create [options]            Create new notebook"
        echo "  run [notebook]              Run notebook"
        echo "  clone [notebook] [name]    Clone notebook"
        echo "  upload [file]              Upload file to Colab"
        echo "  status                      Show current status"
        echo ""
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use 'colab-skill help' for usage"
        exit 1
        ;;
esac
EOF
                    
                    chmod +x /tmp/colab-skill
                    echo "Colab skill wrapper created at: /tmp/colab-skill"
                    echo "Copy to /usr/local/bin/colab-skill for system-wide use"
                fi
                
                echo "Colab skill setup complete!"
                echo ""
                echo "Available colab commands:"
                colab --help | head -20
                
            else
                echo "Need to authenticate first. Run: colab auth login"
                return 1
            fi
        else
            echo "colab CLI not installed. Install with: pip install google-colab-cli"
            return 1
        fi
    else
        # On host system - prepare configuration for container
        echo "Setting up colab skill configuration for container..."
        
        # Create a setup script that will run in the container
        cat > "${COLAB_CONFIG_DIR}/setup-colab-skill.sh" << 'EOF'
#!/bin/bash
# This script runs inside the container to set up colab skill

set -euo pipefail

echo "Setting up colab skill in container..."

# Install colab CLI if not present
if ! command -v colab &> /dev/null; then
    echo "Installing colab CLI..."
    pip install google-colab-cli
fi

# Authenticate
if ! colab auth status >/dev/null 2>&1; then
    echo "Authenticating colab CLI..."
    echo "Please follow the OAuth flow in your browser..."
    colab auth login
fi

echo "Colab CLI is ready to use!"
echo "Available commands: colab --help"
EOF
        
        chmod +x "${COLAB_CONFIG_DIR}/setup-colab-skill.sh"
        echo "Colab skill setup script created: ${COLAB_CONFIG_DIR}/setup-colab-skill.sh"
        echo ""
        echo "To complete setup, run in container:"
        echo "  podman exec -it cafa6-sandbox /workspace/.podman/colab-config/setup-colab-skill.sh"
    fi
}

# Function to clean up colab CLI configuration
cleanup_colab() {
    echo "Cleaning up colab CLI configuration..."
    
    if is_in_container; then
        # Remove credentials
        if [ -f "${HOME}/.config/colab-cli/credentials.json" ]; then
            echo "Removing credentials..."
            rm -f "${HOME}/.config/colab-cli/credentials.json"
        fi
        
        # Remove config
        if [ -d "${HOME}/.config/colab-cli" ]; then
            echo "Removing colab CLI configuration..."
            rm -rf "${HOME}/.config/colab-cli"
        fi
        
        echo "Colab CLI cleanup complete"
    else
        # On host system
        if [ -d "${COLAB_CONFIG_DIR}" ]; then
            echo "Removing colab configuration from host..."
            rm -rf "${COLAB_CONFIG_DIR}"
            mkdir -p "${COLAB_CONFIG_DIR}"  # Recreate empty directory
        fi
        
        echo "Host cleanup complete"
    fi
}

# Main logic
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "${1}" in
    setup)
        setup_oauth
        ;;
    auth)
        auth_colab_cli
        ;;
    config)
        configure_colab_cli
        ;;
    test)
        test_colab_connection
        ;;
    status)
        show_status
        ;;
    skill|skill-setup)
        setup_colab_skill
        ;;
    cleanup)
        cleanup_colab
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: ${1}"
        usage
        exit 1
        ;;
esac