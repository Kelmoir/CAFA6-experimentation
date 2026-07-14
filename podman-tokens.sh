#!/bin/bash

# CAFA 6 Protein Challenge - Token Management Script
# This script helps manage authentication tokens for the podman sandbox

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
SECRETS_DIR="${PROJECT_DIR}/.podman/secrets"

echo "=== Token Management for Podman Sandbox ==="

# Ensure secrets directory exists
mkdir -p "${SECRETS_DIR}"

# Function to create a token file
create_token_file() {
    local token_name="$1"
    local token_path="${SECRETS_DIR}/${token_name}"
    local token_value="$2"
    
    if [ -z "${token_value}" ]; then
        echo "Token value cannot be empty for ${token_name}"
        return 1
    fi
    
    # Create token file with secure permissions
    echo "${token_value}" > "${token_path}"
    chmod 600 "${token_path}"
    
    echo "Created token file: ${token_path}"
    echo "Permissions: $(stat -c '%a' "${token_path}")"
}

# Function to validate token file
validate_token_file() {
    local token_name="$1"
    local token_path="${SECRETS_DIR}/${token_name}"
    
    if [ ! -f "${token_path}" ]; then
        echo "Token file does not exist: ${token_path}"
        return 1
    fi
    
    local permissions
    permissions=$(stat -c '%a' "${token_path}")
    
    if [ "${permissions}" != "600" ]; then
        echo "WARNING: Token file permissions are not secure: ${permissions}"
        echo "Recommended: 600 (read/write for owner only)"
    fi
    
    local size
    size=$(wc -c < "${token_path}")
    
    if [ "${size}" -lt 10 ]; then
        echo "WARNING: Token file seems too short: ${size} bytes"
        echo "Expected: > 20 bytes for most tokens"
    fi
    
    echo "Token file validated: ${token_path}"
    return 0
}

# Function to list token files
list_token_files() {
    echo "Token files in ${SECRETS_DIR}:"
    
    if [ -d "${SECRETS_DIR}" ]; then
        local count=0
        for file in "${SECRETS_DIR}"/*; do
            if [ -f "${file}" ]; then
                local filename
                filename=$(basename "${file}")
                local size
                size=$(wc -c < "${file}")
                local permissions
                permissions=$(stat -c '%a' "${file}")
                
                echo "  ${filename} (${size} bytes, permissions: ${permissions})"
                count=$((count + 1))
            fi
        done
        
        if [ "${count}" -eq 0 ]; then
            echo "  No token files found"
        fi
    else
        echo "  Secrets directory does not exist"
    fi
}

# Function to check token format
check_token_format() {
    local token_name="$1"
    local token_path="${SECRETS_DIR}/${token_name}"
    
    if [ ! -f "${token_path}" ]; then
        echo "Token file not found: ${token_path}"
        return 1
    fi
    
    local token_content
    token_content=$(cat "${token_path}")
    
    # Remove trailing whitespace
    token_content=$(echo "${token_content}" | sed 's/[[:space:]]*$//')
    
    # Check for common token patterns
    if [[ "${token_name}" == "huggingface_token" ]]; then
        if [[ "${token_content}" =~ ^hf_[a-zA-Z0-9]{38,}$ ]]; then
            echo "Hugging Face token format: VALID"
            return 0
        else
            echo "Hugging Face token format: INVALID"
            echo "Expected: hf_ followed by 38+ alphanumeric characters"
            return 1
        fi
    elif [[ "${token_name}" == "colab_token" ]]; then
        # Google tokens have various formats
        if [[ "${token_content}" =~ ^[a-zA-Z0-9_-]{20,}$ ]]; then
            echo "Google Colab token format: VALID"
            return 0
        else
            echo "Google Colab token format: INVALID"
            echo "Expected: Alphanumeric string with dashes and underscores"
            return 1
        fi
    else
        echo "Unknown token type: ${token_name}"
        return 1
    fi
}

# Function to test token connectivity
test_huggingface_token() {
    local token_path="${SECRETS_DIR}/huggingface_token"
    
    if [ ! -f "${token_path}" ]; then
        echo "Hugging Face token file not found"
        return 1
    fi
    
    local token
    token=$(cat "${token_path}")
    
    echo "Testing Hugging Face token connectivity..."
    
    # Test with curl to Hugging Face API
    if command -v curl &> /dev/null; then
        local response
        response=$(curl -s -w "%{http_code}" -H "Authorization: Bearer ${token}" \
            "https://huggingface.co/api/whoami-user" 2>/dev/null || echo "000")
        
        local http_code
        http_code=$(echo "${response}" | tail -c 3)
        local body
        body=$(echo "${response}" | head -n -1)
        
        if [ "${http_code}" = "200" ]; then
            echo "Hugging Face token: CONNECTED"
            return 0
        else
            echo "Hugging Face token: CONNECTION FAILED (HTTP ${http_code})"
            return 1
        fi
    else
        echo "curl not available, cannot test connectivity"
        return 1
    fi
}

# Main functions
usage() {
    echo "Usage: ./podman-tokens.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  create [token_name] [token_value]    Create a token file"
    echo "  validate [token_name]               Validate token file"
    echo "  list                              List token files"
    echo "  check [token_name]                Check token format"
    echo "  test [token_name]                 Test token connectivity"
    echo "  help                              Show this help"
    echo ""
    echo "Token names:"
    echo "  huggingface_token        - Hugging Face authentication token"
    echo "  google_credentials.json - Google OAuth2 credentials (optional)"
    echo ""
    echo "Note: Google Colab CLI uses OAuth2, not static tokens."
    echo "Use podman-colab-setup.sh to set up colab CLI authentication."
}

# Handle command line arguments
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "${1}" in
    create)
        if [ $# -lt 3 ]; then
            echo "Usage: ./podman-tokens.sh create [token_name] [token_value]"
            echo "Example: ./podman-tokens.sh create huggingface_token hf_AbCdEfGh..."
            exit 1
        fi
        create_token_file "$2" "$3"
        ;;
    validate)
        if [ $# -lt 2 ]; then
            echo "Usage: ./podman-tokens.sh validate [token_name]"
            echo "Example: ./podman-tokens.sh validate huggingface_token"
            exit 1
        fi
        validate_token_file "$2"
        ;;
    list)
        list_token_files
        ;;
    check)
        if [ $# -lt 2 ]; then
            echo "Usage: ./podman-tokens.sh check [token_name]"
            echo "Example: ./podman-tokens.sh check huggingface_token"
            exit 1
        fi
        check_token_format "$2"
        ;;
    test)
        if [ $# -lt 2 ]; then
            echo "Usage: ./podman-tokens.sh test [token_name]"
            echo "Example: ./podman-tokens.sh test huggingface_token"
            exit 1
        fi
        if [ "$2" = "huggingface_token" ]; then
            test_huggingface_token
        else
            echo "Testing not supported for token: $2"
            exit 1
        fi
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