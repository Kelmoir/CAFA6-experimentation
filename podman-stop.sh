#!/bin/bash

# CAFA 6 Protein Challenge - Podman Sandbox Stop Script
# This script stops and removes the podman container for the project

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# Source configuration
CONFIG_FILE="${PROJECT_DIR}/.podman/config/container.config"
if [ -f "${CONFIG_FILE}" ]; then
    source "${CONFIG_FILE}"
else
    echo "Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

# Set default values if not defined
CONTAINER_NAME="${CONTAINER_NAME:-cafa6-sandbox}"
IMAGE_NAME="${IMAGE_NAME:-cafa6-protein-challenge}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Function to check if container exists
container_exists() {
    podman container exists "${CONTAINER_NAME}" 2>/dev/null
}

# Function to check if container is running
container_running() {
    local status
    status=$(podman container inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "")
    [ "${status}" = "running" ]
}

# Function to stop the container
stop_container() {
    echo "Stopping container: ${CONTAINER_NAME}"
    
    if container_running; then
        podman stop "${CONTAINER_NAME}"
        echo "Container stopped: ${CONTAINER_NAME}"
    else
        echo "Container is not running: ${CONTAINER_NAME}"
    fi
}

# Function to remove the container
remove_container() {
    echo "Removing container: ${CONTAINER_NAME}"
    
    if container_exists; then
        podman rm "${CONTAINER_NAME}" 2>/dev/null || true
        echo "Container removed: ${CONTAINER_NAME}"
    else
        echo "Container does not exist: ${CONTAINER_NAME}"
    fi
}

# Function to clean up resources
cleanup() {
    echo "Cleaning up container resources..."
    
    # Remove container
    remove_container
    
    # Optionally remove image (commented out by default)
    # echo "Removing image: ${IMAGE_NAME}:${IMAGE_TAG}"
    # podman rmi "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null || true
}

# Main logic
main() {
    echo "=== CAFA 6 Protein Challenge Podman Sandbox Stop ==="
    echo "Project directory: ${PROJECT_DIR}"
    
    # Check if container exists
    if ! container_exists; then
        echo "Container does not exist: ${CONTAINER_NAME}"
        echo "Nothing to stop or remove."
        exit 0
    fi
    
    # Check container status
    local container_status
    container_status=$(podman container inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null)
    
    echo "Current container status: ${container_status}"
    
    # Stop the container if it's running
    if container_running; then
        stop_container
    fi
    
    # Remove the container
    if [ "${1:-}" = "--cleanup" ] || [ "${1:-}" = "-c" ]; then
        echo "Performing cleanup..."
        cleanup
    else
        remove_container
    fi
    
    echo ""
    echo "=== Container Status ==="
    if container_exists; then
        podman ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No container found"
    else
        echo "Container ${CONTAINER_NAME} has been stopped and removed."
    fi
}

# Handle command line arguments
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: ./podman-stop.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cleanup, -c    Remove container after stopping"
    echo "  --help, -h       Show this help message"
    echo ""
    exit 0
else
    main "$@"
fi