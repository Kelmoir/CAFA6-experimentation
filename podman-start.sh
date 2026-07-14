#!/bin/bash

# CAFA 6 Protein Challenge - Podman Sandbox Start Script
# This script builds (if needed) and starts the podman container for the project

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

# Function to check if image exists
image_exists() {
    podman image exists "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null
}

# Function to build the image
build_image() {
    echo "Building container image: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    local build_cmd=(
        podman build \
            --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
            --file "${PROJECT_DIR}/Dockerfile" \
            --no-cache \
            "${PROJECT_DIR}"
    )
    
    echo "Running: ${build_cmd[*]}"
    "${build_cmd[@]}"
    
    if [ $? -ne 0 ]; then
        echo "Failed to build image"
        exit 1
    fi
    
    echo "Image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# Function to start the container
start_container() {
    echo "Starting container: ${CONTAINER_NAME}"
    
    # Prepare volume mounts
    local volume_args=()
    for volume in "${VOLUMES[@]}"; do
        # Expand environment variables in volume paths
        local expanded_volume
        expanded_volume="$(eval echo "${volume}")"
        volume_args+=("--volume" "${expanded_volume}")
    done
    
    # Prepare environment variables
    local env_args=()
    for env_var in "${ENV_VARS[@]}"; do
        env_args+=("--env" "${env_var}")
    done
    
    # Prepare port mappings
    local port_args=()
    for port in "${PORTS[@]}"; do
        if [ -n "${port}" ]; then
            port_args+=("--publish" "${port}")
        fi
    done
    
    # Prepare security options
    local security_args=()
    for opt in "${SECURITY_OPTS[@]}"; do
        security_args+=("--security-opt" "${opt}")
    done
    
    # Prepare capability options
    local capability_args=()
    for cap in "${DROP_CAPABILITIES[@]}"; do
        capability_args+=("--cap-drop" "${cap}")
    done
    for cap in "${ADD_CAPABILITIES[@]}"; do
        capability_args+=("--cap-add" "${cap}")
    done
    
    # Resource limits
    local resource_args=()
    if [ -n "${MEMORY_LIMIT:-}" ]; then
        resource_args+=("--memory" "${MEMORY_LIMIT}")
    fi
    if [ -n "${CPU_LIMIT:-}" ]; then
        resource_args+=("--cpus" "${CPU_LIMIT}")
    fi
    if [ -n "${CPU_SHARES:-}" ]; then
        resource_args+=("--cpu-shares" "${CPU_SHARES}")
    fi
    
    # Network configuration
    local network_args=()
    if [ -n "${NETWORK_NAME:-}" ]; then
        network_args+=("--network" "${NETWORK_NAME}")
    else
        network_args+=("--network" "${NETWORK_MODE}")
    fi
    
    # DNS configuration
    local dns_args=()
    for dns_opt in "${DNS_OPTIONS[@]}"; do
        dns_args+=("--dns-opt" "${dns_opt}")
    done
    
    # User namespace
    local userns_args=()
    if [ "${USERNS_MODE:-auto}" = "auto" ]; then
        userns_args+=("--userns=auto")
    fi
    
    # Auto-remove
    local rm_args=()
    if [ "${AUTO_REMOVE:-false}" = "true" ]; then
        rm_args+=("--rm")
    fi
    
    # Restart policy
    local restart_args=()
    if [ "${RESTART_POLICY:-no}" = "unless-stopped" ]; then
        restart_args+=("--restart=unless-stopped")
    elif [ "${RESTART_POLICY:-no}" = "always" ]; then
        restart_args+=("--restart=always")
    fi
    
    # Build the run command
    local run_cmd=(
        podman run \
            --detach \
            --name "${CONTAINER_NAME}" \
            "${rm_args[@]}" \
            "${network_args[@]}" \
            "${resource_args[@]}" \
            "${dns_args[@]}" \
            "${userns_args[@]}" \
            "${capability_args[@]}" \
            "${security_args[@]}" \
            "${port_args[@]}" \
            "${volume_args[@]}" \
            "${env_args[@]}" \
            "${restart_args[@]}" \
            "${IMAGE_NAME}:${IMAGE_TAG}"
    )
    
    echo "Starting container with command:"
    echo "${run_cmd[*]}"
    
    "${run_cmd[@]}"
    
    if [ $? -ne 0 ]; then
        echo "Failed to start container"
        exit 1
    fi
    
    echo "Container started successfully: ${CONTAINER_NAME}"
    echo "Container ID: $(podman container inspect --format '{{.Id}}' ${CONTAINER_NAME} | head -1)"
}

# Main logic
main() {
    echo "=== CAFA 6 Protein Challenge Podman Sandbox ==="
    echo "Project directory: ${PROJECT_DIR}"
    
    # Check if we need to build the image
    if ! image_exists; then
        echo "Image not found, building..."
        build_image
    else
        echo "Using existing image: ${IMAGE_NAME}:${IMAGE_TAG}"
    fi
    
    # Check if container already exists
    if container_exists; then
        local container_status
        container_status=$(podman container inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null)
        
        if [ "${container_status}" = "running" ]; then
            echo "Container is already running: ${CONTAINER_NAME}"
            echo "Use ./podman-stop.sh to stop it first, or use --force to restart"
            exit 0
        else
            echo "Container exists but is not running. Removing old container..."
            podman rm "${CONTAINER_NAME}" 2>/dev/null || true
        fi
    fi
    
    # Start the container
    start_container
    
    # Show container status
    echo ""
    echo "=== Container Status ==="
    podman ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo "To interact with the container:"
    echo "  podman exec -it ${CONTAINER_NAME} bash"
    echo ""
    echo "To stop the container:"
    echo "  ./podman-stop.sh"
}

# Handle command line arguments
if [ "${1:-}" = "--force" ] || [ "${1:-}" = "-f" ]; then
    echo "Force rebuild and restart requested"
    # Remove existing container if it exists
    if container_exists; then
        echo "Removing existing container..."
        podman rm "${CONTAINER_NAME}" 2>/dev/null || true
    fi
    # Always rebuild
    build_image
    start_container
elif [ "${1:-}" = "--build" ] || [ "${1:-}" = "-b" ]; then
    echo "Build only requested"
    build_image
else
    main
fi