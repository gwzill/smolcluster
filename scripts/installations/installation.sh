#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[install] $*"
}

warn() {
    echo "[install][warn] $*"
}

err() {
    echo "[install][error] $*" >&2
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Required command not found: $1"
        return 1
    fi
}

ensure_macos_cmds() {
    local missing=()
    local cmd

    for cmd in "$@"; do
        command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi

    if ! command -v brew >/dev/null 2>&1; then
        err "Homebrew is required to install missing commands on macOS: ${missing[*]}"
        err "Install Homebrew from https://brew.sh and rerun."
        exit 1
    fi

    log "Installing missing macOS commands: ${missing[*]}"
    brew install "${missing[@]}"
}

install_yq_linux_fallback() {
    local arch
    local url
    local tmp

    arch="$(uname -m)"
    case "$arch" in
        x86_64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        armv7l) arch="arm" ;;
        *)
            err "Unsupported Linux architecture for yq fallback install: $arch"
            return 1
            ;;
    esac

    url="https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${arch}"
    tmp="$(mktemp)"

    log "Installing yq from upstream binary (${arch})..."
    curl -fL "$url" -o "$tmp"
    chmod +x "$tmp"
    sudo mv "$tmp" /usr/local/bin/yq
}

ensure_linux_cmds() {
    local missing=()
    local apt_missing=()
    local cmd

    require_cmd apt

    for cmd in "$@"; do
        command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi

    log "Installing missing Linux commands: ${missing[*]}"
    sudo apt update

    for cmd in "${missing[@]}"; do
        [[ "$cmd" == "yq" ]] || apt_missing+=("$cmd")
    done

    if [[ ${#apt_missing[@]} -gt 0 ]]; then
        sudo apt install -y "${apt_missing[@]}"
    fi

    if printf '%s\n' "${missing[@]}" | grep -qx "yq"; then
        if ! command -v yq >/dev/null 2>&1; then
            if ! sudo apt install -y yq; then
                warn "apt package 'yq' is unavailable on this distro. Falling back to binary install."
                install_yq_linux_fallback
            fi
        fi
    fi

    for cmd in "${missing[@]}"; do
        require_cmd "$cmd"
    done
}

ensure_essential_cmds() {
    case "$(uname -s)" in
        Darwin)
            ensure_macos_cmds curl yq
            ;;
        Linux)
            ensure_linux_cmds curl yq
            ;;
        *)
            err "Unsupported OS: $(uname -s)"
            exit 1
            ;;
    esac
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv already installed: $(command -v uv)"
        return 0
    fi

    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        warn "uv installed but not found in current PATH."
        warn "Add this to your shell config: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    else
        log "uv installed: $(command -v uv)"
    fi
}

install_macos() {
    log "Detected macOS"

    if ! command -v brew >/dev/null 2>&1; then
        err "Homebrew is required on macOS. Install from https://brew.sh and rerun."
        exit 1
    fi

    log "Updating Homebrew..."
    brew update

    log "Installing tmux, docker, colima, redis, jq..."
    brew install tmux docker colima redis jq

    # Enable Redis as a persistent Homebrew service
    if command -v redis-server >/dev/null 2>&1; then
        brew services start redis 2>/dev/null || \
            redis-server --daemonize yes --logfile /tmp/redis.log --bind 127.0.0.1 >/dev/null 2>&1 || true
        log "Redis started"
    fi

    if ! colima status >/dev/null 2>&1; then
        log "Starting Colima..."
        colima start
    else
        log "Colima already running"
    fi

    log "Verifying Docker daemon via Colima"
    docker ps

    if command -v redis-cli >/dev/null 2>&1; then
        log "redis-cli available: $(command -v redis-cli)"
    fi

    install_uv

    log "macOS dependency setup complete"
}

install_linux() {
    log "Detected Linux"

    require_cmd sudo
    require_cmd apt
    require_cmd systemctl


    log "Installing base tools (tmux, curl, certs, redis-tools, redis-server, avahi)..."
    sudo apt update

    # Install non-Docker deps first; Docker package choice is handled separately.
    sudo apt install -y tmux curl ca-certificates redis-tools redis-server avahi-daemon avahi-utils libnss-mdns

    # Enable and start Redis
    if systemctl list-unit-files 2>/dev/null | grep -q '^redis'; then
        sudo systemctl enable redis-server 2>/dev/null || sudo systemctl enable redis 2>/dev/null || true
        sudo systemctl start redis-server 2>/dev/null || sudo systemctl start redis 2>/dev/null || true
        log "Redis service enabled and started"
    fi

    if systemctl list-unit-files 2>/dev/null | grep -q '^avahi-daemon\.service'; then
        log "Enabling and starting Avahi (mDNS)"
        sudo systemctl enable avahi-daemon
        sudo systemctl start avahi-daemon
    else
        warn "avahi-daemon.service not found; mDNS may not be available on this system."
    fi

    if command -v docker >/dev/null 2>&1; then
        log "Docker CLI already available: $(command -v docker)"
    elif dpkg -s containerd.io >/dev/null 2>&1 || dpkg -s docker-ce >/dev/null 2>&1; then
        warn "Detected Docker CE/containerd.io packages; skipping docker.io to avoid conflicts."
        warn "If Docker is not usable, install/repair Docker CE packages from Docker's official repo."
        exit 1
    else
        log "Installing docker.io"
        sudo apt install -y docker.io
    fi

    if command -v docker >/dev/null 2>&1; then
        log "Enabling and starting Docker service"
        if systemctl list-unit-files 2>/dev/null | grep -q '^docker\.service'; then
            sudo systemctl enable docker
            sudo systemctl start docker
        else
            warn "docker.service not found; Docker may be managed differently on this system."
        fi

        if docker compose version >/dev/null 2>&1; then
            log "docker compose plugin already available"
        else
            log "Installing docker compose plugin"
            if ! sudo apt install -y docker-compose-plugin; then
                warn "Could not install docker-compose-plugin automatically."
                warn "Install it manually or provide legacy docker-compose binary."
                exit 1
            fi
        fi

        if id -nG "$USER" | grep -qw docker; then
            log "User already in docker group"
        else
            log "Adding user to docker group"
            sudo usermod -aG docker "$USER" && newgrp docker
        fi
    else
        warn "Docker CLI not found after package setup; skipping Docker service/group configuration."
    fi

    # Jetson / NVIDIA runtime path
    if [[ -f /etc/nv_tegra_release ]] || [[ "$(uname -m)" == "aarch64" ]]; then
        log "Jetson/ARM environment detected; attempting NVIDIA container toolkit install"
        if sudo apt install -y nvidia-container-toolkit; then
            sudo systemctl restart docker
            log "nvidia-container-toolkit installed"
        else
            warn "Could not install nvidia-container-toolkit automatically."
            warn "Install NVIDIA runtime manually if GPU containers are required."
        fi
    fi

    install_uv

    log "Checking Docker"
    if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
        docker ps
    elif command -v docker >/dev/null 2>&1; then
        warn "docker ps failed in current shell (likely needs re-login/newgrp)."
        warn "Try: newgrp docker"
        warn "Or use: sudo docker ps"
    else
        warn "Docker CLI is unavailable; install Docker CE or docker.io before running container workloads."
    fi

    if command -v redis-cli >/dev/null 2>&1; then
        log "redis-cli available: $(command -v redis-cli)"
    fi

    log "Linux dependency setup complete"
}

main() {
    ensure_essential_cmds

    require_cmd curl
    require_cmd yq

    case "$(uname -s)" in
        Darwin)
            install_macos
            ;;
        Linux)
            install_linux
            ;;
        *)
            err "Unsupported OS: $(uname -s)"
            exit 1
            ;;
    esac

    log "Done"
}

main "$@"
