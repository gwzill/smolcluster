#!/usr/bin/env bash
# Full cluster bootstrap: SSH keys → local deps → local env → parallel remote setup.
#
# Usage:
#   ./scripts/installations/setup.sh              # nodes from ~/.config/smolcluster/nodes
#   ./scripts/installations/setup.sh mini2 mini3  # explicit node list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_URL="https://github.com/YuvrajSingh-mist/smolcluster.git"
REMOTE_DIR="\$HOME/Desktop/smolcluster"
KEY_PATH="$HOME/.ssh/smolcluster_key"
NODES_CACHE="$HOME/.config/smolcluster/nodes"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

log()  { echo "  [setup] $*"; }
ok()   { echo "  [setup] ✓ $*"; }
warn() { echo "  [setup] ⚠ $*" >&2; }
hr()   { echo ""; echo "  ────────────────────────────────────────────────"; }

LOCAL_USER="$(id -un 2>/dev/null || whoami)"
LOCAL_HOSTNAME="$(hostname 2>/dev/null || true)"
LOCAL_HOSTNAME_SHORT="$(hostname -s 2>/dev/null || true)"
declare -a LOCAL_NAMES=("localhost" "$LOCAL_HOSTNAME" "$LOCAL_HOSTNAME_SHORT")
declare -a LOCAL_IPS=("127.0.0.1" "::1")
while IFS= read -r ip; do
    [[ -n "$ip" ]] && LOCAL_IPS+=("$ip")
done < <(hostname -I 2>/dev/null | tr ' ' '\n')

is_local_ssh_target() {
    local node="$1"
    local cfg_user
    local cfg_host
    local name
    local ip

    cfg_user="$(ssh -G "$node" 2>/dev/null | awk '/^user / {print $2; exit}')"
    cfg_host="$(ssh -G "$node" 2>/dev/null | awk '/^hostname / {print $2; exit}')"

    # If ssh config lookup fails, fall back to alias text.
    [[ -z "$cfg_user" ]] && cfg_user="$LOCAL_USER"
    [[ -z "$cfg_host" ]] && cfg_host="$node"

    [[ "$cfg_user" != "$LOCAL_USER" ]] && return 1

    for name in "${LOCAL_NAMES[@]}"; do
        [[ -n "$name" && "$cfg_host" == "$name" ]] && return 0
    done

    for ip in "${LOCAL_IPS[@]}"; do
        [[ -n "$ip" && "$cfg_host" == "$ip" ]] && return 0
    done

    return 1
}

is_local_jetson() {
    [[ -f /etc/nv_tegra_release || "$(uname -m)" == "aarch64" ]]
}





# ─── STEP 1: SSH keys ────────────────────────────────────────────────────────
hr
echo "  STEP 1 — SSH key"
if [[ ! -f "$KEY_PATH" ]]; then
    log "No smolcluster SSH key found. Running setup_ssh.sh..."
    bash "$SCRIPT_DIR/setup_ssh.sh"
else
    ok "SSH key exists: $KEY_PATH"
fi

# ─── STEP 2: Resolve worker list ─────────────────────────────────────────────
WORKERS=()
if [[ $# -gt 0 ]]; then
    WORKERS=("$@")
elif [[ -f "$NODES_CACHE" ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && WORKERS+=("$line")
    done < "$NODES_CACHE"
fi

if [[ ${#WORKERS[@]} -eq 0 ]]; then
    warn "No worker nodes found."
    warn "Pass them as args:  $0 mini2 mini3"
    warn "Or run setup_ssh.sh first to populate $NODES_CACHE"
    exit 1
fi

hr
log "Workers: ${WORKERS[*]}"

# ─── STEP 3: Local dependency install ────────────────────────────────────────
hr
echo "  STEP 2 — Local dependencies"
bash "$SCRIPT_DIR/installation.sh"

# ─── STEP 4: Local Python environment ────────────────────────────────────────
hr
echo "  STEP 3 — Local Python environment"
cd "$PROJECT_DIR"

if [[ ! -d ".venv" ]]; then
    log "Creating .venv with Python 3.10..."
    uv venv --python 3.10 .venv
else
    ok ".venv already exists"
fi

log "Installing smolcluster (editable)..."
uv pip install -e .
ok "Local env ready: $PROJECT_DIR/.venv"

if is_local_jetson; then
    log "Jetson platform detected locally; applying Jetson-specific Python setup..."
    bash "$SCRIPT_DIR/setup_jetson.sh"
    ok "Jetson-specific Python setup complete for local node"
fi


# ─── STEP 5: Parallel remote setup ───────────────────────────────────────────
hr
echo "  STEP 4 — Remote setup on ${#WORKERS[@]} node(s) in parallel"
echo ""

# Each node runs in a subshell background process.
# Output is buffered per-node then flushed atomically to avoid interleaving.
_setup_node() {
    local node="$1"
    local pfx="  [setup:$node]"
    local out
    local status=0
    out="$(mktemp)"

    if (
        echo "$pfx Starting..."

        if is_local_ssh_target "$node"; then
            echo "$pfx Local controller target detected — skipping remote setup"
            exit 0
        fi

        # SSH reachability check
        if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" "echo ok" >/dev/null 2>&1; then
            echo "$pfx ✗ SSH unreachable — is the node online and in ~/.ssh/config?"
            exit 1
        fi

        # ── System dependencies ────────────────────────────────────────────────
        echo "$pfx Installing system dependencies (installation.sh)..."
        ssh -o BatchMode=yes "$node" "bash -s" < "$SCRIPT_DIR/installation.sh" \
            | sed "s|^|$pfx   |" || echo "$pfx   ⚠ installation.sh had warnings (may be non-fatal)"

        # ── Git clone / pull ───────────────────────────────────────────────────
        echo "$pfx Syncing repository..."
        # shellcheck disable=SC2087
        if ! ssh -o BatchMode=yes "$node" bash <<ENDSSH
            export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
            RDIR="$HOME/Desktop/smolcluster"
            if git -C "\$RDIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
                echo "Repo exists — pulling..."
                cd "\$RDIR" && git pull --ff-only 2>&1
            elif [[ -d "\$RDIR" && -n "\$(ls -A "\$RDIR" 2>/dev/null)" ]]; then
                echo "Directory exists but is not a git repo: \$RDIR"
                BKP="\${RDIR}.backup.\$(date +%Y%m%d%H%M%S)"
                mv "\$RDIR" "\$BKP"
                echo "Backed up to: \$BKP"
                echo "Cloning smolcluster..."
                git clone $REPO_URL "\$RDIR" 2>&1
            else
                echo "Cloning smolcluster..."
                mkdir -p "\$(dirname "\$RDIR")"
                git clone $REPO_URL "\$RDIR" 2>&1
            fi
ENDSSH
        then
            echo "$pfx ✗ Repository sync failed"
            exit 1
        fi
        # (output already interleaved — prefix added below via sed in parent)

        # ── Python venv + editable install ────────────────────────────────────
        echo "$pfx Setting up Python environment..."
        # shellcheck disable=SC2087
        if ! ssh -o BatchMode=yes "$node" bash <<ENDSSH
            export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
            cd "$HOME/Desktop/smolcluster"
            if [[ ! -d .venv ]]; then
                echo "Creating .venv..."
                uv venv --python 3.10 .venv 2>&1
            else
                echo ".venv already exists"
            fi
            echo "Installing smolcluster (editable)..."
            uv pip install -e . 2>&1
ENDSSH
        then
            echo "$pfx ✗ Python environment setup failed"
            exit 1
        fi

        echo "$pfx Checking for Jetson-specific Python setup..."
        if ! ssh -o BatchMode=yes "$node" "export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH\"; cd \"\$HOME/Desktop/smolcluster\"; if [[ -f /etc/nv_tegra_release || \"\$(uname -m)\" == \"aarch64\" ]]; then echo \"Jetson platform detected — running setup_jetson.sh...\"; PROJECT_DIR=\"\$HOME/Desktop/smolcluster\" bash -s; else echo \"Non-Jetson platform detected — keeping default Python packages.\"; fi" < "$SCRIPT_DIR/setup_jetson.sh"
        then
            echo "$pfx ✗ Jetson-specific Python setup failed"
            exit 1
        fi

        echo "$pfx Verifying promtail installation..."
        if ! ssh -o BatchMode=yes "$node" "export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH\"; command -v promtail >/dev/null 2>&1 || command -v promtail.exe >/dev/null 2>&1 || test -f /c/promtail/promtail.exe || test -f /mnt/c/promtail/promtail.exe || test -f '/c/Program Files/GrafanaLabs/Promtail/promtail.exe' || test -f 'C:\\\\promtail\\\\promtail.exe'"
        then
            echo "$pfx ✗ promtail not found after installation"
            echo "$pfx   Re-run: ssh $node 'bash ~/Desktop/smolcluster/scripts/installations/installation.sh'"
            exit 1
        fi

        echo "$pfx ✓ Done"
    ) >"$out" 2>&1; then
        status=0
    else
        status=$?
    fi

    cat "$out"
    rm -f "$out"
    return "$status"
}

# Launch all nodes in parallel
declare -a PIDS=()
for node in "${WORKERS[@]}"; do
    _setup_node "$node" &
    PIDS+=($!)
done

# Wait and collect results
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        ok "${WORKERS[$i]}: ready"
    else
        warn "${WORKERS[$i]}: FAILED (see output above)"
        FAIL=1
    fi
done

# ─── STEP 6: Summary ─────────────────────────────────────────────────────────
hr
echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "  Cluster setup complete — all ${#WORKERS[@]} node(s) ready."
    echo ""
    echo "  Next steps:"
    echo ""
    echo "    1. Copy .env to each worker:"
    for node in "${WORKERS[@]}"; do
        echo "         scp .env $node:~/Desktop/smolcluster/"
    done
    echo ""
    echo "    2. Start the dashboard:"
    echo "         uv run python -m smolcluster.dashboard"
    echo ""
    echo "    3. Smoke test (dry run):"
    echo "         ./scripts/inference/launch_inference.sh --algorithm syncps --dry-run"
    echo ""
else
    warn "One or more nodes failed. Fix errors above and re-run."
    exit 1
fi
