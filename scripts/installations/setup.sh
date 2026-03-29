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
    out="$(mktemp)"

    {
        echo "$pfx Starting..."

        # SSH reachability check
        if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" "echo ok" >/dev/null 2>&1; then
            echo "$pfx ✗ SSH unreachable — is the node online and in ~/.ssh/config?"
            exit 1
        fi

        # ── System dependencies ────────────────────────────────────────────────
        echo "$pfx Installing system dependencies (installation.sh)..."
        ssh -o BatchMode=yes "$node" \
            "curl -fsSL https://raw.githubusercontent.com/YuvrajSingh-mist/smolcluster/main/scripts/installations/installation.sh | bash 2>&1" \
            | sed "s|^|$pfx   |" || echo "$pfx   ⚠ installation.sh had warnings (may be non-fatal)"

        # ── Git clone / pull ───────────────────────────────────────────────────
        echo "$pfx Syncing repository..."
        # shellcheck disable=SC2087
        ssh -o BatchMode=yes "$node" bash <<ENDSSH
            export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
            RDIR="$HOME/Desktop/smolcluster"
            if [[ -d "\$RDIR/.git" ]]; then
                echo "Repo exists — pulling..."
                cd "\$RDIR" && git pull --ff-only 2>&1
            else
                echo "Cloning smolcluster..."
                mkdir -p "\$(dirname "\$RDIR")"
                git clone $REPO_URL "\$RDIR" 2>&1
            fi
ENDSSH
        # (output already interleaved — prefix added below via sed in parent)

        # ── Python venv + editable install ────────────────────────────────────
        echo "$pfx Setting up Python environment..."
        # shellcheck disable=SC2087
        ssh -o BatchMode=yes "$node" bash <<ENDSSH
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

        echo "$pfx ✓ Done"
    } >"$out" 2>&1

    cat "$out"
    rm -f "$out"
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
