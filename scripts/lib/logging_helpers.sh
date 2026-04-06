#!/bin/bash

# ensure_redis_running — start Redis if not already reachable.
# Works on macOS (Homebrew redis-server) and Linux/Jetson (systemd redis-server).
ensure_redis_running() {
    local redis_url="${SMOLCLUSTER_REDIS_URL:-redis://127.0.0.1:6379/0}"
    local redis_host="127.0.0.1"
    local redis_port="6379"

    # Parse host:port from URL if non-default
    if [[ "$redis_url" =~ redis://([^:/]+):([0-9]+) ]]; then
        redis_host="${BASH_REMATCH[1]}"
        redis_port="${BASH_REMATCH[2]}"
    fi

    # If already reachable, nothing to do
    if redis-cli -h "$redis_host" -p "$redis_port" ping >/dev/null 2>&1; then
        echo "📦 Redis already running on $redis_host:$redis_port"
        return 0
    fi

    if ! command -v redis-server >/dev/null 2>&1; then
        echo "[warn] redis-server not found — dashboard state will not be persisted."
        echo "       Install Redis with: brew install redis   (macOS)"
        echo "                      or: sudo apt install redis-server   (Linux)"
        return 0
    fi

    echo "📦 Starting Redis on $redis_host:$redis_port..."
    case "$(uname -s)" in
        Darwin)
            # macOS: start as user daemon via redis-server
            redis-server --daemonize yes \
                --logfile /tmp/redis.log \
                --bind 127.0.0.1 \
                --port "$redis_port" >/dev/null 2>&1 || true
            sleep 1
            ;;
        Linux)
            # Try systemd first (Ubuntu/Debian), fall back to direct daemon
            if command -v systemctl >/dev/null 2>&1 && \
               systemctl list-unit-files 2>/dev/null | grep -q '^redis'; then
                sudo systemctl start redis-server 2>/dev/null || \
                sudo systemctl start redis 2>/dev/null || true
            else
                redis-server --daemonize yes \
                    --logfile /tmp/redis.log \
                    --bind 127.0.0.1 \
                    --port "$redis_port" >/dev/null 2>&1 || true
            fi
            sleep 1
            ;;
    esac

    if redis-cli -h "$redis_host" -p "$redis_port" ping >/dev/null 2>&1; then
        echo "✅ Redis started on $redis_host:$redis_port"
    else
        echo "[warn] Redis did not start — dashboard state persistence unavailable."
    fi
}

# start_logging_stack — ensures log directory exists and Redis is running.
# Logs are streamed from remote nodes via SSH tail and stored in Redis Streams.
start_logging_stack() {
    local project_dir="$1"
    mkdir -p "$project_dir/logging/cluster-logs"
    echo "📁 Log directory ready: $project_dir/logging/cluster-logs"
    ensure_redis_running
}