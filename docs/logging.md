
# Distributed Logging

SmolCluster writes structured logs from every training node into a shared directory and exposes them through the dashboard. An optional Grafana + Loki stack is available for richer log search and retention.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Log Files](#log-files)
  - [Location](#location)
  - [Naming Convention](#naming-convention)
  - [Line Format](#line-format)
- [Dashboard Monitoring](#dashboard-monitoring)
- [Optional: Grafana + Loki Stack](#optional-grafana--loki-stack)
  - [Prerequisites](#prerequisites)
  - [Start the Stack](#start-the-stack)
  - [Start Promtail on Remote Nodes](#start-promtail-on-remote-nodes)
  - [Verify Promtail](#verify-promtail)
  - [Accessing Grafana](#accessing-grafana)
  - [LogQL Query Examples](#logql-query-examples)
- [Configuration Files](#configuration-files)

---

## Overview

Every launch script calls `start_logging_stack` which does two things:

1. Creates `logging/cluster-logs/` in the repository root (if it doesn't exist).
2. Starts Redis (required for dashboard state persistence).

Each training process — server and workers — writes a structured log file into that directory via `setup_cluster_logging()` in `src/smolcluster/utils/logging_utils.py`. No external log shipper is required for basic monitoring.

## Architecture

```
  Controller machine                  Worker nodes
  ─────────────────                  ─────────────
  ┌──────────────────┐               ┌─────────────────────┐
  │ Dashboard :8765  │◄──── Redis ───│ training metrics    │
  │ (primary UI)     │               └─────────────────────┘
  └──────────────────┘
         │
         ▼
  logging/cluster-logs/
  ├── server-mini1.log
  ├── worker-rank0-mini2.log
  └── worker-rank1-mini3.log
         ▲               ▲
  Written by server   Written by each worker
  on its own machine  on its own machine
```

> If your nodes share the repository via a network mount (NFS / Samba), all log files land in the same `logging/cluster-logs/` automatically. If nodes have independent checkouts, copy or tail logs back to the controller as needed.

## Log Files

### Location

```
<repo-root>/logging/cluster-logs/
```

`logging_utils.py` tries this path first. If it is not writable it falls back to `logging/cluster-logs-fallback/` inside the repo, then `<cwd>/smolcluster-logs/`.

### Naming Convention

| Process | File name |
|---------|-----------|
| Parameter server / leader | `server-{hostname}.log` |
| Worker rank N | `worker-rank{N}-{hostname}.log` |

Example for a three-node cluster:

```
server-mini1.log
worker-rank0-mini2.log
worker-rank1-mini3.log
```

### Line Format

Every log line follows the same structured format so it can be parsed by both humans and Promtail:

```
2026-04-06 12:34:56,789 | INFO | rank:0 | [Step 42] Loss: 2.314
```

| Field | Description |
|-------|-------------|
| timestamp | `YYYY-MM-DD HH:MM:SS,ms` |
| level | `INFO`, `WARNING`, `ERROR`, `DEBUG` |
| rank | `server` for the parameter server, integer for workers |
| message | Free-form log message |

## Dashboard Monitoring

The primary monitoring interface is the SmolCluster dashboard:

```bash
bash scripts/inference/launch_api.sh
```

Open **http://localhost:8765** in your browser. The dashboard shows:

- Live training metrics (loss, grad norm, step count) streamed via Server-Sent Events
- Node connectivity and health
- Inference token throughput when running a chat server

Redis (started automatically by `start_logging_stack`) persists the metrics state so the dashboard survives page refreshes.

---

## Optional: Grafana + Loki Stack

For full-text log search, historical retention, and cross-node log correlation, run the Grafana + Loki stack. This is optional and separate from the dashboard.

### Prerequisites

**Docker** (controller machine only):

```bash
# macOS
brew install --cask docker
# start Docker Desktop before continuing

# Linux
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # re-login after this
```

**Promtail** (every node that should ship logs):

```bash
# macOS
brew install promtail

# Linux (arm64 / Jetson)
curl -OL "https://github.com/grafana/loki/releases/download/v2.9.0/promtail-linux-arm64.zip"
unzip promtail-linux-arm64.zip && chmod +x promtail-linux-arm64
sudo mv promtail-linux-arm64 /usr/local/bin/promtail

# Linux (amd64)
curl -OL "https://github.com/grafana/loki/releases/download/v2.9.0/promtail-linux-amd64.zip"
unzip promtail-linux-amd64.zip && chmod +x promtail-linux-amd64
sudo mv promtail-linux-amd64 /usr/local/bin/promtail

promtail --version   # verify
```

### Start the Stack

Run this on the **controller machine** from the repository root:

```bash
cd logging
docker compose up -d

# Verify containers are up
docker ps | grep -E "loki|grafana|promtail"

# Wait for Loki to be ready
curl -s http://localhost:3100/ready
```

Loki listens on `:3100`, Grafana on `:3000`.

### Start Promtail on Remote Nodes

Promtail reads `logging/cluster-logs/` on each remote node and pushes to Loki on the controller. The default controller IP in the config files is `10.10.0.4` — edit `promtail-server-remote.yaml` and `promtail-worker-remote.yaml` if your controller IP differs.

**Server node (mini1):**

```bash
ssh mini1 "cd ~/Desktop/smolcluster && \
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && \
  nohup promtail -config.file=logging/promtail-server-remote.yaml \
    > /tmp/promtail.log 2>&1 &"
```

**Worker nodes (mini2, mini3, …):**

```bash
for host in mini2 mini3; do
  ssh "$host" "cd ~/Desktop/smolcluster && \
    export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:\$HOME/bin:\$PATH && \
    nohup promtail -config.file=logging/promtail-worker-remote.yaml \
      > /tmp/promtail.log 2>&1 &"
done
```

### Verify Promtail

```bash
# Check the process is running
ssh mini2 "pgrep -a promtail"

# Check for push errors
ssh mini2 "tail -20 /tmp/promtail.log"

# Confirm entries are being sent (counter should increment)
ssh mini2 "curl -s localhost:9080/metrics | grep promtail_sent_entries_total"
```

### Accessing Grafana

1. Open **http://localhost:3000**
2. Log in — anonymous access is enabled by default (no password needed). If prompted use `admin` / `admin`.
3. Go to **Explore** (compass icon) → select **Loki** as the data source.

### LogQL Query Examples

**All logs from a specific host:**

```logql
{job="smolcluster", host="mini1"}
```

**Errors across all nodes:**

```logql
{job="smolcluster"} |= "ERROR"
```

**Logs for a specific worker rank:**

```logql
{job="smolcluster"} | pattern `<_> | <level> | rank:<rank> | <_>` | rank = "0"
```

**Filter by training step:**

```logql
{job="smolcluster"} |= "[Step 100]"
```

**Loss values over time (metric query):**

```logql
{job="smolcluster"} | pattern `<_> | INFO | <_> | <_> Loss: <loss>` | unwrap loss | avg_over_time [1m]
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `logging/docker-compose.yml` | Starts Loki, Grafana, and a local Promtail in Docker |
| `logging/loki-config.yaml` | Loki storage and schema settings |
| `logging/grafana-datasources.yml` | Pre-provisions the Loki datasource in Grafana |
| `logging/promtail-config.yaml` | Promtail config for the Docker container (reads `cluster-logs/`) |
| `logging/promtail-server-remote.yaml` | Promtail config for a remote server node (pushes to controller) |
| `logging/promtail-worker-remote.yaml` | Promtail config for a remote worker node (pushes to controller) |
| `scripts/lib/logging_helpers.sh` | `start_logging_stack` / `ensure_redis_running` shell helpers |
| `src/smolcluster/utils/logging_utils.py` | `setup_cluster_logging()` — Python logging initialisation |
