"""Ensure the cluster log directory exists for file-based logging."""

from pathlib import Path


def ensure_logging_infrastructure():
    """Create the local cluster-logs directory. Logs are written directly to files;
    remote node logs are streamed here via SSH tail by the launch scripts."""
    project_root = Path(__file__).parent.parent.parent.parent
    cluster_log_dir = project_root / "logging" / "cluster-logs"
    cluster_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Log directory ready: {cluster_log_dir}")
