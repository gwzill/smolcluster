"""Auto-start centralized logging infrastructure."""

import subprocess
import time
from pathlib import Path


def ensure_logging_infrastructure():
    """
    Automatically start Loki/Grafana/Promtail if not running.
    Only runs on controller node (where docker-compose exists).
    Workers just write log files.
    """
    # Check if we're on the controller (has docker-compose file)
    project_root = Path(__file__).parent.parent.parent.parent
    cluster_log_dir = project_root / "logging" / "cluster-logs"
    docker_compose_path = project_root / "logging" / "docker-compose.yml"

    if not docker_compose_path.exists():
        # Not on controller, just create log directory
        cluster_log_dir.mkdir(parents=True, exist_ok=True)
        return

    # On controller - check if Loki is already running
    try:
        import requests

        response = requests.get("http://localhost:3100/ready", timeout=1)
        if response.status_code == 200:
            print("✅ Logging infrastructure already running")
            return
    except Exception:
        pass

    # Not running - start it
    print("🚀 Starting logging infrastructure (Loki + Grafana + Promtail)...")

    try:
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd=docker_compose_path.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            # Wait for Loki to be ready
            for _i in range(10):
                try:
                    import requests

                    response = requests.get("http://localhost:3100/ready", timeout=1)
                    if response.status_code == 200:
                        print("✅ Logging infrastructure started!")
                        print("📊 Grafana UI: http://localhost:3000 (admin/admin)")
                        return
                except Exception:
                    time.sleep(1)

            print("⚠️  Logging infrastructure started but Loki not ready yet")
        else:
            print(f"⚠️  Failed to start logging: {result.stderr}")
    except FileNotFoundError:
        print(
            "⚠️  docker-compose not found. Install Docker to enable centralized logging."
        )
    except Exception as e:
        print(f"⚠️  Could not start logging infrastructure: {e}")

    # Create log directory anyway
    cluster_log_dir.mkdir(parents=True, exist_ok=True)
