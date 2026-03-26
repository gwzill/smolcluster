"""
NodeManager: manages cluster node selection and training process lifecycle.

Flow:
    select(hostname, ssh_user)      — mark a node as "ready" (no SSH yet)
    deselect(hostname)              — unmark
    start_training(algorithm, ...)  — SSH + launch server + all selected workers
    stop_training()                 — kill everything
    probe_username(hostname, user)  — SSH whoami to get the real username
"""

import asyncio
import logging
import shlex
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REMOTE_REPO = "~/smolcluster"


class NodeManager:
    """
    selected:  hostname → {ssh_user, rank}        — user clicked "Add to Cluster"
    processes: hostname → {proc, rank, algorithm}  — actually running SSH procs
    """

    def __init__(self):
        self.selected:  Dict[str, dict] = {}
        self.processes: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    # ── Selection (no SSH) ─────────────────────────────────────────────────────

    async def select(self, hostname: str, ssh_user: str = "", rank: Optional[int] = None) -> int:
        async with self._lock:
            if rank is None:
                existing_ranks = {v["rank"] for v in self.selected.values()}
                rank = next(r for r in range(1, 100) if r not in existing_ranks)
            self.selected[hostname] = {"ssh_user": ssh_user, "rank": rank}
        logger.info(f"[node_manager] Selected {hostname} as rank {rank}")
        return rank

    async def deselect(self, hostname: str) -> None:
        async with self._lock:
            self.selected.pop(hostname, None)
        logger.info(f"[node_manager] Deselected {hostname}")

    # ── Training lifecycle ─────────────────────────────────────────────────────

    async def start_training(self, algorithm: str, server_hostname: str) -> None:
        """
        1. Launch training server on this machine (server_hostname).
        2. SSH into each selected node and start a worker.
        """
        async with self._lock:
            if self.processes:
                raise ValueError("Training already running — stop it first")

        # Start server process locally
        server_cmd = [
            "uv", "run", "python", "-m", "smolcluster.train",
            "server", server_hostname,
            "--algorithm", algorithm,
        ]
        logger.info(f"[node_manager] Starting server: {shlex.join(server_cmd)}")
        server_proc = subprocess.Popen(server_cmd)

        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0,
                "algorithm": algorithm,
                "proc": server_proc,
                "role": "server",
            }

        # Give server a moment to bind its socket
        await asyncio.sleep(2)

        # Start workers on selected nodes
        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target = f"{ssh_user}@{hostname}.local" if ssh_user else f"{hostname}.local"
            worker_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=15",
                target,
                f"cd {REMOTE_REPO} && uv run python -m smolcluster.train "
                f"worker {rank} {hostname} --algorithm {algorithm}",
            ]
            logger.info(f"[node_manager] Starting worker {hostname} rank {rank}")
            proc = subprocess.Popen(worker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            async with self._lock:
                self.processes[hostname] = {
                    "rank": rank,
                    "algorithm": algorithm,
                    "proc": proc,
                    "role": "worker",
                }
            asyncio.create_task(self._monitor(hostname, proc))

        asyncio.create_task(self._monitor(server_hostname, server_proc))

    async def start_inference(self, server_hostname: str) -> None:
        """
        Launch inference connectivity test:
          1. Start infer server locally.
          2. SSH each selected node to run the infer worker.
        """
        async with self._lock:
            if self.processes:
                raise ValueError("Training/inference already running — stop it first")

        num_workers = len(self.selected)
        server_cmd = [
            "uv", "run", "python", "-m", "smolcluster.applications.infer",
            "server", server_hostname, str(num_workers),
        ]
        logger.info(f"[node_manager] Starting infer server: {shlex.join(server_cmd)}")
        server_proc = subprocess.Popen(server_cmd)

        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": "infer",
                "proc": server_proc, "role": "server",
            }

        await asyncio.sleep(1)

        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target = f"{ssh_user}@{hostname}.local" if ssh_user else f"{hostname}.local"
            worker_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=15",
                target,
                f"cd {REMOTE_REPO} && uv run python -m smolcluster.applications.infer "
                f"worker {rank} {hostname} {server_hostname}",
            ]
            logger.info(f"[node_manager] Starting infer worker {hostname} rank {rank}")
            proc = subprocess.Popen(worker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            async with self._lock:
                self.processes[hostname] = {
                    "rank": rank, "algorithm": "infer",
                    "proc": proc, "role": "worker",
                }
            asyncio.create_task(self._monitor(hostname, proc))

        asyncio.create_task(self._monitor(server_hostname, server_proc))

    async def stop_training(self) -> None:
        async with self._lock:
            procs = dict(self.processes)
            self.processes.clear()

        for hostname, info in procs.items():
            proc: subprocess.Popen = info["proc"]
            if proc.poll() is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
            logger.info(f"[node_manager] Stopped {hostname}")

    # ── Username probe ─────────────────────────────────────────────────────────

    @staticmethod
    async def probe_username(hostname: str, ssh_user: str = "") -> Optional[str]:
        """
        SSH to hostname.local and run `whoami`.
        Returns the username string, or None if unreachable / auth failed.
        """
        target = f"{ssh_user}@{hostname}.local" if ssh_user else f"{hostname}.local"
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",          # never prompt for password
            "-o", "ConnectTimeout=5",
            target, "whoami",
        ]
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run, cmd,
                    capture_output=True, text=True
                ),
                timeout=8.0,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"[node_manager] probe {hostname}: {e}")
        return None

    # ── Snapshots ──────────────────────────────────────────────────────────────

    def snapshot_selected(self) -> Dict[str, dict]:
        return {
            h: {"rank": v["rank"], "ssh_user": v["ssh_user"]}
            for h, v in self.selected.items()
        }

    def snapshot_processes(self) -> Dict[str, dict]:
        return {
            h: {
                "rank": v["rank"],
                "algorithm": v["algorithm"],
                "role": v["role"],
                "status": _proc_status(v["proc"]),
            }
            for h, v in self.processes.items()
        }

    async def stop_all(self) -> None:
        await self.stop_training()
        async with self._lock:
            self.selected.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _monitor(self, hostname: str, proc: subprocess.Popen) -> None:
        await asyncio.to_thread(proc.wait)
        async with self._lock:
            self.processes.pop(hostname, None)
        logger.info(f"[node_manager] {hostname} exited (rc={proc.returncode})")


def _proc_status(proc: subprocess.Popen) -> str:
    rc = proc.poll()
    return "running" if rc is None else f"exited:{rc}"
