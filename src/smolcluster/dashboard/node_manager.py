"""
NodeManager: manages cluster node selection and process lifecycle.

Flow:
    select(hostname, ssh_user)      — mark a node as "ready" (no SSH yet)
    deselect(hostname)              — unmark
    start_training(algorithm, ...)  — launch server locally + SSH workers
    start_inference(server_host)    — launch infer server + SSH workers
    stop_training()                 — kill everything
    probe_username(hostname, user)  — SSH whoami
"""

import asyncio
import logging
import os
import shlex
import subprocess
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REMOTE_REPO = "~/smolcluster"
LOG_MAXLINES = 500  # global circular buffer


def _build_ssh_target(ssh_user: str, hostname: str) -> str:
    """
    Build the SSH target from whatever the user provided:
      - bare alias  "mini2"               → use as-is  (SSH config handles user+key)
      - username    "yuvrajsingh2"        → "yuvrajsingh2@hostname.local"
      - user@host   "yuvrajsingh2@mini2"  → use as-is
      - empty                             → "hostname.local"
    Heuristic: a bare alias has no '@' and no '.' in it.
    """
    if not ssh_user:
        return f"{hostname}.local"
    if "@" in ssh_user:       # already fully qualified
        return ssh_user
    if "." not in ssh_user:   # bare alias like "mini2"
        return ssh_user
    return f"{ssh_user}@{hostname}.local"


class NodeManager:
    """
    selected:  hostname → {ssh_user, rank}
    processes: hostname → {proc, rank, algorithm, role}
    """

    def __init__(self):
        self.selected:  Dict[str, dict] = {}
        self.processes: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        # Log streaming — global ordered list, trimmed to LOG_MAXLINES
        self._logs: List[dict] = []
        self._log_seq = 0

    # ── Log helpers ────────────────────────────────────────────────────────────

    def _log(self, hostname: str, line: str):
        self._log_seq += 1
        self._logs.append({"seq": self._log_seq, "hostname": hostname,
                            "line": line, "ts": time.time()})
        if len(self._logs) > LOG_MAXLINES:
            self._logs = self._logs[-LOG_MAXLINES:]

    def logs_since(self, seq: int = 0) -> List[dict]:
        return [l for l in self._logs if l["seq"] > seq]

    # ── Selection ──────────────────────────────────────────────────────────────

    async def select(self, hostname: str, ssh_user: str = "",
                     rank: Optional[int] = None) -> int:
        async with self._lock:
            if rank is None:
                taken = {v["rank"] for v in self.selected.values()}
                rank = next(r for r in range(1, 100) if r not in taken)
            self.selected[hostname] = {"ssh_user": ssh_user, "rank": rank}
        logger.info(f"[node_manager] Selected {hostname} as rank {rank}")
        return rank

    async def deselect(self, hostname: str) -> None:
        async with self._lock:
            self.selected.pop(hostname, None)
        logger.info(f"[node_manager] Deselected {hostname}")

    # ── Training lifecycle ─────────────────────────────────────────────────────

    async def start_training(self, algorithm: str, server_hostname: str) -> None:
        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        # ── Server (local) ────────────────────────────────────────────────────
        server_cmd = [
            "uv", "run", "python", "-m", "smolcluster.train",
            "server", server_hostname, "--algorithm", algorithm,
        ]
        self._log(server_hostname, f"$ {shlex.join(server_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *server_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": algorithm, "role": "server", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))

        await asyncio.sleep(2)  # let server bind socket

        # ── Workers (SSH) ─────────────────────────────────────────────────────
        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target = _build_ssh_target(ssh_user, hostname)
            remote = (
                f"cd {REMOTE_REPO} && PYTHONUNBUFFERED=1 "
                f"uv run python -m smolcluster.train "
                f"worker {rank} {hostname} --algorithm {algorithm}"
            )
            cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                   target, remote]
            self._log(hostname, f"$ ssh {target} [rank {rank}]")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            async with self._lock:
                self.processes[hostname] = {
                    "rank": rank, "algorithm": algorithm, "role": "worker", "proc": proc,
                }
            asyncio.create_task(self._stream(hostname, proc))

    # ── Inference lifecycle ────────────────────────────────────────────────────

    async def start_inference(self, server_hostname: str) -> None:
        """
        Launch infer server locally + SSH each selected worker to run infer worker.
        """
        async with self._lock:
            if self.processes:
                raise ValueError("Already running — stop it first")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        num_workers = len(self.selected)

        # ── Infer server (local) ──────────────────────────────────────────────
        server_cmd = [
            "uv", "run", "python", "-m", "smolcluster.applications.infer",
            "server", server_hostname, str(num_workers),
        ]
        self._log(server_hostname, f"$ {shlex.join(server_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *server_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        async with self._lock:
            self.processes[server_hostname] = {
                "rank": 0, "algorithm": "infer", "role": "server", "proc": proc,
            }
        asyncio.create_task(self._stream(server_hostname, proc))

        await asyncio.sleep(1)

        # ── Workers (SSH) ─────────────────────────────────────────────────────
        async with self._lock:
            selected = dict(self.selected)

        for hostname, info in selected.items():
            if hostname == server_hostname:
                continue
            rank = info["rank"]
            ssh_user = info.get("ssh_user", "")
            target = _build_ssh_target(ssh_user, hostname)
            remote = (
                f"cd {REMOTE_REPO} && PYTHONUNBUFFERED=1 "
                f"uv run python -m smolcluster.applications.infer "
                f"worker {rank} {hostname} {server_hostname}"
            )
            cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                   target, remote]
            self._log(hostname, f"$ ssh {target} → infer worker rank {rank}")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            async with self._lock:
                self.processes[hostname] = {
                    "rank": rank, "algorithm": "infer", "role": "worker", "proc": proc,
                }
            asyncio.create_task(self._stream(hostname, proc))

    async def stop_training(self) -> None:
        async with self._lock:
            procs = dict(self.processes)
            self.processes.clear()

        for hostname, info in procs.items():
            proc = info["proc"]
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
            logger.info(f"[node_manager] Stopped {hostname}")

    # ── Username probe ─────────────────────────────────────────────────────────

    @staticmethod
    async def probe_username(hostname: str, ssh_user: str = "") -> Optional[str]:
        target = _build_ssh_target(ssh_user, hostname)
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
               "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               target, "whoami"]
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True),
                timeout=8.0,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"[node_manager] probe {hostname}: {e}")
        return None

    # ── Snapshots ──────────────────────────────────────────────────────────────

    def snapshot_selected(self) -> Dict[str, dict]:
        return {h: {"rank": v["rank"], "ssh_user": v["ssh_user"]}
                for h, v in self.selected.items()}

    def snapshot_processes(self) -> Dict[str, dict]:
        return {
            h: {
                "rank": v["rank"], "algorithm": v["algorithm"], "role": v["role"],
                "status": "running" if v["proc"].returncode is None else f"exited:{v['proc'].returncode}",
            }
            for h, v in self.processes.items()
        }

    async def stop_all(self) -> None:
        await self.stop_training()
        async with self._lock:
            self.selected.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _stream(self, hostname: str, proc) -> None:
        """Read stdout/stderr, feed into log buffer, clean up on exit."""
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip()
                if line:
                    self._log(hostname, line)
        except Exception as e:
            logger.debug(f"[node_manager] stream {hostname}: {e}")
        finally:
            await proc.wait()
            async with self._lock:
                self.processes.pop(hostname, None)
            logger.info(f"[node_manager] {hostname} exited (rc={proc.returncode})")
