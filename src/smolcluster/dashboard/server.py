"""
smolcluster Dashboard — FastAPI backend.

Node states:
    discovered  — seen via mDNS (SSH broadcast)
    selected    — user clicked "Add to Cluster" (no SSH yet)
    running     — training/inference started, SSH process alive

Endpoints:
    GET  /                                → index.html
    GET  /api/nodes                       → discovered + selected + running
    GET  /api/events                      → SSE 1s stream
    GET  /api/nodes/{hostname}/probe      → SSH whoami (get real username)
    POST /api/nodes/{hostname}/select     → mark node as ready for training
    POST /api/nodes/{hostname}/deselect   → unmark
    POST /api/training/start              → launch server + all selected workers
    POST /api/training/stop               → kill everything
    POST /api/inference/start             → connectivity test (ping all workers)
    POST /api/inference/stop              → kill inference processes
    GET  /api/training                    → latest metrics from shared file
"""

import asyncio
import json
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from smolcluster.dashboard.node_manager import NodeManager
from smolcluster.utils.discovery import NodeDiscovery, register_node

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent / "frontend"
METRICS_FILE   = Path("/tmp/smolcluster_metrics.json")
INFERENCE_FILE = Path("/tmp/smolcluster_inference.json")

# ── App state ─────────────────────────────────────────────────────────────────
discovery: NodeDiscovery
node_manager: NodeManager
_zc = None
_server_hostname: str = ""
_loop: asyncio.AbstractEventLoop = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global discovery, node_manager, _zc, _server_hostname, _loop
    _loop = asyncio.get_running_loop()
    discovery = NodeDiscovery(on_change=_on_node_change)
    node_manager = NodeManager()
    _server_hostname = socket.gethostname().removesuffix(".local")
    _zc = await asyncio.to_thread(register_node, 8080, "server", _server_hostname)
    logger.info(f"[dashboard] http://{_server_hostname}.local:8080")
    yield
    await node_manager.stop_all()
    discovery.close()
    if _zc:
        await asyncio.to_thread(_zc.close)


app = FastAPI(title="smolcluster Dashboard", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Auto-probe usernames when nodes appear ────────────────────────────────────
_probed: dict = {}   # hostname → username or None

def _on_node_change():
    """Kick off a username probe for any newly discovered node.
    Called from zeroconf's background thread — must not use create_task directly."""
    for hostname, info in discovery.snapshot().items():
        if hostname not in _probed:
            _probed[hostname] = None   # mark as in-flight
            if _loop and _loop.is_running():
                asyncio.run_coroutine_threadsafe(_probe_and_store(hostname, info), _loop)

async def _probe_and_store(hostname: str, info: dict):
    # Try the guessed user first, then no user (SSH default)
    guess = _guess_user(hostname)
    for attempt in ([guess] if guess else []) + [""]:
        user = await NodeManager.probe_username(hostname, attempt)
        if user:
            _probed[hostname] = user
            logger.info(f"[dashboard] {hostname} username → {user}")
            return
    _probed[hostname] = ""   # unreachable or no key — leave blank


def _guess_user(hostname: str) -> str:
    import re
    m = re.search(r'macmini(\d+)', hostname, re.IGNORECASE)
    if m:
        return f"yuvrajsingh{m[1]}"
    return ""


# ── Request models ────────────────────────────────────────────────────────────
class SelectRequest(BaseModel):
    ssh_user: str = ""
    rank: Optional[int] = None

class StartRequest(BaseModel):
    algorithm: str = "syncps"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/nodes")
async def get_nodes():
    return {
        "discovered": discovery.snapshot(),
        "selected":   node_manager.snapshot_selected(),
        "running":    node_manager.snapshot_processes(),
        "usernames":  dict(_probed),
    }


@app.get("/api/nodes/{hostname}/probe")
async def probe_node(hostname: str, ssh_user: str = ""):
    user = await NodeManager.probe_username(hostname, ssh_user)
    if user is None:
        raise HTTPException(status_code=502, detail="Could not reach node or SSH key not set up")
    _probed[hostname] = user
    return {"username": user}


@app.post("/api/nodes/{hostname}/select")
async def select_node(hostname: str, req: SelectRequest):
    rank = await node_manager.select(hostname, req.ssh_user, req.rank)
    return {"status": "selected", "rank": rank}


@app.post("/api/nodes/{hostname}/deselect")
async def deselect_node(hostname: str):
    await node_manager.deselect(hostname)
    return {"status": "deselected"}


@app.post("/api/training/start")
async def start_training(req: StartRequest):
    if not node_manager.selected:
        raise HTTPException(status_code=400, detail="No nodes selected")
    try:
        await node_manager.start_training(req.algorithm, _server_hostname)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "started"}


@app.post("/api/training/stop")
async def stop_training():
    await node_manager.stop_training()
    return {"status": "stopped"}


@app.post("/api/inference/start")
async def start_inference():
    if not node_manager.selected:
        raise HTTPException(status_code=400, detail="No nodes selected")
    try:
        await node_manager.start_inference(_server_hostname)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "started"}


@app.post("/api/inference/stop")
async def stop_inference():
    await node_manager.stop_training()  # same kill mechanism
    INFERENCE_FILE.unlink(missing_ok=True)
    return {"status": "stopped"}


@app.get("/api/training")
async def get_training():
    return _read_metrics()


@app.get("/api/events")
async def sse_events():
    async def generator():
        while True:
            payload = json.dumps({
                "nodes": {
                    "discovered": discovery.snapshot(),
                    "selected":   node_manager.snapshot_selected(),
                    "running":    node_manager.snapshot_processes(),
                    "usernames":  dict(_probed),
                },
                "training":  _read_metrics(),
                "inference": _read_inference(),
            })
            yield f"data: {payload}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}
    try:
        return json.loads(METRICS_FILE.read_text())
    except Exception:
        return {}


def _read_inference() -> dict:
    if not INFERENCE_FILE.exists():
        return {}
    try:
        return json.loads(INFERENCE_FILE.read_text())
    except Exception:
        return {}
