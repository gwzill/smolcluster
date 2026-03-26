"""
Zero-config cluster node discovery via mDNS/Bonjour (zeroconf).

Usage on the dashboard / server node:
    discovery = NodeDiscovery()          # start browsing
    zc = register_node(8080, "server")   # advertise this machine

Usage on worker nodes (optional):
    zc = register_node(0, "available")   # just make this node discoverable
"""

import logging
import platform
import socket
import threading
from typing import Callable, Dict, Optional

from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
from zeroconf._exceptions import NonUniqueNameException

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_smolcluster._tcp.local."

# Additional service types to passively discover — any Mac with Remote Login
# (SSH) enabled advertises _ssh._tcp.local. without running smolcluster.
_PASSIVE_SERVICE_TYPES = [
    "_ssh._tcp.local.",          # macOS Remote Login, Raspberry Pi
    "_sftp-ssh._tcp.local.",     # SFTP (often co-advertised)
]


def _local_ip() -> str:
    """Best-effort: get the LAN IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def register_node(port: int, role: str = "available", hostname: Optional[str] = None) -> Zeroconf:
    """
    Advertise this machine on the local network via mDNS.

    Args:
        port:     Port the dashboard/worker listens on (use 0 if not applicable).
        role:     "server" | "worker" | "available"
        hostname: Override hostname (defaults to socket.gethostname()).

    Returns:
        Zeroconf instance — call .close() on shutdown to stop advertising.
    """
    if hostname is None:
        hostname = socket.gethostname()
    # Strip any existing .local suffix — zeroconf appends the service type itself
    hostname = hostname.removesuffix(".local")

    ip = _local_ip()
    zc = Zeroconf()

    info = ServiceInfo(
        SERVICE_TYPE,
        f"{hostname}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(ip)],
        port=max(port, 1),  # zeroconf requires port >= 1
        properties={
            "hostname": hostname,
            "role": role,
            "os": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(),
        },
    )
    try:
        zc.register_service(info)
        logger.info(f"[discovery] Registered {hostname} ({ip}) as role={role}")
    except NonUniqueNameException:
        # A stale mDNS record from a previous run is still live (TTL ~75s).
        # Safe to ignore — the old record will expire on its own and the browse
        # side works regardless.
        logger.warning(
            f"[discovery] Service {hostname} already registered on the network "
            f"(stale record from a previous run). Restart in ~75s to re-register."
        )
    return zc


class NodeDiscovery:
    """
    Continuously discovers smolcluster peers on the local network.

    Attributes:
        nodes: Dict[hostname, {ip, port, os, os_version, machine, role, hostname}]
    """

    def __init__(self, on_change: Optional[Callable] = None):
        """
        Args:
            on_change: Optional callback fired whenever the node list changes.
        """
        self.nodes: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._on_change = on_change
        self._zc = Zeroconf()
        # Browse for smolcluster nodes AND any SSH-advertising machine (no setup needed)
        all_types = [SERVICE_TYPE] + _PASSIVE_SERVICE_TYPES
        self._browser = ServiceBrowser(self._zc, all_types, self)

    # ── zeroconf listener interface ────────────────────────────────────────────

    def add_service(self, zc: Zeroconf, svc_type: str, name: str) -> None:
        try:
            info = zc.get_service_info(svc_type, name)
        except Exception:
            return
        if not info:
            return
        props = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in (info.properties or {}).items()
        }

        # Derive a clean short hostname suitable for SSH and display.
        if svc_type == SERVICE_TYPE:
            # smolcluster service: we embedded hostname in props explicitly
            hostname = props.get("hostname", "")
        elif info.server:
            # SSH/SFTP: info.server is the SRV target, e.g. "macmini2.local."
            hostname = info.server.rstrip(".").removesuffix(".local")
        else:
            raw_name = name.replace(f".{svc_type}", "").rstrip(".")
            hostname = props.get("hostname") or raw_name.split(".")[0]

        hostname = hostname.removesuffix(".local")
        if not hostname:
            return

        # Pick the best IP: prefer cluster/LAN (non-link-local) over 169.254.x.x
        ip = ""
        for addr in info.addresses:
            candidate = socket.inet_ntoa(addr)
            if not candidate.startswith("169.254."):
                ip = candidate
                break
        if not ip and info.addresses:
            ip = socket.inet_ntoa(info.addresses[0])

        with self._lock:
            existing = self.nodes.get(hostname, {})
            # smolcluster registration takes priority over passive SSH discovery
            if existing.get("role") in ("server", "worker") and svc_type != SERVICE_TYPE:
                return
            # Don't downgrade a good LAN IP to a link-local one
            existing_ip = existing.get("ip", "")
            if existing_ip and not existing_ip.startswith("169.254.") and ip.startswith("169.254."):
                ip = existing_ip
            node = {
                "hostname": hostname,
                "ip": ip,
                "port": info.port,
                "os": props.get("os", existing.get("os", _guess_os(props))),
                "os_version": props.get("os_version", existing.get("os_version", "")),
                "machine": props.get("machine", existing.get("machine", "")),
                "role": props.get("role", existing.get("role", "available")),
            }
            self.nodes[hostname] = node

        logger.info(f"[discovery] Found node: {hostname} ({ip}) via {svc_type}")
        if self._on_change:
            self._on_change()

    def remove_service(self, zc: Zeroconf, svc_type: str, name: str) -> None:
        hostname = name.split(".")[0]
        with self._lock:
            self.nodes.pop(hostname, None)
        logger.info(f"[discovery] Lost node: {hostname}")
        if self._on_change:
            self._on_change()

    def update_service(self, zc: Zeroconf, svc_type: str, name: str) -> None:
        self.add_service(zc, svc_type, name)

    # ── public helpers ─────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self.nodes)

    def close(self) -> None:
        self._zc.close()


def _guess_os(props: dict) -> str:
    """Guess OS from SSH TXT record properties (best effort)."""
    # macOS SSH TXT records sometimes include platform hints
    for v in props.values():
        if isinstance(v, str):
            v_lo = v.lower()
            if "darwin" in v_lo or "mac" in v_lo:
                return "Darwin"
            if "linux" in v_lo:
                return "Linux"
    return "unknown"
