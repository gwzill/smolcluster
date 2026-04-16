"""Centralized logging configuration for smolcluster."""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour palette
# ---------------------------------------------------------------------------

_RESET = "\033[0m"

_LEVEL_COLOURS = {
    logging.DEBUG:    "\033[36m",    # cyan
    logging.INFO:     "\033[32m",    # green
    logging.WARNING:  "\033[33m",    # yellow
    logging.ERROR:    "\033[31m",    # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}

_TAG_COLOUR = "\033[35m"   # magenta — for bracketed tags like [MODEL], [LORA]
_DIM = "\033[2m"


class ColourFormatter(logging.Formatter):
    """Single-line coloured formatter.

    Format:  HH:MM:SS  LEVEL     logger.name  message
    Bracketed tags like [MODEL], [checkpoint], [vllm worker 0] are highlighted.
    """

    _FMT = "{dim}{asctime}{reset}  {level_col}{levelname:<8}{reset}  {dim}{name}{reset}  {msg}"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, "%H:%M:%S")

        msg = re.sub(
            r"(\[[^\]]{1,40}\])",
            rf"{_TAG_COLOUR}\1{_RESET}",
            record.message,
        )

        line = self._FMT.format(
            dim=_DIM,
            asctime=record.asctime,
            reset=_RESET,
            level_col=_LEVEL_COLOURS.get(record.levelno, ""),
            levelname=record.levelname,
            name=record.name,
            msg=msg,
        )

        if record.exc_info:
            line = f"{line}\n{_LEVEL_COLOURS.get(logging.ERROR, '')}{self.formatException(record.exc_info)}{_RESET}"

        return line


def setup_logging(
    level: int = logging.INFO,
    *,
    force: bool = False,
) -> None:
    """Configure the root logger with a coloured console handler.

    Call once from the main entry-point of each script.
    Subsequent calls are no-ops unless ``force=True``.
    """
    root = logging.getLogger()

    if root.handlers and not force:
        return

    if force:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColourFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Cluster / Loki logging (file-based, structured for Promtail)
# ---------------------------------------------------------------------------

class RankFilter(logging.Filter):
    """Attach rank and component fields to every log record."""

    def __init__(self, rank: Optional[int] = None, component: str = "server"):
        super().__init__()
        self.rank = rank if rank is not None else -1
        self.component = component

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.component = self.component
        return True


def setup_cluster_logging(
    logger: logging.Logger,
    component: str,
    rank: Optional[int] = None,
    hostname: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Add structured file logging to an existing logger for Loki/Promtail ingestion."""

    def _project_log_dir() -> Path:
        return Path(__file__).resolve().parents[3] / "logging" / "cluster-logs"

    def _pick_writable(preferred: Optional[str]) -> Path:
        default = _project_log_dir()
        for candidate in [Path(preferred) if preferred else default, default, Path.cwd() / "smolcluster-logs"]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_probe"
                probe.open("a").close()
                probe.unlink(missing_ok=True)
                return candidate
            except OSError:
                continue
        raise OSError("No writable directory found for cluster logs")

    log_file = _pick_writable(log_dir) / (
        f"server-{hostname or 'unknown'}.log" if component == "server"
        else f"worker-rank{rank}-{hostname or 'unknown'}.log"
    )

    # Avoid duplicate handlers
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in logger.handlers):
        return

    logger.addFilter(RankFilter(rank=rank, component=component))

    try:
        fh = logging.FileHandler(log_file, mode="a")
    except PermissionError:
        fallback = _pick_writable(str(_project_log_dir().parent / "cluster-logs-fallback"))
        fh = logging.FileHandler(fallback / log_file.name, mode="a")

    prefix = "rank:server" if component == "server" else f"rank:{rank}"
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(f"%(asctime)s | %(levelname)s | {prefix} | %(message)s"))
    logger.addHandler(fh)
    logger.info("Logging initialised: %s (component=%s rank=%s hostname=%s)", log_file, component, rank, hostname)


def log_step(logger: logging.Logger, step: int, message: str, level: int = logging.INFO) -> None:
    logger.log(level, "step:%d | %s", step, message)


def log_metric(logger: logging.Logger, step: int, metric_name: str, value: float, extra_info: Optional[str] = None) -> None:
    msg = f"step:{step} | metric:{metric_name} | value:{value:.6f}"
    if extra_info:
        msg += f" | {extra_info}"
    logger.info(msg)


def emit_transport_event(phase: str, **fields) -> None:
    """Emit machine-readable transport events for dashboard particle animation.

    The dashboard listens for lines in the form:
            [TRANSPORT_EVENT] {"phase":"request"|"response", ...}
    """
    payload = {"phase": str(phase or "").strip().lower()}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            payload[k] = v
        else:
            payload[k] = str(v)
    print(f"[TRANSPORT_EVENT] {json.dumps(payload)}", flush=True)
