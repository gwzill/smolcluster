"""GRPO rollout generation — queries vLLM workers to sample candidate responses for each prompt."""
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml

from smolcluster.utils import emit_smol_event

logger = logging.getLogger(__name__)

lock = threading.Lock()
debug_lock = threading.Lock()

# Resolve config paths relative to project root
_module_dir = Path(__file__).parent
_smolcluster_root = _module_dir.parents[3]  # Navigate from utils -> grpo -> reasoning -> applications -> smolcluster
_train_target = os.environ.get("GRPO_TRAIN_TARGET", "gsm8k")
_config_path = _smolcluster_root / "configs" / "reasoning" / "grpo" / f"config_{_train_target}.yaml"
_cluster_config_path = _smolcluster_root / "configs" / "inference" / "cluster_config_inference.yaml"
_project_root = _smolcluster_root.parent.parent
_debug_dir = _project_root / ".grpo_debug"
_debug_dir.mkdir(parents=True, exist_ok=True)
_debug_log_path = _debug_dir / "vllm_rollouts.jsonl"

with open(_config_path) as f:
    grpo_config = yaml.safe_load(f)

with open(_cluster_config_path) as f:
    cluster_config = yaml.safe_load(f)

API_URL = grpo_config.get("api_url")
NUM_ROLLOUTS = grpo_config["num_rollouts"]


def _get_debug_log_path() -> Path:
    override = os.getenv("SMOLCLUSTER_VLLM_DEBUG_LOG_PATH")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = _project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    _debug_dir.mkdir(parents=True, exist_ok=True)
    return _debug_log_path


def append_vllm_debug_log(record: dict[str, Any]) -> None:
    debug_log_path = _get_debug_log_path()
    with debug_lock:
        with debug_log_path.open("a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")


def build_vllm_worker_urls(config: dict[str, Any]) -> dict[int, str]:
    """Build OpenAI-compatible vLLM completion URLs for each configured worker rank.
    Workers and host_ip are read from cluster_config_inference.yaml."""
    vllm_cluster = config["vllm_cluster"]
    port = vllm_cluster.get("port", 8000)
    completion_path = vllm_cluster.get("completion_path", "/v1/completions")

    host_ip = cluster_config["host_ip"]
    num_workers = cluster_config["num_workers"]
    workers = cluster_config["workers"]["regular"][:num_workers]

    worker_urls: dict[int, str] = {}
    for worker in workers:
        hostname = worker["hostname"]
        rank = worker["rank"]
        worker_urls[rank] = f"http://{host_ip[hostname]}:{port}{completion_path}"

    return worker_urls


def _fetch_n_from_worker(
    vllm_url: str,
    worker_rank: int,
    prompt: str,
    n: int,
    rollouts: dict[int, list[str]],
    lock: threading.Lock,
    max_tokens: int,
    step: int | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> None:
    """
    Send ONE request to a vLLM worker asking for n completions.
    vLLM shares the prompt KV cache across all n continuations, which is
    significantly more efficient than n separate single-completion requests.
    """
    api_params = {"prompt": prompt, "max_tokens": max_tokens, "n": n}
    if sampling_params:
        api_params.update(sampling_params)
    logger.info("[vllm worker %d] Requesting n=%d completions from %s", worker_rank, n, vllm_url)
    emit_smol_event("rollout", "out", "grpo")

    try:
        response = requests.post(vllm_url, json=api_params, timeout=300)
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices") or []

        non_empty = [c.get("text", "") for c in choices if c and c.get("text", "").strip()]

        append_vllm_debug_log({
            "step": step,
            "worker_rank": worker_rank,
            "url": vllm_url,
            "n": n,
            "max_tokens": max_tokens,
            "sampling_params": sampling_params or {},
            "prompt_preview": prompt,
            "status_code": response.status_code,
            "num_choices": len(choices),
            "num_non_empty": len(non_empty),
            "texts_preview": dict(enumerate(non_empty)),
        })

        if non_empty:
            emit_smol_event("rollout", "in", "grpo")
            model_served = result.get("model", "")
            expected_model = api_params.get("model", "")
            if expected_model == "adapter":
                if model_served == "adapter":
                    logger.info("[vllm worker %d] LoRA adapter confirmed in use (model=%r)", worker_rank, model_served)
                else:
                    logger.warning(
                        "[vllm worker %d] LoRA mismatch: requested model=%r but served model=%r — "
                        "adapter may not be loaded yet",
                        worker_rank, expected_model, model_served,
                    )
            logger.info("[vllm worker %d] Got %d/%d non-empty completion(s)", worker_rank, len(non_empty), n)
            for idx, text in enumerate(non_empty):
                logger.info(
                    "[vllm worker %d | sample %d] %.120s",
                    worker_rank, idx + 1, text.replace("\n", " "),
                )
            with lock:
                rollouts[worker_rank] = non_empty
            return

        emit_smol_event("rollout", "in", "grpo")
        logger.warning("[vllm worker %d] All %d completions empty", worker_rank, n)
    except Exception as e:
        emit_smol_event("rollout", "in", "grpo")
        append_vllm_debug_log(
            {
                "step": step,
                "worker_rank": worker_rank,
                "url": vllm_url,
                "sampling_params": sampling_params or {},
                "error": str(e),
            }
        )
        logger.error("[vllm worker %d] Request failed: %s", worker_rank, e)

    with lock:
        rollouts[worker_rank] = []


def generate_rollouts_vllm(
    prompt: str,
    max_tokens: int,
    num_rollouts: int | None = None,
    step: int | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> dict[int, list[str]]:
    """
    Generate rollouts using worker-node vLLM OpenAI-compatible completion endpoints.

    Sends ONE request per worker with n=num_rollouts, so vLLM shares the prompt
    KV cache across all completions (much more efficient than N separate requests).

    Returns:
        Dict mapping worker rank -> list of generated text strings.
    """
    effective_n: int = int(NUM_ROLLOUTS if num_rollouts is None else num_rollouts)

    _worker_urls = build_vllm_worker_urls(grpo_config)
    logger.info(
        "[generate_rollouts_vllm] Requesting n=%d completions from each of %d worker(s): %s",
        effective_n, len(_worker_urls), list(_worker_urls.keys()),
    )
    logger.info("[generate_rollouts_vllm] Prompt (first 120 chars): %.120s", prompt.replace('\n', ' '))

    rollouts: dict[int, list[str]] = {}
    vllm_lock = threading.Lock()

    # One thread per worker — each sends a single request with n completions.
    threads = [
        threading.Thread(
            target=_fetch_n_from_worker,
            args=(url, rank, prompt, effective_n, rollouts, vllm_lock, max_tokens, step, sampling_params),
        )
        for rank, url in _worker_urls.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = sum(len(items) for items in rollouts.values())
    logger.info("[generate_rollouts_vllm] All workers done. %d non-empty rollout(s) collected", total)
    return rollouts


# ---------------------------------------------------------------------------
# vLLM TTFT metrics — query /metrics endpoint, return p50 TTFT in ms
# ---------------------------------------------------------------------------

def _p50_from_prometheus_histogram(text: str, metric_name: str) -> float | None:
    """Parse a Prometheus-format cumulative histogram and return the p50 value."""
    bucket_re = re.compile(
        rf'^{re.escape(metric_name)}_bucket\{{[^}}]*\ble="([^"]+)"[^}}]*\}}\s+([\d.eE+\-]+)',
        re.MULTILINE,
    )
    count_re = re.compile(
        rf'^{re.escape(metric_name)}_count(?:\{{[^}}]*\}})?\s+([\d.eE+\-]+)',
        re.MULTILINE,
    )
    buckets: list[tuple[float, float]] = []
    for m in bucket_re.finditer(text):
        le_str, val_str = m.group(1), m.group(2)
        if le_str == "+Inf":
            continue
        try:
            buckets.append((float(le_str), float(val_str)))
        except ValueError:
            continue
    cm = count_re.search(text)
    if not cm or not buckets:
        return None
    total = float(cm.group(1))
    if total < 1:
        return None
    target = total * 0.5
    buckets.sort(key=lambda x: x[0])
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if count <= prev_count or le <= prev_le:
                return le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return buckets[-1][0] if buckets else None


def _median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    return (values[n // 2 - 1] + values[n // 2]) / 2 if n % 2 == 0 else values[n // 2]


def fetch_vllm_perf_metrics(worker_urls: dict[int, str]) -> dict[str, float]:
    """Query vLLM /metrics once per worker, return median p50 TTFT and TPOT in ms.

    Uses vllm:time_to_first_token_seconds for tok/sec in (prefill throughput)
    and vllm:inter_token_latency_seconds for tok/sec out (decode throughput).
    Per-worker p50 is interpolated from cumulative histogram buckets; the
    median across all workers is returned so a single slow worker doesn't skew
    the display — and this naturally scales to any number of vLLM workers.

    Returns: {"ttft_p50_ms": float, "tpot_p50_ms": float}  (keys absent if unavailable)
    """
    ttfts_s: list[float] = []
    tpots_s: list[float] = []

    for rank, completion_url in worker_urls.items():
        metrics_url = completion_url.split("/v1/")[0] + "/metrics"
        try:
            r = requests.get(metrics_url, timeout=3)
            r.raise_for_status()
            text = r.text

            ttft = _p50_from_prometheus_histogram(text, "vllm:time_to_first_token_seconds")
            if ttft is not None and ttft > 0:
                ttfts_s.append(ttft)
                logger.debug("[vllm_metrics] worker %d TTFT p50 = %.1f ms", rank, ttft * 1000)

            tpot = _p50_from_prometheus_histogram(text, "vllm:inter_token_latency_seconds")
            if tpot is not None and tpot > 0:
                tpots_s.append(tpot)
                logger.debug("[vllm_metrics] worker %d TPOT p50 = %.1f ms  (%.1f tok/s)", rank, tpot * 1000, 1 / tpot)

        except Exception as exc:
            logger.debug("[vllm_metrics] worker %d metrics unavailable: %s", rank, exc)

    result: dict[str, float] = {}
    if ttfts_s:
        result["ttft_p50_ms"] = round(_median(ttfts_s) * 1000, 2)
    if tpots_s:
        result["tpot_p50_ms"] = round(_median(tpots_s) * 1000, 2)
    return result


# ---------------------------------------------------------------------------
# Rollout orchestration (prompt-level)
# ---------------------------------------------------------------------------

def organize_rollouts(
    rollouts: dict[int, list[str]],
) -> list[str]:
    """Flatten worker rollouts into an ordered list, dropping empty completions."""
    texts: list[str] = []
    for _rank, items in sorted(rollouts.items()):
        texts.extend(t for t in items if t and t.strip())
    return texts


def _fetch_for_prompt(
    idx: int,
    prompt: str,
    true_answer: str,
    config: dict[str, Any],
    step: int | None = None,
) -> tuple[int, list[str], str]:
    """Fetch vLLM rollouts for a single (pre-formatted) prompt. Called concurrently."""
    worker_rollouts = generate_rollouts_vllm(
        prompt,
        max_tokens=config["max_output_tokens"],
        num_rollouts=config["num_rollouts"],
        step=step,
        sampling_params=config.get("vllm_request_overrides"),
    )
    texts = organize_rollouts(worker_rollouts)
    return idx, texts, true_answer


def build_rollouts_per_prompt(
    prompts: list[str],
    true_answers: list[str],
    config: dict[str, Any],
    step: int | None = None,
) -> list[tuple[list[str], str]]:
    """Dispatch rollout generation for all prompts concurrently, return per-prompt.

    Returns:
        List of (rollout_texts, true_answer) — one entry per prompt, in input order.
        Prompts that produced zero usable rollouts have an empty rollout_texts list.
    """
    n = len(prompts)
    ordered: list[tuple[list[str], str] | None] = [None] * n

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {
            executor.submit(_fetch_for_prompt, idx, prompt, ans, config, step): idx
            for idx, (prompt, ans) in enumerate(zip(prompts, true_answers, strict=False))
        }
        for future in as_completed(futures):
            idx, prompt_rollouts, true_answer = future.result()
            logger.info(
                "[rollout %d/%d] Received %d usable rollout(s) from workers",
                idx + 1, n, len(prompt_rollouts),
            )
            for r_idx, text in enumerate(prompt_rollouts):
                logger.info(
                    "[rollout %d/%d | sample %d] %.200s",
                    idx + 1, n, r_idx + 1, text.replace("\n", " "),
                )
            ordered[idx] = (prompt_rollouts, true_answer)

    return [
        item if item is not None else ([], true_answers[i])
        for i, item in enumerate(ordered)
    ]


def build_batched_rollout_texts(
    prompts: list[str],
    true_answers: list[str],
    config: dict[str, Any],
    step: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Dispatch rollout generation for all prompts concurrently, collect in order.

    Returns:
        (rollout_texts, rollout_targets, rollout_questions) — one entry per usable
        rollout across all prompts.
    """
    per_prompt = build_rollouts_per_prompt(prompts, true_answers, config, step=step)
    rollout_texts: list[str] = []
    rollout_targets: list[str] = []
    rollout_questions: list[str] = []
    for (prompt_rollouts, true_answer), prompt in zip(per_prompt, prompts, strict=False):
        rollout_texts.extend(prompt_rollouts)
        rollout_targets.extend([true_answer] * len(prompt_rollouts))
        rollout_questions.extend([prompt] * len(prompt_rollouts))
    return rollout_texts, rollout_targets, rollout_questions
