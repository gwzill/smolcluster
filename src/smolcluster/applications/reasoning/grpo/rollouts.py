import threading
from typing import Any, Dict, List, Optional

import requests
import yaml

lock = threading.Lock()

with open("configs/inference/reasoning/grpo/config.yaml") as f:
    grpo_config = yaml.safe_load(f)

API_URL = grpo_config.get("api_url")
NUM_ROLLOUTS = grpo_config["num_rollouts"]

def query(url: str, payload: Dict) -> Dict:
    """Query the inference API."""
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def build_vllm_worker_urls(config: Dict[str, Any]) -> Dict[int, str]:
    """Build OpenAI-compatible vLLM completion URLs for each configured worker rank."""
    vllm_cluster = config["vllm_cluster"]
    host_ip = vllm_cluster["host_ip"]
    port = vllm_cluster.get("port", 8000)
    completion_path = vllm_cluster.get("completion_path", "/v1/completions")

    worker_urls: Dict[int, str] = {}
    for worker in vllm_cluster["workers"]["regular"][: vllm_cluster["num_workers"]]:
        hostname = worker["hostname"]
        rank = worker["rank"]
        worker_urls[rank] = f"http://{host_ip[hostname]}:{port}{completion_path}"

    return worker_urls


def handle_worker_rollout(
    url: str,
    payload: Dict,
    worker_rank: int,
    rollout_idx: int,
    rollouts: Dict,
    lock: threading.Lock,
) -> None:
    """Generate a single rollout for a worker and store result."""
    response = query(url, payload)
    
    with lock:
        if worker_rank not in rollouts:
            rollouts[worker_rank] = [None] * NUM_ROLLOUTS
        rollouts[worker_rank][rollout_idx] = response["generated_text"]


def generate_rollouts_for_prompt(
    prompt: str,
    num_workers: int,
    decoding_strategy: str,
    max_tokens: int,
) -> Dict[int, list[str]]:
    """Generate NUM_ROLLOUTS outputs from each worker for a single prompt."""
    if API_URL is None:
        raise ValueError("api_url must be set in the GRPO config to use generate_rollouts")

    threads = []
    rollouts = {}
    
    for worker_rank in range(num_workers):
        for rollout_idx in range(NUM_ROLLOUTS):
            
            payload = {
                "text": prompt,
                "worker_rank": worker_rank,
                "max_tokens": max_tokens,
                "decoding_strategy": decoding_strategy,
            }
            
            thread = threading.Thread(
                target=handle_worker_rollout,
                args=(API_URL, payload, worker_rank, rollout_idx, rollouts, lock),
            )
            thread.start()
            threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return rollouts


def generate_rollouts(
    prompt: str,
    num_workers: int,
    decoding_strategy: str = "top_p",
    max_tokens: int = 256,
) -> Dict[int, list[str]]:
    """
    Generate NUM_ROLLOUTS outputs from each worker for a prompt.
    
    Returns:
        Dict mapping worker_rank -> list of NUM_ROLLOUTS generated texts
    """
    return generate_rollouts_for_prompt(
        prompt, num_workers, decoding_strategy, max_tokens
    )


def handle_vllm_rollout(
    vllm_url: str,
    worker_rank: int,
    prompt: str,
    rollout_idx: int,
    rollouts: Dict[int, List[str]],
    lock: threading.Lock,
    max_tokens: int,
) -> None:
    """Generate a single rollout from vLLM and store result."""
    try:
        # Build OpenAI-compatible request
        api_params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
     
        
        response = requests.post(vllm_url, json=api_params, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        generated_text = result.get("choices", [{}])[0].get("text", "")
        
        with lock:
            rollouts[worker_rank][rollout_idx] = generated_text
    except Exception as e:
        print(f"Error generating rollout {rollout_idx} from vLLM: {e}")
        with lock:
            rollouts[worker_rank][rollout_idx] = ""


def generate_rollouts_vllm(
    prompt: str,
    max_tokens: int = 256,
    num_rollouts: Optional[int] = None,
    worker_urls: Optional[Dict[int, str]] = None,
) -> Dict[int, list[str]]:
    """
    Generate rollouts using worker-node vLLM OpenAI-compatible completion endpoints.
    
    Args:
        prompt: The prompt to complete
        decoding_strategy: "top_p", "temperature", or "greedy"
        max_tokens: Max tokens to generate
        num_rollouts: Number of rollouts to generate (uses grpo_config value if None)
        worker_urls: Optional explicit mapping of worker rank to completion URL
    
    Returns:
        Dict mapping worker rank -> list of generated texts
    """
    effective_num_rollouts: int = int(NUM_ROLLOUTS if num_rollouts is None else num_rollouts)

    if worker_urls is None:
        worker_urls = build_vllm_worker_urls(grpo_config)
    
    threads = []
    rollouts: Dict[int, List[str]] = {
        worker_rank: [""] * effective_num_rollouts for worker_rank in worker_urls
    }
    vllm_lock = threading.Lock()

    for worker_rank, worker_url in worker_urls.items():
        for rollout_idx in range(effective_num_rollouts):
            thread = threading.Thread(
                target=handle_vllm_rollout,
                args=(
                    worker_url,
                    worker_rank,
                    prompt,
                    rollout_idx,
                    rollouts,
                    vllm_lock,
                    max_tokens,
                ),
            )
            thread.start()
            threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return rollouts