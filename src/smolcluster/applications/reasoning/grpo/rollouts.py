import requests
from typing import Dict
import threading
import yaml

lock = threading.Lock()

with open("configs/inference/reasoning/grpo/config.yaml") as f:
    grpo_config = yaml.safe_load(f)

API_URL = grpo_config["api_url"]
NUM_ROLLOUTS = grpo_config["num_rollouts"]

def query(url: str, payload: Dict) -> Dict:
    """Query the inference API."""
    response = requests.post(url, json=payload)
    return response.json()


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