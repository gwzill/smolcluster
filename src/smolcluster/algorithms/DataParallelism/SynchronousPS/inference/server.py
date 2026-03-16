import logging
import os
import socket
from pathlib import Path
from typing import Any, Optional, cast

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from smolcluster.utils.common_utils import receive_message, send_message
from smolcluster.utils.decoding import sample_next_token
from smolcluster.utils.device import get_device

CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs"

with open(CONFIG_DIR / "inference" / "model_config_inference.yaml") as f:
    inference_config = yaml.safe_load(f)

with open(CONFIG_DIR / "inference" / "cluster_config_inference.yaml") as f:
    cluster_config = yaml.safe_load(f)

model_configs = inference_config.get("dp", inference_config)
MODEL_NAME = model_configs.get("active_model", "hf_model")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or None


def resolve_model_config(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "hf_model_name" in cfg:
        return "hf_model", cfg

    model_name = cfg.get("active_model")
    if model_name and model_name in cfg:
        return model_name, cfg[model_name]

    raise ValueError("No valid model config found. Expected entry with 'hf_model_name'.")


MODEL_NAME, MODEL_CFG = resolve_model_config(model_configs)

HOST_IP = "0.0.0.0"
SERVER_HOSTNAME = cluster_config["server"]
port_config = cluster_config["port"]
if isinstance(port_config, dict):
    SERVER_PORT = port_config.get(SERVER_HOSTNAME, port_config.get("default", 65432))
else:
    SERVER_PORT = int(port_config)

NUM_WORKERS = int(cluster_config["num_workers"])

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("[SYNCPS-INF-SERVER]")


def get_defaults(payload: dict) -> tuple[int, str, float, float, int]:
    strategies = MODEL_CFG.get("decoding_strategies", {})
    active = payload.get("decoding_strategy") or MODEL_CFG.get(
        "active_decoding_strategy", "greedy"
    )
    active_cfg = strategies.get(active, {})

    max_tokens = payload.get("max_tokens") or MODEL_CFG.get("max_new_tokens", 128)
    temperature = payload.get("temperature")
    if temperature is None:
        temperature = active_cfg.get("temperature", 1.0)

    top_p = payload.get("top_p")
    if top_p is None:
        top_p = active_cfg.get("p", 0.9)

    top_k = payload.get("top_k")
    if top_k is None:
        top_k = active_cfg.get("k", 50)

    return max_tokens, active, temperature, top_p, top_k


def load_model_and_tokenizer() -> tuple[Any, Any]:
    device = get_device()
    hf_model_name = MODEL_CFG["hf_model_name"]
    tokenizer_cfg = MODEL_CFG.get("tokenizer", {})
    logger.info(f"Loading model '{MODEL_NAME}' from {hf_model_name} on {device}")
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, token=HF_TOKEN)  # type: ignore[call-arg]
    model = cast(Any, model).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, token=HF_TOKEN, **tokenizer_cfg)
    return model, tokenizer


def main() -> None:
    model, tokenizer = load_model_and_tokenizer()
    device = get_device()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST_IP, SERVER_PORT))
    server_socket.listen(8)

    logger.info(f"SyncPS inference server listening on {HOST_IP}:{SERVER_PORT}")

    worker_sockets: dict[int, socket.socket] = {}
    client_socket = None

    while len(worker_sockets) < NUM_WORKERS or client_socket is None:
        conn, addr = server_socket.accept()
        logger.info(f"Accepted connection from {addr}")

        message = receive_message(conn)
        if message is None:
            conn.close()
            continue

        command, payload = message
        if command == "register":
            rank = int(payload)
            worker_sockets[rank] = conn
            logger.info(f"Registered worker rank {rank}")
        elif command == "register_client":
            client_socket = conn
            send_message(client_socket, ("client_registered", None))
            logger.info("Registered chat backend client")
        else:
            logger.warning(f"Unexpected registration command: {command}")
            conn.close()

    for rank, worker_socket in sorted(worker_sockets.items()):
        logger.info(f"Signaling worker {rank} to start inference")
        send_message(worker_socket, "start_inference")

    logger.info("Ready to process inference requests")

    while True:
        request = receive_message(client_socket)
        if request is None:
            logger.warning("Client disconnected")
            break

        command, payload = request
        if command == "disconnect":
            logger.info("Client requested disconnect")
            break
        if command != "inference":
            send_message(client_socket, ("error", {"message": f"Unknown command: {command}"}))
            continue

        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            send_message(client_socket, ("error", {"message": "Empty prompt"}))
            continue

        max_tokens, decoding_strategy, temperature, top_p, top_k = get_defaults(payload)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids
        original_prompt_length = tokenized_prompt.shape[1]

        for token_idx in range(max_tokens):
            with torch.no_grad():
                server_logits = model(tokenized_prompt.to(device)).logits.cpu()

            all_logits = [server_logits]
            for rank, worker_socket in sorted(worker_sockets.items()):
                send_message(
                    worker_socket,
                    (
                        "generate_activations",
                        {
                            "activations": None,
                            "input_ids": tokenized_prompt.cpu(),
                            "max_new_tokens": 1,
                            "decoding_strategy": decoding_strategy,
                        },
                    ),
                )
                response = receive_message(worker_socket)
                if response is None:
                    raise RuntimeError(f"Worker {rank} disconnected during inference")
                resp_command, resp_payload = response
                if resp_command != "forward_activations":
                    raise RuntimeError(
                        f"Unexpected response from worker {rank}: {resp_command}"
                    )
                all_logits.append(resp_payload["activations"])

            averaged_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            tokenized_prompt, should_stop = sample_next_token(
                averaged_logits,
                tokenized_prompt,
                temperature,
                tokenizer,
                decoding_strategy=decoding_strategy,
                top_p=top_p,
                top_k=top_k,
            )

            new_token_id = tokenized_prompt[0, -1].item()
            new_token_text = tokenizer.decode([new_token_id], skip_special_tokens=True)
            send_message(
                client_socket,
                ("token", {"text": new_token_text, "token_idx": token_idx}),
            )

            if should_stop:
                break

        generated_tokens = tokenized_prompt[0, original_prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        send_message(client_socket, ("inference_complete", {"text": generated_text}))

    for rank, worker_socket in sorted(worker_sockets.items()):
        try:
            send_message(worker_socket, "down")
            worker_socket.close()
            logger.info(f"Closed worker connection {rank}")
        except Exception:
            pass

    try:
        client_socket.close()
    except Exception:
        pass

    server_socket.close()


if __name__ == "__main__":
    main()
