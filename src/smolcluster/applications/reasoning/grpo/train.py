import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from smolcluster.applications.reasoning.grpo.data.grpo_data import build_train_val_dataloaders
from smolcluster.applications.reasoning.grpo.rewards import (
    calculate_answer_reward,
    calculate_formatted_reward,
)
from smolcluster.applications.reasoning.grpo.rollouts import generate_rollouts


with open("configs/inference/reasoning/grpo/config.yaml") as f:
    grpo_config = yaml.safe_load(f)

with open("configs/inference/model_config_inference.yaml") as f:
    model_config = yaml.safe_load(f)


PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process "
    "in the mind and then provides the user with the answer. The reasoning process "
    "and answer are enclosed within <think>...</think> and '### ''"
    "tags, respectively, i.e., <think> reasoning process here </think>"
    "### answer here. User: {question}. Assistant:"
)


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    rewards_std = rewards.std()
    return (rewards - rewards.mean()) / rewards_std


def _sequence_logprobs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()

    token_logprobs = torch.log_softmax(shift_logits, dim=-1)
    gathered = token_logprobs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    masked_logprobs = gathered * shift_mask
    token_counts = shift_mask.sum(dim=1).clamp(min=1.0)
    return masked_logprobs.sum(dim=1) / token_counts


def compute_logprobs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _sequence_logprobs_from_logits(outputs.logits, input_ids, attention_mask)


def parse_numeric_answer(text: str) -> float:
    number_matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not number_matches:
        return float("nan")
    return float(number_matches[-1])


def compute_grpo_loss(
    ref_logprobs: torch.Tensor,
    curr_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    config: Dict[str, Any],
) -> torch.Tensor:
    logprobs_ratio = torch.exp(curr_logprobs - ref_logprobs)
    clipped_ratio = torch.clip(
        logprobs_ratio,
        1 - config["clip_ratio"],
        1 + config["clip_ratio"],
    )
    kl = (ref_logprobs - curr_logprobs) - torch.exp(ref_logprobs - curr_logprobs) - 1
    return -torch.mean(
        torch.min(logprobs_ratio * advantages, clipped_ratio * advantages)
        - config["kl_beta"] * kl
    )


def organize_rollouts(rollouts: Dict[int, List[str]]) -> List[str]:
    organized: List[str] = []
    for _worker_rank, generated_texts in sorted(rollouts.items()):
        for generated_text in generated_texts:
            if generated_text is not None:
                organized.append(generated_text)
    return organized


def build_batched_rollout_texts(
    questions: List[str],
    true_answers: List[str],
    config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    rollout_texts: List[str] = []
    rollout_targets: List[str] = []

    for question, true_answer in zip(questions, true_answers):
        prompt = PROMPT.format(question=question)
        worker_rollouts = generate_rollouts(
            prompt,
            config["num_workers"],
            config["decoding_strategy"],
            config["max_tokens"],
        )
        prompt_rollouts = organize_rollouts(worker_rollouts)
        rollout_texts.extend(prompt_rollouts)
        rollout_targets.extend([true_answer] * len(prompt_rollouts))

    return rollout_texts, rollout_targets


def tokenize_rollouts(
    tokenizer: Any,
    rollout_texts: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenizer(
        rollout_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)


def compute_rewards(
    rollout_texts: List[str],
    rollout_targets: List[str],
    device: torch.device,
) -> torch.Tensor:
    reward_values: List[float] = []

    for generated_text, target_answer in zip(rollout_texts, rollout_targets):
        predicted_numeric_answer = parse_numeric_answer(generated_text)
        target_numeric_answer = parse_numeric_answer(str(target_answer))
        answer_reward = calculate_answer_reward(predicted_numeric_answer, target_numeric_answer)
        format_reward = calculate_formatted_reward(generated_text)
        reward_values.append(float(answer_reward + format_reward))

    return torch.tensor(reward_values, dtype=torch.float32, device=device)


def train_step(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    questions: List[str],
    true_answers: List[str],
) -> None:
    rollout_texts, rollout_targets = build_batched_rollout_texts(questions, true_answers, config)
    if not rollout_texts:
        return

    device = next(model.parameters()).device
    input_ids, attention_mask = tokenize_rollouts(tokenizer, rollout_texts, device)

    curr_logprobs = compute_logprobs(model, input_ids, attention_mask)
    with torch.no_grad():
        ref_logprobs = compute_logprobs(ref_model, input_ids, attention_mask)

    rewards = compute_rewards(rollout_texts, rollout_targets, device)
    advantages = compute_advantages(rewards)
    loss = compute_grpo_loss(ref_logprobs, curr_logprobs, advantages, config)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(
    model: Any,
    ref_model: Any,
    config: Dict[str, Any],
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
) -> None:
    model.train()

    for _epoch in range(config["num_epochs"]):
        for questions, true_answers in train_dataloader:
            train_step(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                config=config,
                questions=list(questions),
                true_answers=list(true_answers),
            )


def main() -> None:
    train_dataloader, _val_dataloader = build_train_val_dataloaders(grpo_config["batch_size"])

    model = AutoModelForCausalLM.from_pretrained(model_config["hf_model_name"])
    ref_model = deepcopy(model)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=grpo_config["learning_rate"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["hf_model_name"])


    train(model, ref_model, grpo_config, train_dataloader, optimizer, tokenizer)


if __name__ == "__main__":
    main()
