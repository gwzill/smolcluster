import re
from typing import List, Tuple

import torch
from datasets import load_dataset

import re
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process "
    "in the mind and then provides the user with the FINAL answer. The reasoning process "
    "is enclosed within <think>...</think> and the FINAL answer must be written as "
    "<answer>...</answer>, i.e., <think> reasoning process here </think><answer> answer_here </answer>. User: {question}. Assistant: "
)




def extract_answer_from_gsm8k(gsm8k_answer: str) -> str:
    match = re.search(r"### (.*)", gsm8k_answer)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Could not extract final answer from GSM8K answer: {gsm8k_answer}")


def build_train_val_examples(
    data_config: Dict[str, Any],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load a HuggingFace dataset and return pre-formatted (prompt, answer) pairs.

    Prompts are formatted at load time so each training step skips the per-step
    format call.

    Args:
        data_config: Dict with keys ``dataset_name``, ``subset``, ``train_split``,
                     ``val_split`` (matches the ``data:`` section of config.yaml).

    Returns:
        (train_examples, val_examples) — each a list of (prompt_str, answer_str).
    """
    dataset = load_dataset(
        data_config["dataset_name"],
        data_config.get("subset"),
    )
    train_split = dataset[data_config["train_split"]]
    val_split = dataset[data_config["val_split"]]

    train_examples = [
        (PROMPT.format(question=question), extract_answer_from_gsm8k(answer))
        for question, answer in zip(train_split["question"], train_split["answer"])
    ]
    val_examples = [
        (PROMPT.format(question=question), extract_answer_from_gsm8k(answer))
        for question, answer in zip(val_split["question"], val_split["answer"])
    ]
    return train_examples, val_examples



class QAPairsDataset(torch.utils.data.Dataset[Tuple[str, str]]):
    def __init__(self, questions: List[str], answers: List[str]):
        self.questions = questions
        self.answers = answers

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.questions[index], self.answers[index]


def extract_answer_from_gsm8k(gsm8k_answer: str) -> str:
    match = re.search(r"### (.*)", gsm8k_answer)
    if match:
        return match.group(1).strip()
    raise ValueError(f"Could not extract final answer from GSM8K answer: {gsm8k_answer}")


def build_train_val_dataloaders(
    batch_size: int,
    data_config: dict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset = load_dataset(
        data_config["dataset_name"],
        data_config.get("subset"),
    )
    train_split = dataset[data_config["train_split"]]
    val_split = dataset[data_config["val_split"]]

    train_questions: List[str] = list(train_split["question"])
    train_answers: List[str] = [extract_answer_from_gsm8k(ans) for ans in train_split["answer"]]
    val_questions: List[str] = list(val_split["question"])
    val_answers: List[str] = [extract_answer_from_gsm8k(ans) for ans in val_split["answer"]]

    train_dataset = QAPairsDataset(train_questions, train_answers)
    val_dataset = QAPairsDataset(val_questions, val_answers)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, val_dataloader