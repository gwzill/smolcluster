# GRPO in Smolcluster

This directory contains Smolcluster's MLX-based implementation of Group Relative Policy Optimization (GRPO) for reasoning-style tasks such as GSM8K and summarization.

The implementation is organized around one training process that updates the policy model locally and one or more rollout workers that generate completions from a serving model. The training loop scores groups of completions, converts rewards into within-group advantages, and updates the policy with a clipped objective.

---

## Table of Contents

- [How GRPO Works](#how-grpo-works)
- [GSM8K](#gsm8k)
  - [Training](#gsm8k-training)
  - [Rewards](#gsm8k-rewards)
  - [Evaluation](#gsm8k-evaluation)
- [Summarization](#summarization)
  - [Training](#summarization-training)
  - [Rewards](#summarization-rewards)
  - [Evaluation](#summarization-evaluation)
  - [Hosted Eval Artifacts](#hosted-eval-artifacts)
- [Config Knobs](#config-knobs)
- [Folder Structure](#folder-structure)
- [File Map](#file-map)

---

## How GRPO Works

At a high level, each GRPO step looks like this:

1. Sample a batch of prompts.
2. Generate multiple rollouts per prompt.
3. Score each rollout with task-specific reward functions.
4. Normalize rewards within each prompt group to produce advantages.
5. Compute per-rollout completion log-probabilities under the current policy.
6. Apply a PPO-style clipped objective, optionally with a KL penalty.
7. Update the policy on GPU via MLX.
8. Periodically save checkpoints and optionally sync rollout workers to the newest policy.

> The model scores only completion tokens, not prompt tokens, when computing rollout log-probabilities.

Read more about GRPO [here](https://www.alphaxiv.org/abs/2402.03300).

The default GRPO configuration lives in `src/smolcluster/configs/reasoning/grpo/config.yaml`.

---

## GSM8K

### GSM8K Training

**Entry point:** `train_gsm8k.py`

**Dataset:** GSM8K math word problems.

```bash
cd src/smolcluster/applications/reasoning/grpo
uv run train_gsm8k.py
```

### GSM8K Rewards

The GSM8K training path combines three simple reward terms, all implemented in `rewards/math_rewards.py`.

| Reward | Function | Range | What it measures |
|---|---|---|---|
| `answer_reward` | `calculate_answer_reward` | {0, 1} | Whether the predicted numeric answer matches the target |
| `think_reward` | `calculate_think_reward` | {0, 1} | Whether the model used `<think>...</think>` tags with non-empty reasoning |
| `formatted_reward` | `calculate_formatted_reward` | {0, 1} | Whether the model used both `<think>` and `<answer>` tags correctly with a parseable number |

Total reward:

$$
r = r_{\text{answer}} + 0.1 \cdot r_{\text{think}} + 0.1 \cdot r_{\text{format}}
$$

A fully correct, properly formatted response with reasoning tags gets a maximum reward of $1.2$.

### GSM8K Evaluation

**Script:** `evaluation/evaluate_gsm8k.py`

**Dataset:** GSM8K test split.

Computes sampled accuracy-style metrics and supports checkpoint comparison.

```bash
export OPENAI_API_KEY=your_key_here
cd src/smolcluster/applications/reasoning/grpo/evaluation
uv run evaluate_gsm8k.py --checkpoint-dir ../../checkpoints/grpo-gsm8k/latest
```

---

## Summarization

### Summarization Training

**Entry point:** `train_summarization.py`

**Dataset:** `mlabonne/smoltldr` (Reddit posts, train split).

```bash
cd src/smolcluster/applications/reasoning/grpo
uv run train_summarization.py
```

### Summarization Rewards

The summarization training path uses composable reward signals implemented in `rewards/summarization_rewards.py`. Each signal is toggled independently via `quality_metrics` in `config.yaml`.

#### Quality metrics (`calculate_summary_quality`)

| Key | Range | What it measures |
|---|---|---|
| `rouge` | [0, 1] | ROUGE-L F1 vs reference — phrase ordering via longest-common-subsequence overlap |
| `meteor` | [0, 1] | METEOR score — harmonic mean of precision/recall with stemming and synonym matching |
| `bleu` | [0, 1] | BLEU score — n-gram precision vs reference |

#### Length penalty (`calculate_length_reward`)

| Key | Range | What it measures |
|---|---|---|
| `length_penalty` | (-1, 0] | Penalises deviation from a target token length (default: 64 tokens). 0 = exactly on target. Uses tokenizer token count when a tokenizer is passed, else character count. |

$$
r_{\text{length}} = -\frac{|len(\hat{y}) - L_{\text{target}}|}{L_{\text{target}}}
$$

#### Total reward

The total reward is the **sum of all enabled signals** (no fixed weights):

$$
r = \sum_{m \in \text{enabled}} r_m
$$

#### Configuring rewards

Toggle signals in `config.yaml`:

```yaml
quality_metrics:
  length_penalty: true   # penalise deviation from 64-token target length
  rouge: false           # ROUGE-L F1 vs reference
  meteor: false          # METEOR score
  bleu: false            # BLEU score
```

The default ships with `length_penalty: true` and all quality metrics off — a pure length-control baseline. The ablation experiments enable quality metrics one at a time (or in combination) on top of this baseline.

### Summarization Evaluation

**Dataset:** `mlabonne/smoltldr` (validation split).

#### Generate summaries and score with G-Eval metrics

**Script:** `evaluation/evaluate_summarization.py`

Generates summaries on the validation split, then scores each with four LLM-judge metrics:
- Faithfulness
- Coverage
- Conciseness
- Clarity

```bash
export OPENAI_API_KEY=your_key_here
cd src/smolcluster/applications/reasoning/grpo/evaluation
uv run evaluate_summarization.py --checkpoint-dir ../../checkpoints/grpo-summarization-length-quality/latest
```

#### Compare two evaluation runs

**Script:** `evaluation/compare_eval_runs.py`

Compares two saved summarization eval runs with paired significance tests on per-example metric scores and composite score.

```bash
cd src/smolcluster/applications/reasoning/grpo/evaluation
uv run python compare_eval_runs.py \
  --baseline-run grpo-summarization-length-only \
  --candidate-run grpo-summarization-length-quality \
  --alpha 0.05
```

Run names correspond to directories in `eval-rollouts/`. Output is saved to `eval-rollouts/<candidate>/comparison-vs-<baseline>.json`.

Eval also writes local artifacts under `evaluation/eval-rollouts/<run_tag>/`:

- `rollouts.json` — per-example prompts, generations, and judge outputs
- `summary.json` — aggregate metric means, pass rates, and run metadata
- comparison or significance JSON reports when statistical tests are run

### Hosted Eval Artifacts

**Dataset repo (eval rollouts):** [YuvrajSingh9886/reddit-posts-summarization-grpo](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo)

The dataset is organised by model → reward regime → run:

```
LFM2.5-350M-bf16/
├── combined_results.md
├── length-penalty-included/
│   ├── results_summary.md
│   └── grpo-summarization-<run>/rollouts.json + summary.json + comparison*.json
└── length-penalty-fine-tuned/
    ├── results_summary.md
    └── grpo-summarization-<run>/rollouts.json + summary.json + comparison*.json
Qwen2.5-0.5b-Instruct-bf16/   (same structure)
baseline/
    results_summary.md
    LFM2.5-350M-bf16/   + Qwen2.5-0.5b-Instruct-bf16/
```

#### Result summaries

| Model | Regime | Link |
|---|---|---|
| LFM | length-penalty-included | [results_summary.md](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/results_summary.md) |
| LFM | length-penalty-fine-tuned | [results_summary.md](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-fine-tuned/results_summary.md) |
| LFM | combined | [combined_results.md](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/combined_results.md) |
| Qwen | length-penalty-included | [results_summary.md](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/Qwen2.5-0.5b-Instruct-bf16/length-penalty-included/results_summary.md) |
| Qwen | length-penalty-fine-tuned | [results_summary.md](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/Qwen2.5-0.5b-Instruct-bf16/length-penalty-fine-tuned/results_summary.md) |
| Qwen | combined | [combined_results.md](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/Qwen2.5-0.5b-Instruct-bf16/combined_results.md) |
| Both | baseline | [results_summary.md](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/baseline/results_summary.md) |

#### Eval rollout links — LFM2.5-350M, length-penalty-included

| Run | rollouts | summary |
|---|---|---|
| length-only | [rollouts.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-only/rollouts.json) | [summary.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-only/summary.json) |
| length-quality-rouge | [rollouts.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-rouge/rollouts.json) | [summary.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-rouge/summary.json) |
| length-quality-meteor | [rollouts.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-meteor/rollouts.json) | [summary.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-meteor/summary.json) |
| length-quality-bleu | [rollouts.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-bleu/rollouts.json) | [summary.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-bleu/summary.json) |
| length-quality-meteor-rouge | [rollouts.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-meteor-rouge/rollouts.json) | [summary.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-meteor-rouge/summary.json) |
| length-quality-meteor-bleu | [rollouts.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-meteor-bleu/rollouts.json) | [summary.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-meteor-bleu/summary.json) |
| length-quality-bleu-rouge | [rollouts.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-bleu-rouge/rollouts.json) | [summary.json](https://huggingface.co/datasets/YuvrajSingh9886/reddit-posts-summarization-grpo/resolve/main/LFM2.5-350M-bf16/length-penalty-included/grpo-summarization-length-quality-bleu-rouge/summary.json) |

For Qwen or fine-tuned runs, replace `LFM2.5-350M-bf16/length-penalty-included` with `Qwen2.5-0.5b-Instruct-bf16/length-penalty-included` or `*/length-penalty-fine-tuned` respectively.

#### Checkpoint model repos

All final checkpoints are uploaded as individual HF model repos.

**LFM2.5-350M — length-penalty-included**

| Run | Model repo |
|---|---|
| length-only | [LFM2.5-350M-grpo-summarization-length-only](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-length-only) |
| length-quality-rouge | [LFM2.5-350M-grpo-summarization-length-quality-rouge](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-length-quality-rouge) |
| length-quality-meteor | [LFM2.5-350M-grpo-summarization-length-quality-meteor](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-length-quality-meteor) |
| length-quality-bleu | [LFM2.5-350M-grpo-summarization-length-quality-bleu](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-length-quality-bleu) |
| length-quality-meteor-rouge | [LFM2.5-350M-grpo-summarization-length-quality-meteor-rouge](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-length-quality-meteor-rouge) |
| length-quality-meteor-bleu | [LFM2.5-350M-grpo-summarization-length-quality-meteor-bleu](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-length-quality-meteor-bleu) |
| length-quality-bleu-rouge | [LFM2.5-350M-grpo-summarization-length-quality-bleu-rouge](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-length-quality-bleu-rouge) |

**LFM2.5-350M — length-penalty-fine-tuned**

| Run | Model repo |
|---|---|
| quality-rouge | [LFM2.5-350M-grpo-summarization-quality-rouge](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-quality-rouge) |
| quality-meteor | [LFM2.5-350M-grpo-summarization-quality-meteor](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-quality-meteor) |
| quality-bleu | [LFM2.5-350M-grpo-summarization-quality-bleu](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-quality-bleu) |
| quality-meteor-rouge | [LFM2.5-350M-grpo-summarization-quality-meteor-rouge](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-quality-meteor-rouge) |
| quality-meteor-bleu | [LFM2.5-350M-grpo-summarization-quality-meteor-bleu](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-quality-meteor-bleu) |
| quality-bleu-rouge | [LFM2.5-350M-grpo-summarization-quality-bleu-rouge](https://huggingface.co/YuvrajSingh9886/LFM2.5-350M-grpo-summarization-quality-bleu-rouge) |

**Qwen2.5-0.5B — length-penalty-included**

| Run | Model repo |
|---|---|
| length-only | [Qwen2.5-0.5B-grpo-summarization-length-only](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-length-only) |
| length-quality-rouge | [Qwen2.5-0.5B-grpo-summarization-length-quality-rouge](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-length-quality-rouge) |
| length-quality-meteor | [Qwen2.5-0.5B-grpo-summarization-length-quality-meteor](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-length-quality-meteor) |
| length-quality-bleu | [Qwen2.5-0.5B-grpo-summarization-length-quality-bleu](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-length-quality-bleu) |
| length-quality-meteor-rouge | [Qwen2.5-0.5B-grpo-summarization-length-quality-meteor-rouge](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-length-quality-meteor-rouge) |
| length-quality-meteor-bleu | [Qwen2.5-0.5B-grpo-summarization-length-quality-meteor-bleu](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-length-quality-meteor-bleu) |
| length-quality-bleu-rouge | [Qwen2.5-0.5B-grpo-summarization-length-quality-bleu-rouge](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-length-quality-bleu-rouge) |

**Qwen2.5-0.5B — length-penalty-fine-tuned**

| Run | Model repo |
|---|---|
| quality-rouge | [Qwen2.5-0.5B-grpo-summarization-quality-rouge](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-quality-rouge) |
| quality-meteor | [Qwen2.5-0.5B-grpo-summarization-quality-meteor](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-quality-meteor) |
| quality-bleu | [Qwen2.5-0.5B-grpo-summarization-quality-bleu](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-quality-bleu) |
| quality-meteor-rouge | [Qwen2.5-0.5B-grpo-summarization-quality-meteor-rouge](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-quality-meteor-rouge) |
| quality-meteor-bleu | [Qwen2.5-0.5B-grpo-summarization-quality-meteor-bleu](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-quality-meteor-bleu) |
| quality-bleu-rouge | [Qwen2.5-0.5B-grpo-summarization-quality-bleu-rouge](https://huggingface.co/YuvrajSingh9886/Qwen2.5-0.5B-grpo-summarization-quality-bleu-rouge) |

---

## Config Knobs

The most important GRPO settings live in `config.yaml`.

| Key | What it controls |
|---|---|
| `device` | MLX device — usually `gpu` on Apple Silicon |
| `dtype` | `float32` or `bfloat16` |
| `num_epochs` | Number of passes over the dataset |
| `batch_size` | Prompts per optimization batch |
| `num_rollouts` | Completions generated per worker per prompt |
| `max_input_tokens` | Maximum prompt-plus-completion token budget |
| `use_kl` | Whether to include the reference-model KL term |
| `kl_beta` | KL penalty weight |
| `clip_ratio` | PPO clipping parameter |
| `grad_checkpoint` | MLX gradient checkpointing for lower memory usage |
| `grad_chunk_size` | Prompt chunk size during backward computation |
| `rollout_grad_chunk` | Rollout chunk size during backward computation |
| `force_lora` | Force LoRA training even when not strictly required |
| `weight_sync.save_every_steps` | How often to refresh `checkpoints/grpo/latest` |
| `weight_sync.sync_steps` | How often to push fresh weights to rollout workers |
| `vllm` | Whether rollout generation is delegated to vLLM workers |

Related config files:

- `src/smolcluster/configs/reasoning/grpo/config.yaml`
- `src/smolcluster/configs/inference/model_config_inference.yaml`
- `src/smolcluster/configs/inference/cluster_config_inference.yaml`

---

## Folder Structure

```text
src/smolcluster/applications/reasoning/grpo/
├── README.md                  # This guide
├── train_gsm8k.py             # GRPO training entry point for GSM8K
├── train_summarization.py     # GRPO training entry point for summarization
├── data/
│   ├── gsm8k.py               # GSM8K data loading and prompt prep
│   └── summarization.py       # Summarization data loading and prompt prep
├── rewards/
│   ├── math_rewards.py        # GSM8K reward helpers
│   └── summarization_rewards.py
├── evaluation/
│   ├── compare_eval_runs.py   # paired run-vs-run summarization comparison
│   ├── evaluate_gsm8k.py      # pass@k and checkpoint-comparison evaluation
│   └── evaluate_summarization.py
├── scripts/
│   └── launch_grpo_train.sh   # tmux + vLLM launcher with health checks
└── utils/
    ├── amp.py                 # mixed precision helpers
    ├── rollouts.py            # rollout requests to inference workers
    ├── training_utils.py      # tokenization, batching, device helpers
    └── worker_sync.py         # checkpoint sync and worker reload logic
```

Checkpoints are saved to:

- `checkpoints/grpo/step_<N>/`
- `checkpoints/grpo/latest/` — stable rolling checkpoint, overwritten on each periodic save

---

## File Map

| File | Purpose |
|---|---|
| `train_gsm8k.py` | Reward computation, advantage normalization, GRPO loss, training loop |
| `train_summarization.py` | Same GRPO loop for summarization |
| `utils/rollouts.py` | Rollout request fan-out to workers |
| `utils/worker_sync.py` | Periodic checkpoint save and worker reload |
| `evaluation/evaluate_gsm8k.py` | Sampled evaluation and checkpoint comparison |
| `evaluation/evaluate_summarization.py` | LLM-judge summarization evaluation and artifact writing |
| `evaluation/compare_eval_runs.py` | Paired statistical comparison between saved summarization eval runs |
