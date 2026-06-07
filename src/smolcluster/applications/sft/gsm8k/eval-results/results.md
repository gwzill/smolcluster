# GSM8K SFT Eval Results

**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-bf16`
**Eval date:** 2026-06-06 / 2026-06-07
**Framework:** lm-evaluation-harness, 0-shot, greedy decoding, `--apply_chat_template`

---

## GSM8K

| Variant | strict-match | flexible-extract |
|---------|-------------|-----------------|
| Base | 0.4610 ± 0.014 | 0.4731 ± 0.014 |
| SFT-nothink | 0.3571 ± 0.013 | 0.3594 ± 0.013 |
| SFT-think | 0.3268 ± 0.013 | 0.3283 ± 0.013 |

> strict ≈ flexible for both SFT variants confirming correct output format:
> bare number at end (nothink) and `<answer>N</answer>` tags (think).

---

## ARC-Challenge

| Variant | acc | acc_norm |
|---------|-----|----------|
| Base | 0.2841 ± 0.013 | 0.3242 ± 0.014 |
| SFT-nothink | 0.2765 ± 0.013 | 0.3157 ± 0.014 |
| SFT-think | 0.2654 ± 0.013 | 0.3080 ± 0.014 |

---

## HellaSwag

| Variant | acc | acc_norm |
|---------|-----|----------|
| Base | 0.3964 ± 0.005 | 0.4854 ± 0.005 |
| SFT-nothink | 0.3921 ± 0.005 | 0.4954 ± 0.005 |
| SFT-think | 0.3874 ± 0.005 | 0.4554 ± 0.005 |

---

## IFEval

| Variant | prompt strict | prompt loose | inst strict | inst loose |
|---------|--------------|--------------|-------------|------------|
| Base | 0.2588 ± 0.019 | 0.2865 ± 0.020 | 0.3609 | 0.3861 |
| SFT-nothink | 0.1811 ± 0.017 | 0.2033 ± 0.017 | 0.2770 | 0.2998 |
| SFT-think | 0.1534 ± 0.016 | 0.2200 ± 0.018 | 0.2698 | 0.3357 |

---

## MMLU

Not run for any variant.

---

## Summary

- **Base outperforms both SFT variants on GSM8K** (0.473 vs 0.359/0.328 flexible-extract). Models are undertrained — Optuna sweep still running to find better hyperparameters.
- **SFT-think < SFT-nothink on GSM8K** (0.327 vs 0.357) — think-format adds generation overhead with no accuracy gain at this checkpoint.
- **HellaSwag drops most for SFT-think** (-0.030 acc_norm vs base vs -0.010 for nothink), suggesting the `<think>` format more aggressively displaces general language modelling behaviour.
- **IFEval regresses for both SFT variants** — expected for narrow GSM8K-only fine-tuning on a 0.5B model.
- **SFT-nothink is the best SFT checkpoint** at this stage across all benchmarks.

---

## Raw Results Index

| Session | Variant | Model | Tasks |
|---------|---------|-------|-------|
| `base/lm_eval_20260606_074712_*` | Base | `Qwen2.5-0.5B-Instruct-bf16` | arc_challenge, hellaswag, ifeval |
| `base/lm_eval_20260606_191553_*` | Base | `Qwen2.5-0.5B-Instruct-bf16` | gsm8k |
| `sft-nothink/lm_eval_20260607_150433_*` | SFT-nothink | `sft-no-think/final_model` | gsm8k, arc_challenge, hellaswag, ifeval |
| `sft-think/lm_eval_20260607_073455_*` | SFT-think | `sft-think/final_model` | gsm8k, arc_challenge, hellaswag, ifeval |
