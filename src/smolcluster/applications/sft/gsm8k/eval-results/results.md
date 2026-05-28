# GSM8K SFT — Evaluation Results

## Model Versions

| Version | Description |
|---------|-------------|
| **base** | `mlx-community/Qwen2.5-0.5B-Instruct-bf16` — no fine-tuning |
| **sft** | base + LoRA SFT on GSM8K chain-of-thought _(pending)_ |

---

## Summary

| Task | Metric | Base |
|------|--------|------|
| **GSM8K CoT (0-shot)** | exact\_match (strict) | 33.51% |
| **IFEval** | prompt strict acc | 19.78% |
| **IFEval** | inst strict acc | 32.61% |
| **MMLU** | acc (macro avg) | 45.88% |
| **ARC-Challenge** | acc\_norm | 33.45% |
| **HellaSwag** | acc\_norm | 52.28% |

---

## Base Model — Full Results

**Model:** `mlx-community/Qwen2.5-0.5B-Instruct-bf16`  
**Eval date:** 2026-05-27  
**Eval harness:** lm-evaluation-harness, 0-shot, device: mps

### GSM8K CoT Zero-Shot (1,319 examples)

| Metric | Score | ±stderr |
|--------|-------|---------|
| exact\_match (strict) | **33.51%** | ±1.30% |
| exact\_match (flexible-extract) | 2.88% | ±0.46% |

### IFEval (541 examples)

| Metric | Score | ±stderr |
|--------|-------|---------|
| prompt-level strict acc | **19.78%** | ±1.71% |
| prompt-level loose acc | 22.37% | ±1.79% |
| inst-level strict acc | **32.61%** | — |
| inst-level loose acc | 34.65% | — |

### MMLU (14,042 examples)

| Category | Acc | ±stderr |
|----------|-----|---------|
| **Overall** | **45.88%** | ±0.41% |
| STEM | 39.36% | ±0.86% |
| Humanities | 42.44% | ±0.69% |
| Social Sciences | 52.75% | ±0.89% |
| Other | 50.92% | ±0.88% |

<details>
<summary>MMLU per-subject breakdown</summary>

| Subject | Acc | n |
|---------|-----|---|
| abstract\_algebra | 31.00% | 100 |
| anatomy | 42.96% | 135 |
| astronomy | 46.05% | 152 |
| business\_ethics | 53.00% | 100 |
| clinical\_knowledge | 50.94% | 265 |
| college\_biology | 44.44% | 144 |
| college\_chemistry | 31.00% | 100 |
| college\_computer\_science | 35.00% | 100 |
| college\_mathematics | 30.00% | 100 |
| college\_medicine | 45.66% | 173 |
| college\_physics | 29.41% | 102 |
| computer\_security | 71.00% | 100 |
| conceptual\_physics | 37.87% | 235 |
| econometrics | 30.70% | 114 |
| electrical\_engineering | 51.03% | 145 |
| elementary\_mathematics | 34.92% | 378 |
| formal\_logic | 29.37% | 126 |
| global\_facts | 30.00% | 100 |
| high\_school\_biology | 53.23% | 310 |
| high\_school\_chemistry | 39.41% | 203 |
| high\_school\_computer\_science | 44.00% | 100 |
| high\_school\_european\_history | 60.00% | 165 |
| high\_school\_geography | 56.57% | 198 |
| high\_school\_government\_and\_politics | 52.85% | 193 |
| high\_school\_macroeconomics | 43.59% | 390 |
| high\_school\_mathematics | 29.63% | 270 |
| high\_school\_microeconomics | 45.38% | 238 |
| high\_school\_physics | 26.49% | 151 |
| high\_school\_psychology | 63.12% | 545 |
| high\_school\_statistics | 32.87% | 216 |
| high\_school\_us\_history | 54.90% | 204 |
| high\_school\_world\_history | 59.92% | 237 |
| human\_aging | 55.16% | 223 |
| human\_sexuality | 54.96% | 131 |
| international\_law | 73.55% | 121 |
| jurisprudence | 58.33% | 108 |
| logical\_fallacies | 49.08% | 163 |
| machine\_learning | 41.07% | 112 |
| management | 59.22% | 103 |
| marketing | 73.93% | 234 |
| medical\_genetics | 48.00% | 100 |
| miscellaneous | 55.43% | 783 |
| moral\_disputes | 54.34% | 346 |
| moral\_scenarios | 23.80% | 895 |
| nutrition | 59.48% | 306 |
| philosophy | 49.20% | 311 |
| prehistory | 54.01% | 324 |
| professional\_accounting | 32.27% | 282 |
| professional\_law | 35.53% | 1534 |
| professional\_medicine | 36.76% | 272 |
| professional\_psychology | 45.59% | 612 |
| public\_relations | 55.45% | 110 |
| security\_studies | 54.29% | 245 |
| sociology | 67.16% | 201 |
| us\_foreign\_policy | 72.00% | 100 |
| virology | 43.98% | 166 |
| world\_religions | 59.06% | 171 |

</details>

### ARC-Challenge (1,172 examples)

| Metric | Score | ±stderr |
|--------|-------|---------|
| acc | 30.80% | ±1.35% |
| **acc\_norm** | **33.45%** | ±1.38% |

### HellaSwag (10,042 examples)

| Metric | Score | ±stderr |
|--------|-------|---------|
| acc | 40.55% | ±0.49% |
| **acc\_norm** | **52.28%** | ±0.50% |

---

## SFT Model — Results

_To be filled after full training run with best sweep hyperparams._

| Task | Metric | Base | SFT | Δ |
|------|--------|------|-----|---|
| GSM8K CoT (0-shot) | exact\_match strict | 33.51% | — | — |
| IFEval | prompt strict acc | 19.78% | — | — |
| IFEval | inst strict acc | 32.61% | — | — |
| MMLU | acc | 45.88% | — | — |
| ARC-Challenge | acc\_norm | 33.45% | — | — |
| HellaSwag | acc\_norm | 52.28% | — | — |
