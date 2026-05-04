# Teaching LLMs Brazilian Healthcare

🤗 [Collection on HuggingFace](https://huggingface.co/collections/hugo/protocolos-clinicos-br-69f7a8b93b1c28c0914594ff) · 🌐 [Blog post](https://hugoabonizio.github.io/clinical-protocols-br/)

Adapting LLMs to the clinical knowledge of Brazil's Unified Health System (SUS) through continual pre-training (CPT) and reinforcement learning (GRPO) on official Ministry of Health guidelines, with two protocol-grounded benchmarks for measuring guideline-specific recall.

## Overview

We collected the 178 official clinical guidelines published by the Brazilian Ministry of Health (PCDTs, DDTs, Protocols of Use, National Guidelines, and Care Pathways) — about 5.4M tokens — and built a three-stage domain-adaptation pipeline:

1. **Synthetic corpus generation (~70M tokens).** Each guideline is rewritten into three complementary formats (rephrase, Wikipedia-style article, and question–answer pairs with reasoning) by four diverse generator LLMs: GPT-4.1-mini, GPT-5-nano, GPT-OSS-20B, and Qwen3-235B. The full corpus contains 17,800 examples.
2. **Continual pre-training** of Qwen2.5-14B-Instruct on the synthetic corpus.
3. **Reinforcement learning (GRPO)** with LoRA on the HealthBench-BR train split. The reward favors correct verdicts that include at least 50 words of justification, encouraging explicit clinical reasoning rather than label memorization.

The 178 guidelines are split into 89 train / 89 test at the document level. QA-format synthetic data is generated only from train-split guidelines to prevent leakage; rephrase and wiki-style data are generated from all guidelines.

## Benchmarks

- **HealthBench-BR** — 1,780 true/false clinical assertions (10 per guideline, balanced 50/50). Each assertion is paired: one statement is faithful to the guideline, and a second modifies a key detail (dosage, administration route, monitoring interval, etc.). Correct classification requires precise factual recall, not surface-level heuristics.
- **PCDT-QA** — 890 open-ended questions (5 per guideline) with reference answers grounded in the guideline text. Evaluation uses an LLM-as-a-judge pipeline (GPT-4.1) producing a binary correct/incorrect verdict, accommodating the open-ended nature of clinical responses.

Both benchmarks are split at the guideline level (445 or 890 questions per split), with no overlap with the synthetic training data.

## Main results

| Model | HealthBench-BR (test) | PCDT-QA (test) |
|---|---|---|
| Qwen2.5-14B-Instruct (baseline) | 59.4 | 27.9 |
| GPT-4.1 | 73.3 | 70.3 |
| GPT-5.2 (high) | 78.5 | 78.2 |
| Claude Sonnet 4.6 | 77.6 | 70.3 |
| Gemini 3.1 Pro | 79.3 | 80.0 |
| Google AI Overview | 70.5 | 77.3 |
| **Qwen2.5-14B + CPT (4 generators) + RL** | **83.9** | **85.4** |

Our 14B open model outperforms every proprietary frontier baseline on both benchmarks. Ablations show that generator and format diversity are complementary: scaling from one to four generators adds about 20 points on PCDT-QA, and GRPO adds 12.8 points on HealthBench-BR over CPT alone.

## Released artifacts

Everything is collected in the [Protocolos Clínicos BR collection on HuggingFace](https://huggingface.co/collections/hugo/protocolos-clinicos-br-69f7a8b93b1c28c0914594ff).

**Datasets**

- [`hugo/protocolos-clinicos-br`](https://huggingface.co/datasets/hugo/protocolos-clinicos-br) — the 178 raw guidelines (default config) plus the 13 synthetic configs used for CPT
- [`hugo/healthbench-br`](https://huggingface.co/datasets/hugo/healthbench-br) — true/false benchmark
- [`hugo/pcdt-qa`](https://huggingface.co/datasets/hugo/pcdt-qa) — open-ended QA benchmark

**Models** (Qwen2.5-14B-Instruct base, with test-split accuracy on HB-BR / PCDT-QA)

| Model | HB | PCDT |
|---|---|---|
| [`protocolos-clinicos-br-rl-4gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-rl-4gen-14b) ⭐ — CPT (4 gen) + GRPO LoRA *(best)* | 83.9 | 85.4 |
| [`protocolos-clinicos-br-rl-1gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-rl-1gen-14b) — CPT (1 gen) + GRPO LoRA | 75.6 | 65.4 |
| [`protocolos-clinicos-br-cpt-4gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-cpt-4gen-14b) — CPT (4 generators), full FT | 71.1 | 86.3 |
| [`protocolos-clinicos-br-cpt-1gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-cpt-1gen-14b) — CPT (1 generator), full FT | 69.6 | 66.3 |
| [`protocolos-clinicos-br-cpt_lora-4gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-cpt_lora-4gen-14b) — CPT (4 gen) via LoRA | 70.3 | 77.1 |
| [`protocolos-clinicos-br-cpt_lora-1gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-cpt_lora-1gen-14b) — CPT (1 gen) via LoRA | 67.2 | 65.2 |
| [`protocolos-clinicos-br-cpt_replay-1gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-cpt_replay-1gen-14b) — CPT (1 gen) + FineWeb-EN replay | 63.8 | 65.6 |
| [`protocolos-clinicos-br-base_rl-1gen-14b`](https://huggingface.co/hugo/protocolos-clinicos-br-base_rl-1gen-14b) — RL only (no CPT), LoRA | 64.2 | 29.4 |

## Repository layout

```
train/             # CPT, SFT, and RL (GRPO) training scripts
healthbench-br/    # T/F evaluation (with and without RAG)
pcdt-qa/           # Open-ended QA evaluation (with and without RAG)
rag_utils/         # Oracle and RAG retrievers (BM25, embeddings) over the guideline corpus
forgetting/        # OOD benchmarks (BLUEX, MMLU/ARC/HellaSwag, IFEval, HealthQA, DrBode, MedPT)
```

## Citation

```bibtex
@article{TBD,
  title={Teaching LLMs Brazilian Healthcare: Injecting Knowledge from Official Clinical Guidelines},
  author={Abonizio, Hugo and Lopes, Filipe Rocha and Lotufo, Roberto and Nogueira, Rodrigo},
  year={2026}
}
```
