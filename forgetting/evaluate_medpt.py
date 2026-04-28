#!/usr/bin/env python3
"""
Avaliação de catastrophic forgetting: MedPT (classificação de especialidade médica).

Carrega AKCIT/MedPT (384k perguntas reais paciente-médico do Doctoralia),
faz subsampling balanceado das top-N especialidades (por especialidade primária),
pede ao modelo para classificar a especialidade e reporta acurácia + macro F1.

Uso:
  python evaluate_medpt.py --model gpt-4.1 --api_key sk-...
  python evaluate_medpt.py --model test --base_url http://localhost:8000/v1
"""

import re
import sys
import json
import random
import asyncio
import argparse
from collections import Counter, defaultdict

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT_TEMPLATE = (
    "Você é um assistente médico. Dada a pergunta de um paciente, "
    "identifique qual especialidade médica é mais adequada para respondê-la.\n\n"
    "Especialidades possíveis:\n{specialty_list}\n\n"
    "Responda APENAS com o nome exato da especialidade, sem explicação adicional."
)


def get_primary_specialty(spec: str) -> str:
    """Extract primary specialty (first before comma)."""
    return spec.split(",")[0].strip()


def get_top_specialties(dataset, n: int) -> list[str]:
    """Return top-N primary specialties by frequency."""
    primary = [get_primary_specialty(row["medical_specialty"]) for row in dataset]
    counts = Counter(primary)
    return [spec for spec, _ in counts.most_common(n)]


def sample_balanced(dataset, specialties: list[str], n_per_class: int, seed: int) -> list[dict]:
    """Stratified balanced sampling: n_per_class examples per specialty."""
    rng = random.Random(seed)

    # Group by primary specialty
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for row in dataset:
        primary = get_primary_specialty(row["medical_specialty"])
        if primary in specialties:
            by_spec[primary].append(row)

    rows = []
    for spec in specialties:
        pool = by_spec[spec]
        sampled = rng.sample(pool, min(n_per_class, len(pool)))
        for row in sampled:
            rows.append({**row, "_primary_specialty": spec})
    rng.shuffle(rows)
    return rows


def parse_specialty(text: str, specialties: list[str]) -> str | None:
    """Match model response to a specialty. Exact > substring > None."""
    if not text:
        return None
    text_clean = text.strip().lower()

    # Exact match
    for spec in specialties:
        if text_clean == spec.lower():
            return spec

    # Response contains specialty name
    for spec in sorted(specialties, key=len, reverse=True):
        if spec.lower() in text_clean:
            return spec

    # Specialty name contains response (for short answers like "Psicólogo")
    for spec in sorted(specialties, key=len, reverse=True):
        if text_clean in spec.lower() and len(text_clean) >= 4:
            return spec

    return None


def compute_metrics(results: list[dict], specialties: list[str]) -> dict:
    """Compute accuracy and macro F1."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_answer = sum(1 for r in results if r["pred"] is None)

    # Per-class precision/recall/F1
    per_class = {}
    for spec in specialties:
        tp = sum(1 for r in results if r["pred"] == spec and r["expected"] == spec)
        fp = sum(1 for r in results if r["pred"] == spec and r["expected"] != spec)
        fn = sum(1 for r in results if r["pred"] != spec and r["expected"] == spec)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for r in results if r["expected"] == spec)
        per_class[spec] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_precision = sum(v["precision"] for v in per_class.values()) / len(per_class)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(per_class)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "total": total,
        "correct": correct,
        "no_answer": no_answer,
        "per_class": per_class,
    }


async def evaluate(client: AsyncOpenAI, rows: list[dict], args, system_prompt: str) -> list[dict]:
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            try:
                kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["question"]},
                    ],
                )
                extra_body = {}
                if args.reasoning_effort:
                    extra_body["reasoning_effort"] = args.reasoning_effort
                    kwargs["max_completion_tokens"] = args.max_tokens
                else:
                    kwargs["temperature"] = args.temperature
                    kwargs["max_tokens"] = args.max_tokens
                if args.disable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                if extra_body:
                    kwargs["extra_body"] = extra_body
                resp = await client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
            except Exception as e:
                text = f"[ERRO] {e}"
            pred = parse_specialty(text, args._specialties)
            expected = row["_primary_specialty"]
            return {
                "id": row["id"],
                "question": row["question"][:200],
                "condition": row["condition"],
                "expected": expected,
                "pred": pred,
                "correct": pred == expected,
                "raw_response": text,
            }

    tasks = [call(r) for r in rows]
    return await tqdm_asyncio.gather(*tasks, desc="MedPT")


def print_report(metrics: dict):
    print()
    print("=" * 60)
    print("MedPT — RESULTADOS (Classificação de Especialidade)")
    print("=" * 60)
    print(f"Acurácia Geral:   {metrics['accuracy']*100:.1f}% ({metrics['correct']}/{metrics['total']})")
    print(f"Macro F1:         {metrics['macro_f1']*100:.1f}%")
    print(f"Macro Precision:  {metrics['macro_precision']*100:.1f}%")
    print(f"Macro Recall:     {metrics['macro_recall']*100:.1f}%")
    print(f"Sem resposta:     {metrics['no_answer']}")
    print()

    print("Acurácia por especialidade:")
    per_class = metrics["per_class"]
    for spec in sorted(per_class, key=lambda s: per_class[s]["f1"], reverse=True):
        v = per_class[spec]
        acc = v["recall"]  # recall == per-class accuracy with balanced sampling
        print(f"  {spec:45s} F1={v['f1']*100:5.1f}%  Acc={acc*100:5.1f}%  (n={v['support']})")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Forgetting evaluation: MedPT specialty classification")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--reasoning_effort", default=None, help="Reasoning effort (none/low/medium/high)")
    parser.add_argument("--parallel", type=int, default=100)
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode (Qwen3.5)")
    parser.add_argument("--n_specialties", type=int, default=20, help="Number of top specialties to evaluate")
    parser.add_argument("--n_per_class", type=int, default=250, help="Samples per specialty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="Salvar JSON com respostas")
    args = parser.parse_args()

    ds = load_dataset("AKCIT/MedPT", split="train")

    # Determine top-N specialties and sample
    specialties = get_top_specialties(ds, args.n_specialties)
    rows = sample_balanced(ds, specialties, args.n_per_class, args.seed)

    if args.limit:
        rows = rows[: args.limit]

    # Store specialties on args for use in evaluate
    args._specialties = specialties

    # Build system prompt with specialty list
    specialty_list = "\n".join(f"- {s}" for s in specialties)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(specialty_list=specialty_list)

    print(f"Modelo: {args.model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    print(f"Dataset: AKCIT/MedPT")
    print(f"Especialidades: {args.n_specialties} (top por frequência)")
    print(f"Amostras por classe: {args.n_per_class}")
    print(f"Total amostras: {len(rows)} (dataset completo: {len(ds)})")
    print(f"Seed: {args.seed}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print()

    client = AsyncOpenAI(
        api_key=args.api_key or "no-key",
        base_url=args.base_url,
    )

    results = await evaluate(client, rows, args, system_prompt)

    # Fail-fast: abort if all responses are errors
    errors = [r for r in results if r["raw_response"].startswith("[ERRO]")]
    if errors and len(errors) == len(results):
        print(f"\nERRO: Todas as {len(results)} respostas falharam. Primeira mensagem:")
        print(f"  {errors[0]['raw_response'][:200]}")
        sys.exit(1)

    metrics = compute_metrics(results, specialties)
    print_report(metrics)

    if args.output:
        out = {
            "metadata": {
                "model": args.model,
                "base_url": args.base_url,
                "dataset": "AKCIT/MedPT",
                "n_specialties": args.n_specialties,
                "n_per_class": args.n_per_class,
                "total_evaluated": len(rows),
                "total_dataset": len(ds),
                "seed": args.seed,
                "specialties": specialties,
            },
            "metrics": {k: v for k, v in metrics.items() if k != "per_class"},
            "per_class": metrics["per_class"],
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nRespostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
