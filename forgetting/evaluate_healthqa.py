#!/usr/bin/env python3
"""
Avaliação de catastrophic forgetting: HealthQA-BR (exames de saúde brasileiros).

Carrega Larxel/healthqa-br (5632 questões de múltipla escolha de concursos
de saúde), envia para o modelo e reporta acurácia geral, por grupo profissional
e por fonte.

Uso:
  python evaluate_healthqa.py --model gpt-4.1 --api_key sk-...
  python evaluate_healthqa.py --model test --base_url http://localhost:8000/v1
"""

import re
import sys
import json
import asyncio
import argparse
from collections import defaultdict

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. "
    "Reply with ONLY the letter of the correct answer (A, B, C, D, or E). "
    "Do not include any explanation or other text."
)

ANSWER_RE = re.compile(r"\b([A-Ea-e])\b")
CHOICES_RE = re.compile(r"'([A-E])':")


def parse_answer(text: str, num_choices: int) -> str | None:
    """Extract answer letter from model response (last valid letter)."""
    valid = set("ABCDE"[:num_choices])
    matches = list(ANSWER_RE.finditer(text or ""))
    for m in reversed(matches):
        letter = m.group(1).upper()
        if letter in valid:
            return letter
    return None


def count_choices(question: str) -> int:
    """Count number of multiple-choice alternatives in question text."""
    matches = set(CHOICES_RE.findall(question))
    return len(matches) if matches else 5


async def evaluate(client: AsyncOpenAI, rows: list[dict], args) -> list[dict]:
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            prompt = row["question"]
            num_choices = count_choices(prompt)
            try:
                kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
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
            pred = parse_answer(text, num_choices)
            expected = row["answer"].upper()
            return {
                "id": row["id"],
                "group": row["group"],
                "source": row["source"],
                "year": row["year"],
                "expected": expected,
                "pred": pred,
                "correct": pred == expected,
                "raw_response": text,
            }

    tasks = [call(r) for r in rows]
    return await tqdm_asyncio.gather(*tasks, desc="HealthQA-BR")


def print_report(results: list[dict]):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_answer = sum(1 for r in results if r["pred"] is None)

    print()
    print("=" * 60)
    print("HealthQA-BR — RESULTADOS")
    print("=" * 60)
    print(f"Acurácia Geral:   {correct/total*100:.1f}% ({correct}/{total})")
    print(f"Sem resposta:     {no_answer}")
    print()

    # Per-source breakdown
    source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        source_stats[r["source"]]["total"] += 1
        if r["correct"]:
            source_stats[r["source"]]["correct"] += 1

    print("Acurácia por fonte:")
    for src in sorted(source_stats, key=lambda s: source_stats[s]["total"], reverse=True):
        s = source_stats[src]
        acc = s["correct"] / s["total"] * 100
        print(f"  {src:45s} {acc:5.1f}% ({s['correct']}/{s['total']})")
    print()

    # Per-group breakdown
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        g = r["group"] or "Sem grupo"
        group_stats[g]["total"] += 1
        if r["correct"]:
            group_stats[g]["correct"] += 1

    print("Acurácia por grupo profissional:")
    for grp in sorted(group_stats, key=lambda g: group_stats[g]["total"], reverse=True):
        s = group_stats[grp]
        acc = s["correct"] / s["total"] * 100
        print(f"  {grp:50s} {acc:5.1f}% ({s['correct']}/{s['total']})")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Forgetting evaluation: HealthQA-BR")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--reasoning_effort", default=None, help="Reasoning effort (none/low/medium/high)")
    parser.add_argument("--parallel", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode (Qwen3.5)")
    parser.add_argument("--output", default=None, help="Salvar JSON com respostas")
    args = parser.parse_args()

    ds = load_dataset("Larxel/healthqa-br", split="train")
    rows = list(ds)

    if args.limit:
        rows = rows[: args.limit]

    print(f"Modelo: {args.model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    print(f"Dataset: Larxel/healthqa-br")
    print(f"Questões: {len(rows)} (total: {len(ds)})")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print()

    client = AsyncOpenAI(
        api_key=args.api_key or "no-key",
        base_url=args.base_url,
    )

    results = await evaluate(client, rows, args)

    # Fail-fast: abort if all responses are errors
    errors = [r for r in results if r["raw_response"].startswith("[ERRO]")]
    if errors and len(errors) == len(results):
        print(f"\nERRO: Todas as {len(results)} respostas falharam. Primeira mensagem:")
        print(f"  {errors[0]['raw_response'][:200]}")
        sys.exit(1)

    print_report(results)

    if args.output:
        out = {
            "metadata": {
                "model": args.model,
                "base_url": args.base_url,
                "dataset": "Larxel/healthqa-br",
                "total_questions": len(ds),
                "evaluated": len(rows),
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nRespostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
