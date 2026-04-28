#!/usr/bin/env python3
"""
Avaliação de catastrophic forgetting: BLUEX (vestibulares brasileiros).

Carrega portuguese-benchmark-datasets/BLUEX, filtra questões com imagem,
envia múltipla escolha para o modelo e reporta acurácia geral e por subject.

Uso:
  python evaluate_bluex.py --model gpt-4.1 --api_key sk-...
  python evaluate_bluex.py --model test --base_url http://localhost:8000/v1
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


def parse_answer(text: str, num_choices: int) -> str | None:
    """Extract answer letter from model response (last valid letter)."""
    valid = set("ABCDE"[:num_choices])
    matches = list(ANSWER_RE.finditer(text or ""))
    for m in reversed(matches):
        letter = m.group(1).upper()
        if letter in valid:
            return letter
    return None


def format_prompt(question: str, alternatives: list[str]) -> str:
    letters = "ABCDE"
    options = "\n".join(
        f"{letters[i]}. {alt}" for i, alt in enumerate(alternatives)
    )
    return f"{question}\n\n{options}"


async def evaluate(client: AsyncOpenAI, rows: list[dict], args) -> list[dict]:
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            prompt = format_prompt(row["question"], row["alternatives"])
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
            pred = parse_answer(text, len(row["alternatives"]))
            expected = row["answer"].upper()
            return {
                "question": row["question"],
                "subject": row["subject"],
                "expected": expected,
                "pred": pred,
                "correct": pred == expected,
                "raw_response": text,
            }

    tasks = [call(r) for r in rows]
    return await tqdm_asyncio.gather(*tasks, desc="BLUEX")


def print_report(results: list[dict]):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_answer = sum(1 for r in results if r["pred"] is None)

    print()
    print("=" * 60)
    print("BLUEX — RESULTADOS")
    print("=" * 60)
    print(f"Acurácia Geral:   {correct/total*100:.1f}% ({correct}/{total})")
    print(f"Sem resposta:     {no_answer}")
    print()

    subj_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        for subj in r["subject"]:
            subj_stats[subj]["total"] += 1
            if r["correct"]:
                subj_stats[subj]["correct"] += 1

    print("Acurácia por subject:")
    for subj in sorted(subj_stats, key=lambda s: subj_stats[s]["total"], reverse=True):
        s = subj_stats[subj]
        acc = s["correct"] / s["total"] * 100
        print(f"  {subj:30s} {acc:5.1f}% ({s['correct']}/{s['total']})")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Forgetting evaluation: BLUEX")
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

    ds = load_dataset("portuguese-benchmark-datasets/BLUEX", split="questions")

    # Filter: skip images and null answers
    rows = [
        r for r in ds
        if not r["has_associated_images"] and r["answer"] is not None
    ]
    total_raw = len(ds)
    filtered_images = sum(1 for r in ds if r["has_associated_images"])
    filtered_null = sum(1 for r in ds if r["answer"] is None)

    if args.limit:
        rows = rows[: args.limit]

    print(f"Modelo: {args.model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    print(f"Dataset: portuguese-benchmark-datasets/BLUEX")
    print(f"Questões: {len(rows)} (total: {total_raw}, filtradas: {filtered_images} imagem, {filtered_null} sem resposta)")
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
                "dataset": "portuguese-benchmark-datasets/BLUEX",
                "total_questions": total_raw,
                "filtered_images": filtered_images,
                "filtered_null_answer": filtered_null,
                "evaluated": len(rows),
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nRespostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
