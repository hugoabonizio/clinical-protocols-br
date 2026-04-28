#!/usr/bin/env python3
"""
Avaliação de catastrophic forgetting: DrBodeBench (residência médica brasileira).

Carrega recogna-nlp/drbodebench (1598 questões INEP/Revalida + FUVEST/USP, 2011-2025),
usa Chain-of-Thought em português e reporta acurácia geral, por origem, por ano e
com/sem imagem.

Questões com imagem incluem a descrição textual (img_description) no prompt como
substituto da imagem original.

Uso:
  python evaluate_drbodebench.py --model gpt-4.1 --api_key sk-...
  python evaluate_drbodebench.py --model test --base_url http://localhost:8000/v1
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
    "Você é um médico respondendo uma questão de residência médica.\n"
    "Analise a questão passo a passo, raciocinando sobre cada alternativa, "
    "e ao final indique a resposta no formato:\n"
    "Resposta: X\n"
    "onde X é a letra da alternativa correta (A, B, C, D ou E)."
)

ANSWER_STRUCTURED_RE = re.compile(r"[Rr]esposta:\s*([A-Ea-e])")
ANSWER_FALLBACK_RE = re.compile(r"\b([A-Ea-e])\b")


def parse_answer(text: str, num_choices: int) -> str | None:
    """Extract answer letter: first try 'Resposta: X', then fallback to last valid letter."""
    valid = set("ABCDE"[:num_choices])

    # Try structured format first
    matches = list(ANSWER_STRUCTURED_RE.finditer(text or ""))
    for m in reversed(matches):
        letter = m.group(1).upper()
        if letter in valid:
            return letter

    # Fallback: last valid letter in response
    matches = list(ANSWER_FALLBACK_RE.finditer(text or ""))
    for m in reversed(matches):
        letter = m.group(1).upper()
        if letter in valid:
            return letter
    return None


def format_prompt(row: dict) -> tuple[str, int]:
    """Build prompt from DrBodeBench row. Returns (prompt_text, num_choices)."""
    parts = []

    # Include image description if available
    if row["contains_img"] and row["img_description"]:
        parts.append(f"[Descrição da imagem associada à questão]\n{row['img_description']}\n")

    parts.append(row["enunciado"])
    parts.append("")  # blank line before alternatives

    num_choices = 0
    for letter in "ABCDE":
        alt = row["alternativas"].get(letter)
        if alt is not None:
            parts.append(f"{letter}) {alt}")
            num_choices += 1

    return "\n".join(parts), num_choices


async def evaluate(client: AsyncOpenAI, rows: list[dict], args) -> list[dict]:
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            prompt, num_choices = format_prompt(row)
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
            expected = row["resposta"].upper()
            return {
                "origem": row["origem"],
                "prova": row["prova"],
                "numero": row["numero"],
                "has_image": row["contains_img"],
                "has_img_description": bool(row["img_description"]),
                "num_choices": num_choices,
                "expected": expected,
                "pred": pred,
                "correct": pred == expected,
                "raw_response": text,
            }

    tasks = [call(r) for r in rows]
    return await tqdm_asyncio.gather(*tasks, desc="DrBodeBench")


def print_report(results: list[dict]):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_answer = sum(1 for r in results if r["pred"] is None)

    print()
    print("=" * 60)
    print("DrBodeBench — RESULTADOS")
    print("=" * 60)
    print(f"Acurácia Geral:   {correct/total*100:.1f}% ({correct}/{total})")
    print(f"Sem resposta:     {no_answer}")
    print()

    # Image breakdown
    img_types = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if not r["has_image"]:
            key = "Sem imagem"
        elif r["has_img_description"]:
            key = "Com imagem (com descrição)"
        else:
            key = "Com imagem (sem descrição)"
        img_types[key]["total"] += 1
        if r["correct"]:
            img_types[key]["correct"] += 1

    print("Acurácia por tipo de questão:")
    for key in ["Sem imagem", "Com imagem (com descrição)", "Com imagem (sem descrição)"]:
        if key in img_types:
            s = img_types[key]
            acc = s["correct"] / s["total"] * 100
            print(f"  {key:35s} {acc:5.1f}% ({s['correct']}/{s['total']})")
    print()

    # Per-origin
    origin_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        origin_stats[r["origem"]]["total"] += 1
        if r["correct"]:
            origin_stats[r["origem"]]["correct"] += 1

    print("Acurácia por origem:")
    for orig in sorted(origin_stats, key=lambda o: origin_stats[o]["total"], reverse=True):
        s = origin_stats[orig]
        acc = s["correct"] / s["total"] * 100
        print(f"  {orig:35s} {acc:5.1f}% ({s['correct']}/{s['total']})")
    print()

    # Per-year
    year_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        year_stats[r["prova"]]["total"] += 1
        if r["correct"]:
            year_stats[r["prova"]]["correct"] += 1

    print("Acurácia por ano:")
    for year in sorted(year_stats):
        s = year_stats[year]
        acc = s["correct"] / s["total"] * 100
        print(f"  {year:<35} {acc:5.1f}% ({s['correct']}/{s['total']})")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Forgetting evaluation: DrBodeBench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--reasoning_effort", default=None, help="Reasoning effort (none/low/medium/high)")
    parser.add_argument("--parallel", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode (Qwen3.5)")
    parser.add_argument("--output", default=None, help="Salvar JSON com respostas")
    args = parser.parse_args()

    ds = load_dataset("recogna-nlp/drbodebench", split="train")
    rows = list(ds)

    total_raw = len(rows)
    img_count = sum(1 for r in rows if r["contains_img"])
    img_with_desc = sum(1 for r in rows if r["contains_img"] and r["img_description"])
    img_no_desc = img_count - img_with_desc
    four_choice = sum(1 for r in rows if r["alternativas"].get("E") is None)

    if args.limit:
        rows = rows[: args.limit]

    print(f"Modelo: {args.model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    print(f"Dataset: recogna-nlp/drbodebench")
    print(f"Questões: {len(rows)} (total: {total_raw}, com imagem: {img_count} [{img_with_desc} com descrição, {img_no_desc} sem], 4-alternativas: {four_choice})")
    print(f"Max tokens: {args.max_tokens}")
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
                "dataset": "recogna-nlp/drbodebench",
                "mode": "cot",
                "system_prompt": SYSTEM_PROMPT,
                "total_questions": total_raw,
                "evaluated": len(rows),
                "image_questions": img_count,
                "image_with_description": img_with_desc,
                "image_without_description": img_no_desc,
                "four_choice_questions": four_choice,
                "five_choice_questions": total_raw - four_choice,
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nRespostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
