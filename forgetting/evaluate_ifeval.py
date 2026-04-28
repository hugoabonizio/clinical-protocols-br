#!/usr/bin/env python3
"""
Avaliação de catastrophic forgetting: IFEval (Instruction Following Evaluation).

Mede a capacidade do modelo de seguir instruções verificáveis (formato, keywords,
tamanho, etc.) usando o benchmark IFEval (541 prompts, 25 tipos de instrução).

Referência: Zhou et al., "Instruction-Following Evaluation for Large Language
Models" (arXiv:2311.07911).

Uso:
  python evaluate_ifeval.py --model qwen-base --base_url http://localhost:8000/v1
  python evaluate_ifeval.py --model gpt-4.1 --api_key sk-...
"""

import dataclasses
import json
import sys
import asyncio
import argparse
from collections import defaultdict
from typing import Dict, Optional, Union

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from ifeval_lib import instructions_registry


# ---------------------------------------------------------------------------
# IFEval evaluation dataclasses and functions (adapted from lm-eval-harness)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(inp: InputExample, response: str) -> OutputExample:
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        # Filter None values (but keep 0, False, etc.)
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)
        is_following_list.append(
            bool(response.strip() and instruction.check_following(response))
        )
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp: InputExample, response: str) -> OutputExample:
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response, revised_response,
        response_remove_first, response_remove_last, response_remove_both,
        revised_response_remove_first, revised_response_remove_last,
        revised_response_remove_both,
    ]
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)
        is_following = any(
            r.strip() and instruction.check_following(r) for r in all_responses
        )
        is_following_list.append(is_following)
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

async def generate_responses(client: AsyncOpenAI, rows: list[dict], args) -> list[dict]:
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            try:
                kwargs = dict(
                    model=args.model,
                    messages=[{"role": "user", "content": row["prompt"]}],
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
            return {
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
                "response": text,
            }

    tasks = [call(r) for r in rows]
    return await tqdm_asyncio.gather(*tasks, desc="IFEval")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_responses(results: list[dict]) -> list[dict]:
    for r in results:
        inp = InputExample(
            key=r["key"],
            instruction_id_list=r["instruction_id_list"],
            prompt=r["prompt"],
            kwargs=r["kwargs"],
        )
        out_strict = test_instruction_following_strict(inp, r["response"])
        out_loose = test_instruction_following_loose(inp, r["response"])

        r["strict_follow_all"] = out_strict.follow_all_instructions
        r["strict_follow_list"] = out_strict.follow_instruction_list
        r["loose_follow_all"] = out_loose.follow_all_instructions
        r["loose_follow_list"] = out_loose.follow_instruction_list
    return results


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    strict_prompt = sum(1 for r in results if r["strict_follow_all"])
    loose_prompt = sum(1 for r in results if r["loose_follow_all"])

    all_strict = [v for r in results for v in r["strict_follow_list"]]
    all_loose = [v for r in results for v in r["loose_follow_list"]]

    return {
        "prompt_strict_acc": strict_prompt / n if n else 0,
        "prompt_loose_acc": loose_prompt / n if n else 0,
        "inst_strict_acc": sum(all_strict) / len(all_strict) if all_strict else 0,
        "inst_loose_acc": sum(all_loose) / len(all_loose) if all_loose else 0,
        "n_prompts": n,
        "n_instructions": len(all_strict),
    }


def print_report(results: list[dict]):
    metrics = compute_metrics(results)
    errors = sum(1 for r in results if r["response"].startswith("[ERRO]"))

    print()
    print("=" * 60)
    print("IFEval — RESULTADOS")
    print("=" * 60)
    n = metrics["n_prompts"]
    ni = metrics["n_instructions"]
    ps = metrics["prompt_strict_acc"]
    pl = metrics["prompt_loose_acc"]
    is_ = metrics["inst_strict_acc"]
    il = metrics["inst_loose_acc"]
    print(f"  Prompt-level strict:       {ps*100:5.1f}% ({int(ps*n)}/{n})")
    print(f"  Prompt-level loose:        {pl*100:5.1f}% ({int(pl*n)}/{n})")
    print(f"  Instruction-level strict:  {is_*100:5.1f}% ({int(is_*ni)}/{ni})")
    print(f"  Instruction-level loose:   {il*100:5.1f}% ({int(il*ni)}/{ni})")
    if errors:
        print(f"\n  Erros de API: {errors}")
    print("=" * 60)

    # Per instruction type breakdown (strict)
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        for inst_id, followed in zip(r["instruction_id_list"], r["strict_follow_list"]):
            type_stats[inst_id]["total"] += 1
            if followed:
                type_stats[inst_id]["correct"] += 1

    print("\nPor tipo de instrução (strict):")
    for inst_type in sorted(type_stats, key=lambda t: type_stats[t]["total"], reverse=True):
        s = type_stats[inst_type]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {inst_type:45s} {acc:5.1f}% ({s['correct']}/{s['total']})")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Forgetting evaluation: IFEval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--reasoning_effort", default=None, help="Reasoning effort (none/low/medium/high)")
    parser.add_argument("--parallel", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Limitar N amostras")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode (Qwen3.5)")
    parser.add_argument("--output", default=None, help="Salvar JSON com respostas")
    args = parser.parse_args()

    print(f"Modelo: {args.model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print()

    print("Carregando dataset IFEval...")
    ds = load_dataset("google/IFEval", split="train")
    rows = list(ds)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  Prompts: {len(rows)}")

    total_instructions = sum(len(r["instruction_id_list"]) for r in rows)
    print(f"  Instruções: {total_instructions}")
    print()

    client = AsyncOpenAI(
        api_key=args.api_key or "no-key",
        base_url=args.base_url,
    )

    # Phase 1: Generate responses
    print("=== Gerando respostas ===")
    results = await generate_responses(client, rows, args)

    # Fail-fast check
    errors = [r for r in results if r["response"].startswith("[ERRO]")]
    if errors and len(errors) == len(results):
        print(f"\nERRO: Todas as {len(results)} respostas falharam. Primeira mensagem:")
        print(f"  {errors[0]['response'][:200]}")
        sys.exit(1)

    # Phase 2: Evaluate locally
    print("=== Avaliando instruções ===")
    results = evaluate_responses(results)

    # Phase 3: Report
    print_report(results)

    if args.output:
        metrics = compute_metrics(results)
        out = {
            "metadata": {
                "model": args.model,
                "base_url": args.base_url,
                "dataset": "google/IFEval",
                "total_prompts": len(rows),
                "metrics": metrics,
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Respostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
