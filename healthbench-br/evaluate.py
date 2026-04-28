#!/usr/bin/env python3
"""
Avaliação do HealthBench-BR via API OpenAI-compatible.

Carrega o dataset hugo/healthbench-br-v1 da HuggingFace,
envia perguntas V/F para o modelo e reporta acurácia por split.

Uso:
  python evaluate.py --model gpt-5 --api_key sk-...
  python evaluate.py --model lora --api_key "-" --base_url http://host:8000/v1
"""

import re
import json
import asyncio
import argparse

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT = (
    "Você está sendo avaliado em um benchmark de VERDADEIRO ou FALSO (pt-BR).\n"
    "Explique brevemente se quiser, MAS OBRIGATORIAMENTE termine a mensagem com uma linha final EXATA:\n"
    "Resposta: Verdadeiro\n"
    "ou\n"
    "Resposta: Falso\n"
    "A linha 'Resposta:' DEVE aparecer exatamente assim, com 'Verdadeiro' ou 'Falso' no fim."
)

LABEL_RE = re.compile(r"\b(verdadeiro|falso)\b", re.IGNORECASE)


def parse_label(text: str) -> str | None:
    matches = list(LABEL_RE.finditer(text or ""))
    if not matches:
        return None
    last = matches[-1].group(1).lower()
    return "Verdadeiro" if last.startswith("v") else "Falso"


async def evaluate_split(client: AsyncOpenAI, split_data, args) -> list[dict]:
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            try:
                kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row["pergunta"]},
                    ],
                )
                extra_body = {}
                if args.reasoning_effort:
                    extra_body["reasoning_effort"] = args.reasoning_effort
                    extra_body["reasoning"] = {"effort": args.reasoning_effort}
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
            pred = parse_label(text)
            return {
                "pergunta_id": row["pergunta_id"],
                "protocolo": row["protocolo"],
                "pergunta": row["pergunta"],
                "esperado": row["resposta"],
                "pred": pred,
                "correto": pred == row["resposta"],
                "resposta_bruta": text,
            }

    rows = list(split_data)
    if args.limit:
        rows = rows[: args.limit]

    tasks = [call(r) for r in rows]
    results = await tqdm_asyncio.gather(*tasks, desc="Avaliando")
    return results


def print_report(train_res, test_res):
    def acc(results):
        n = len(results)
        c = sum(1 for r in results if r["correto"])
        na = sum(1 for r in results if r["pred"] is None)
        return c, n, na

    tc, tn, tna = acc(train_res)
    ec, en, ena = acc(test_res)
    gc, gn, gna = tc + ec, tn + en, tna + ena

    print()
    if tn > 0:
        print(f"Acurácia Treino:  {tc/tn*100:.1f}% ({tc}/{tn}) - sem resposta: {tna}")
    if en > 0:
        print(f"Acurácia Teste:   {ec/en*100:.1f}% ({ec}/{en}) - sem resposta: {ena}")
    if gn > 0:
        print(f"Acurácia Geral:   {gc/gn*100:.1f}% ({gc}/{gn}) - sem resposta: {gna}")
    if gna > 0:
        answered = gn - gna
        correct_answered = gc
        print(f"Acurácia (com resposta): {correct_answered/answered*100:.1f}% ({correct_answered}/{answered})")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Avaliação HealthBench-BR")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--reasoning_effort", default=None, help="Reasoning effort (none/low/medium/high)")
    parser.add_argument("--parallel", type=int, default=100)
    parser.add_argument("--dataset", default="hugo/healthbench-br-v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode (Qwen3.5)")
    parser.add_argument("--split", default=None, choices=["train", "test"], help="Avaliar apenas um split")
    parser.add_argument("--output", default=None, help="Salvar JSON com respostas")
    args = parser.parse_args()

    ds = load_dataset(args.dataset)

    client = AsyncOpenAI(
        api_key=args.api_key or "no-key",
        base_url=args.base_url,
    )

    print(f"Modelo: {args.model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    print(f"Dataset: {args.dataset}")
    print(f"Parallelism: {args.parallel}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print()

    train_res, test_res = [], []
    if args.split != "test":
        print("=== Train ===")
        train_res = await evaluate_split(client, ds["train"], args)
    if args.split != "train":
        print("=== Test ===")
        test_res = await evaluate_split(client, ds["test"], args)

    print_report(train_res, test_res)

    if args.output:
        out = {"train": train_res, "test": test_res}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Respostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
