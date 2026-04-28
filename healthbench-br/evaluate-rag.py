#!/usr/bin/env python3
"""
Avaliação do HealthBench-BR com RAG (oracle ou chunks) via API OpenAI-compatible.

Modos:
  --rag oracle      Injeta o documento completo do protocolo no prompt
  --rag bm25        Injeta top_k chunks via BM25
  --rag embeddings  Injeta top_k chunks via embeddings OpenAI

Uso:
  python evaluate-rag.py --model cpt3 --base_url http://host:8000/v1 --rag oracle
  python evaluate-rag.py --model cpt3 --base_url http://host:8000/v1 --rag bm25 --rag_top_k 10
"""

import re
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Add project root to path for rag_utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from rag_utils import RAGRetriever, build_oracle_lookup, make_rag_prompt

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


async def evaluate_split(
    client: AsyncOpenAI,
    split_data,
    args,
    oracle_lookup=None,
    retriever=None,
) -> list[dict]:
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            user_content = make_rag_prompt(
                question=row["pergunta"],
                row_titulo=row.get("titulo", ""),
                row_arquivo=row.get("protocolo", ""),
                rag_mode=args.rag,
                oracle_lookup=oracle_lookup,
                retriever=retriever,
                top_k=args.rag_top_k,
                max_context_chars=args.max_context_chars,
            )
            try:
                if args.rag_in_system:
                    # Split: context goes to system, question stays in user
                    from rag_utils import get_rag_context
                    context = get_rag_context(
                        row_titulo=row.get("titulo", ""),
                        row_arquivo=row.get("protocolo", ""),
                        rag_mode=args.rag,
                        oracle_lookup=oracle_lookup,
                        retriever=retriever,
                        top_k=args.rag_top_k,
                        max_context_chars=args.max_context_chars,
                    )
                    sys_content = SYSTEM_PROMPT + "\n\n" + context
                    user_content = row["pergunta"]
                else:
                    sys_content = SYSTEM_PROMPT
                kwargs = dict(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user_content},
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
            pred = parse_label(text)
            return {
                "pergunta_id": row["pergunta_id"],
                "protocolo": row["protocolo"],
                "pergunta": row["pergunta"],
                "esperado": row["resposta"],
                "pred": pred,
                "correto": pred == row["resposta"],
                "resposta_bruta": text,
                "rag_mode": args.rag,
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

    print()
    if train_res:
        tc, tn, tna = acc(train_res)
        print(f"Acurácia Treino:  {tc/tn*100:.1f}% ({tc}/{tn}) - sem resposta: {tna}")
    if test_res:
        ec, en, ena = acc(test_res)
        print(f"Acurácia Teste:   {ec/en*100:.1f}% ({ec}/{en}) - sem resposta: {ena}")
    all_res = train_res + test_res
    if all_res:
        gc, gn, gna = acc(all_res)
        if train_res and test_res:
            print(f"Acurácia Geral:   {gc/gn*100:.1f}% ({gc}/{gn}) - sem resposta: {gna}")
        if gna > 0 and gn > gna:
            answered = gn - gna
            print(f"Acurácia (com resposta): {gc/answered*100:.1f}% ({gc}/{answered})")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Avaliação HealthBench-BR com RAG")
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
    parser.add_argument("--output", default=None, help="Salvar JSON com respostas")
    # RAG args
    parser.add_argument("--rag", required=True, choices=["oracle", "bm25", "embeddings"], help="Modo RAG")
    parser.add_argument("--rag_top_k", type=int, default=5, help="Chunks a recuperar (bm25/embeddings)")
    parser.add_argument("--max_context_chars", type=int, default=80000, help="Truncar documento oracle (chars)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all", help="Splits a avaliar")
    parser.add_argument("--rag_in_system", action="store_true", help="Colocar documento no system message ao invés do user message")
    args = parser.parse_args()

    # Initialize RAG
    oracle_lookup = None
    retriever = None
    if args.rag == "oracle":
        print("Carregando corpus para modo oracle...")
        oracle_lookup = build_oracle_lookup()
    else:
        print(f"Inicializando RAGRetriever ({args.rag}, top_k={args.rag_top_k})...")
        retriever = RAGRetriever(method=args.rag)

    ds = load_dataset(args.dataset)

    client = AsyncOpenAI(
        api_key=args.api_key or "no-key",
        base_url=args.base_url,
    )

    print(f"Modelo: {args.model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    print(f"Dataset: {args.dataset}")
    print(f"RAG: {args.rag}" + (f" (top_k={args.rag_top_k})" if args.rag != "oracle" else ""))
    print(f"Parallelism: {args.parallel}")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print()

    train_res = []
    test_res = []
    if args.split in ("all", "train"):
        print("=== Train ===")
        train_res = await evaluate_split(client, ds["train"], args, oracle_lookup, retriever)
    if args.split in ("all", "test"):
        print("=== Test ===")
        test_res = await evaluate_split(client, ds["test"], args, oracle_lookup, retriever)

    print_report(train_res, test_res)

    if args.output:
        out = {"rag_mode": args.rag}
        if args.rag != "oracle":
            out["rag_top_k"] = args.rag_top_k
        if train_res:
            out["train"] = train_res
        if test_res:
            out["test"] = test_res
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Respostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
