#!/usr/bin/env python3
"""
Avaliação do Health-QA-BR (perguntas abertas) com RAG via API OpenAI-compatible.

Modos:
  --rag oracle      Injeta o documento completo do protocolo no prompt
  --rag bm25        Injeta top_k chunks via BM25
  --rag embeddings  Injeta top_k chunks via embeddings OpenAI

Pipeline:
  1. Carrega hugo/health-qa-br-v1 (perguntas abertas por protocolo)
  2. Augmenta prompts com contexto RAG
  3. Gera respostas com o modelo avaliado
  4. Julga respostas com GPT-4.1 (CORRETA/INCORRETA)
  5. Reporta acurácia por split (treino/teste/geral)

Uso:
  python evaluate-rag.py --model cpt3 --base_url http://host:8000/v1 --rag oracle
  python evaluate-rag.py --model cpt3 --base_url http://host:8000/v1 --rag bm25 --rag_top_k 10
"""

import sys
import json
import asyncio
import argparse
from typing import Literal
from pathlib import Path

# Add project root to path for rag_utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
from rag_utils import RAGRetriever, build_oracle_lookup, make_rag_prompt

JUDGE_PROMPT = """Avalie a resposta a seguir considerando a pergunta e a resposta esperada. Compare a resposta gerada com a resposta esperada e determine se a resposta gerada está correta ou incorreta. Forneça apenas "CORRETA" ou "INCORRETA" como resposta.

Pergunta: {question}
Resposta Esperada: {expected_answer}
Resposta Gerada: {generated_answer}"""


class JudgeResponse(BaseModel):
    raciocinio: str
    resposta: Literal["CORRETA", "INCORRETA"]


def flatten_dataset(ds, split_name: str) -> list[dict]:
    """Flatten nested questions into one row per QA pair."""
    rows = []
    for protocol in ds:
        for qa in protocol["questions"]:
            rows.append({
                "title": protocol["title"],
                "question": qa["question"],
                "answer": qa["answer"],
                "split": split_name,
            })
    return rows


async def generate_answers(
    client: AsyncOpenAI,
    rows: list[dict],
    args,
    oracle_lookup=None,
    retriever=None,
) -> list[dict]:
    """Generate model answers for all questions with RAG context."""
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            user_content = make_rag_prompt(
                question=row["question"],
                row_titulo=row.get("title", ""),
                row_arquivo=None,
                rag_mode=args.rag,
                oracle_lookup=oracle_lookup,
                retriever=retriever,
                top_k=args.rag_top_k,
                max_context_chars=args.max_context_chars,
            )
            try:
                kwargs = dict(
                    model=args.model,
                    messages=[{"role": "user", "content": user_content}],
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
                row["model_answer"] = resp.choices[0].message.content or ""
            except Exception as e:
                row["model_answer"] = f"[ERRO] {e}"
            return row

    tasks = [call(r) for r in rows]
    return await tqdm_asyncio.gather(*tasks, desc="Gerando respostas")


async def judge_answers(client: AsyncOpenAI, rows: list[dict], args) -> list[dict]:
    """Judge model answers against expected answers."""
    sem = asyncio.Semaphore(args.parallel)

    async def call(row):
        async with sem:
            prompt = JUDGE_PROMPT.format(
                question=row["question"],
                expected_answer=row["answer"],
                generated_answer=row["model_answer"],
            )
            try:
                resp = await client.beta.chat.completions.parse(
                    model=args.judge_model,
                    temperature=0.0,
                    max_completion_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=JudgeResponse,
                )
                parsed = resp.choices[0].message.parsed
                row["judge"] = parsed.resposta
                row["reasoning"] = parsed.raciocinio
            except Exception as e:
                row["judge"] = "ERRO"
                row["reasoning"] = str(e)
            row["correct"] = row["judge"] == "CORRETA"
            return row

    tasks = [call(r) for r in rows]
    return await tqdm_asyncio.gather(*tasks, desc="Julgando respostas")


def print_report(results: list[dict], model: str):
    def acc(rows):
        n = len(rows)
        c = sum(1 for r in rows if r["correct"])
        return c, n

    train = [r for r in results if r["split"] == "train"]
    test = [r for r in results if r["split"] == "test"]

    print()
    print("=" * 50)
    print("RESULTADOS DA AVALIAÇÃO")
    print(f"Modelo: {model}")
    print("=" * 50)
    if train:
        tc, tn = acc(train)
        print(f"Acurácia Treino:  {tc/tn*100:.1f}% ({tc}/{tn})")
    if test:
        ec, en = acc(test)
        print(f"Acurácia Teste:   {ec/en*100:.1f}% ({ec}/{en})")
    all_res = results
    gc, gn = acc(all_res)
    if train and test:
        print(f"Acurácia Geral:   {gc/gn*100:.1f}% ({gc}/{gn})")
    print("=" * 50)


async def main():
    parser = argparse.ArgumentParser(description="Avaliação Health-QA-BR com RAG (perguntas abertas)")
    parser.add_argument("--model", required=True, help="Modelo a avaliar")
    parser.add_argument("--base_url", default=None, help="Base URL da API do modelo")
    parser.add_argument("--api_key", default=None, help="API key do modelo")
    parser.add_argument("--judge_model", default="gpt-4.1", help="Modelo juiz (default: gpt-4.1)")
    parser.add_argument("--judge_api_key", default=None, help="API key do juiz (default: mesma do modelo)")
    parser.add_argument("--judge_base_url", default=None, help="Base URL do juiz (default: OpenAI)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--reasoning_effort", default=None, help="Reasoning effort (none/low/medium/high)")
    parser.add_argument("--parallel", type=int, default=100)
    parser.add_argument("--dataset", default="hugo/health-qa-br-v1")
    parser.add_argument("--limit", type=int, default=None, help="Limitar N perguntas por split")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode (Qwen3.5)")
    parser.add_argument("--output", default=None, help="Salvar JSON com respostas")
    # RAG args
    parser.add_argument("--rag", required=True, choices=["oracle", "bm25", "embeddings"], help="Modo RAG")
    parser.add_argument("--rag_top_k", type=int, default=5, help="Chunks a recuperar (bm25/embeddings)")
    parser.add_argument("--max_context_chars", type=int, default=80000, help="Truncar documento oracle (chars)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all", help="Splits a avaliar")
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

    # Flatten
    train_rows = flatten_dataset(ds["train"], "train") if args.split in ("all", "train") else []
    test_rows = flatten_dataset(ds["test"], "test") if args.split in ("all", "test") else []

    if args.limit:
        train_rows = train_rows[:args.limit]
        test_rows = test_rows[:args.limit]

    all_rows = train_rows + test_rows

    print(f"Modelo: {args.model}")
    print(f"Juiz: {args.judge_model}")
    print(f"Base URL: {args.base_url or 'default (OpenAI)'}")
    print(f"Dataset: {args.dataset}")
    print(f"RAG: {args.rag}" + (f" (top_k={args.rag_top_k})" if args.rag != "oracle" else ""))
    print(f"Split: {args.split}")
    print(f"Perguntas: {len(train_rows)} train + {len(test_rows)} test = {len(all_rows)} total")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print()

    # Generate answers
    client_gen = AsyncOpenAI(
        api_key=args.api_key or "no-key",
        base_url=args.base_url,
    )
    all_rows = await generate_answers(client_gen, all_rows, args, oracle_lookup, retriever)

    # Judge answers
    client_judge = AsyncOpenAI(
        api_key=args.judge_api_key or args.api_key,
        base_url=args.judge_base_url,
    )
    all_rows = await judge_answers(client_judge, all_rows, args)

    print_report(all_rows, args.model)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        print(f"\nRespostas salvas em: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
