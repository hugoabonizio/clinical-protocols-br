"""
Generate augmented training data from clinical protocol texts.

Reads OPENAI_API_KEY (or OPENROUTER_API_KEY) from the environment.

Usage:
  export OPENAI_API_KEY=sk-...
  python generate.py --output augmented.jsonl --prompt rephrase --model gpt-4.1-mini --num-generations 10
  python generate.py --output augmented.jsonl --prompt wiki --split test --model gpt-5-nano --reasoning-effort low
"""

import os
import json
import time
import argparse
import concurrent.futures
from textwrap import dedent

from openai import OpenAI
from datasets import load_dataset


PROMPTS = {
    "rephrase": dedent("""\
        A partir do texto do protocolo clínico fornecido abaixo, reescreva modificando a ordem das informações e refraseando partes do texto. Modifique um pouco o estilo de escrita, mas mantenha as mesmas informações originais.

        Título: {title}
        Texto:
        {text}"""),

    "wiki": dedent("""\
        Gere um artigo informativo no formato da Wikipedia com base no protocolo clínico abaixo. Explique termos técnicos e deixe as informações claras tanto para leigos quanto para profissionais. Assegure-se de incluir todas as informações importantes, incluindo números e valores, presentes no texto original.

        Texto:
        {text}"""),

    "questions": dedent("""\
        Gere perguntas e respostas partir do texto do protocolo clínico fornecido abaixo. As perguntas devem extrair parte do texto e citar de maneira literal e questionar sobre informações do texto. As respostas devem ser detalhadas e mostrar o passo a passo até chegar à resposta final. É importante que toda a cadeia de pensamento esteja detalhada na resposta e que ela seja embasada no texto original. Faça tanto perguntas mais gerais quanto específica sobre números ou medidas importantes.

        Texto:
        {text}"""),
}


def generate_one(client, example, prompt_template, model, max_tokens, temperature,
                 num_generations, reasoning_effort=None, max_retries=30, wait_time=2):
    prompt = prompt_template.format(**example)
    messages = [{"role": "user", "content": prompt}]

    extra = {}
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort

    def single_request():
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=1,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    **extra,
                )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"Exception: {e}. Attempt {attempt + 1}/{max_retries}.", flush=True)
                time.sleep(wait_time)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_generations) as executor:
        futures = [executor.submit(single_request) for _ in range(num_generations)]
        texts = [f.result() for f in concurrent.futures.as_completed(futures)]

    return {**example, "texts": [t for t in texts if t is not None]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented training data from clinical protocols")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--prompt", choices=PROMPTS.keys(), required=True, help="Augmentation type")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name")
    parser.add_argument("--dataset", default="hugo/protocolos-clinicos-v1", help="HuggingFace dataset")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--num-generations", type=int, default=10, help="Generations per input")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Max completion tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort (for o-series and gpt-5 models)")
    parser.add_argument("--batch-size", type=int, default=10, help="Parallel workers")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY (or OPENROUTER_API_KEY) in your environment.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    template = PROMPTS[args.prompt]

    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.rename_columns({"titulo": "title", "texto": "text"})
    print(f"Dataset: {args.dataset} [{args.split}] — {len(dataset)} protocols")
    reasoning_str = f" | Reasoning: {args.reasoning_effort}" if args.reasoning_effort else ""
    print(f"Prompt: {args.prompt} | Model: {args.model} | Generations: {args.num_generations}{reasoning_str}\n")

    def process(example):
        return generate_one(
            client, example,
            prompt_template=template,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_generations=args.num_generations,
            reasoning_effort=args.reasoning_effort,
        )

    results = []
    total_texts = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {executor.submit(process, row): i for i, row in enumerate(dataset)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            total_texts += len(result["texts"])
            print(f"  [{len(results)}/{len(dataset)}] {result['title'][:60]}... ({len(result['texts'])} texts)", flush=True)

    with open(args.output, "w") as f:
        for row in results:
            for text in row["texts"]:
                f.write(json.dumps({"title": row["title"], "text": text}, ensure_ascii=False) + "\n")

    print(f"\nDone. {total_texts} examples written to {args.output}")
