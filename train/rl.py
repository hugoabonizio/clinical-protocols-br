"""
GRPO reinforcement learning on HealthBench-BR true/false questions.
Variant with minimum-length reward: completions under 50 words get reward 0.

Usage:
  torchrun --nproc_per_node=2 train/rl.py \
      --model_name_or_path outputs/models/<cpt_checkpoint> \
      --output_dir outputs/models/rl
"""

import os
import re
import json
from typing import List

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from datasets import Dataset
from transformers import AutoTokenizer

import torch
from datasets import load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)


SYSTEM_PROMPT = (
    "Você está sendo avaliado em um benchmark de VERDADEIRO ou FALSO (pt-BR).\n"
    "Explique brevemente se quiser, MAS OBRIGATORIAMENTE termine a mensagem com uma linha final EXATA:\n"
    "Resposta: Verdadeiro\n"
    "ou\n"
    "Resposta: Falso\n"
    "A linha 'Resposta:' DEVE aparecer exatamente assim, com 'Verdadeiro' ou 'Falso' no fim."
)

VER_REGEX = re.compile(r"\b(verdadeiro|falso)\b", re.IGNORECASE)

MIN_WORDS = 50


def clinical_reward_fn(completions: List[str], label: List[int], **kwargs) -> List[float]:
    """Reward: 1.0 correct (>= 50 words), 0.0 otherwise."""
    rewards = []
    for completion, lbl in zip(completions, label):
        text = completion or ""
        matches = list(VER_REGEX.finditer(text))
        if not matches:
            rewards.append(0.0)
            continue
        last_match = matches[-1].group(1).strip().lower()
        prediction = 1 if last_match.startswith("v") else 0
        if prediction != lbl:
            rewards.append(0.0)
            continue
        n_words = len(text.split())
        if n_words < MIN_WORDS:
            rewards.append(0.0)
            continue
        rewards.append(1.0)
    return rewards


def load_healthbench_dataset(split: str, tokenizer) -> Dataset:
    """Load HealthBench-BR from HuggingFace and format for GRPO training."""
    ds = load_dataset("hugo/healthbench-br-v1", split=split)
    data = []
    for row in ds:
        label = 1 if row["resposta"] == "Verdadeiro" else 0
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["pergunta"]},
        ]
        data.append({
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "label": label,
        })
    return Dataset.from_list(data)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
    )
    training_args.model_init_kwargs = dict(
        attn_implementation="kernels-community/flash-attn3",
        dtype="auto",
    )
    training_args.use_liger_kernel = True

    train_dataset = load_healthbench_dataset("train", tokenizer)
    print(f"Loaded {len(train_dataset)} training examples")

    eval_dataset = load_healthbench_dataset("test", tokenizer)
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(200))
    print(f"Loaded {len(eval_dataset)} eval examples")

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=clinical_reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

    # Save all arguments to JSON
    if trainer.is_world_process_zero():
        import dataclasses
        args_dict = {
            "script_args": dataclasses.asdict(script_args) if dataclasses.is_dataclass(script_args) else vars(script_args),
            "training_args": training_args.to_dict(),
            "model_args": dataclasses.asdict(model_args) if dataclasses.is_dataclass(model_args) else vars(model_args),
        }
        args_path = os.path.join(training_args.output_dir, "train_args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, default=str)
        print(f"Saved training arguments to {args_path}")
