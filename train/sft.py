"""
Supervised fine-tuning (SFT) with assistant-only loss on clinical protocol data.

Trains on chat-formatted data (system/user/assistant messages) with loss computed
only on assistant tokens. Based on cpt.py infrastructure.

Usage:
  accelerate launch train/sft.py \
      --dataset outputs/datasets/sft-gpt52-high-train.jsonl \
      --model_name hugo/clinical-protocols-cpt-gpt41mini-14b \
      --output_dir outputs/models/sft-1
"""

import re
import os
import json
import time
import math
import torch
import argparse
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    get_scheduler,
    set_seed,
)
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin, GradientAccumulationPlugin
from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = (
    "Você está sendo avaliado em um benchmark de VERDADEIRO ou FALSO (pt-BR).\n"
    "Explique brevemente se quiser, MAS OBRIGATORIAMENTE termine a mensagem com uma linha final EXATA:\n"
    "Resposta: Verdadeiro\n"
    "ou\n"
    "Resposta: Falso\n"
    "A linha 'Resposta:' DEVE aparecer exatamente assim, com 'Verdadeiro' ou 'Falso' no fim."
)

VER_REGEX = re.compile(r"\b(verdadeiro|falso)\b", re.IGNORECASE)


def check_correct(text, label):
    matches = list(VER_REGEX.finditer(text or ""))
    if not matches:
        return 0, 0, True
    last = matches[-1].group(1).strip().lower()
    prediction = 1 if last.startswith("v") else 0
    return prediction, prediction == label, False


def load_healthbench_questions(split="test"):
    ds = load_dataset("hugo/healthbench-br-v1", split=split)
    return [{"question": row["pergunta"], "answer": row["resposta"] == "Verdadeiro"} for row in ds]


def load_dev_qa_questions(split="test"):
    ds = load_dataset("hugo/health-qa-br-v1", split=split)
    questions = []
    for row in ds:
        for q in row["questions"]:
            questions.append(f'Pergunta: {q["question"]}\nResposta: {q["answer"]}')
    return questions


@contextmanager
def left_padding(tokenizer):
    original = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = original



def tokenize_chat(example, tokenizer, max_length):
    """Tokenize messages with labels only on assistant tokens."""
    messages = example["messages"]

    # Full conversation
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Prompt (everything except last assistant message)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_enc = tokenizer(full_text, max_length=max_length, truncation=True)
    prompt_enc = tokenizer(prompt_text, max_length=max_length, truncation=True)
    prompt_len = len(prompt_enc["input_ids"])

    # Labels: -100 for prompt tokens, actual ids for assistant tokens
    labels = [-100] * prompt_len + full_enc["input_ids"][prompt_len:]

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def collate_sft(batch, pad_token_id):
    """Pad and collate SFT batch."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    max_len = ((max_len + 63) // 64) * 64

    input_ids = []
    attention_mask = []
    labels = []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT with assistant-only loss")
    parser.add_argument("--dataset", type=str, required=True, help="JSONL file with messages")
    parser.add_argument("--model_name", type=str, default="hugo/clinical-protocols-cpt-gpt41mini-14b")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_warmup_with_min_lr")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--logging_steps", type=float, default=0.02)
    parser.add_argument("--gpu_flops", type=float, default=989e12)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_only_last_epoch", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False, help="Skip intra-training evals")
    parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")

    # Load eval data
    hb_questions = {
        "train": load_healthbench_questions("train"),
        "test": load_healthbench_questions("test"),
    }
    dev_questions = {
        "train": load_dev_qa_questions("train"),
        "test": load_dev_qa_questions("test"),
    }

    parallelism_config = ParallelismConfig(dp_shard_size=torch.cuda.device_count())
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        state_dict_type="SHARDED_STATE_DICT",
        mixed_precision_policy=torch.distributed.fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        ),
        parallelism_config=parallelism_config,
        fsdp_plugin=fsdp_plugin,
    )

    # Load and tokenize dataset
    with accelerator.main_process_first():
        data = load_dataset("json", data_files=args.dataset, split="train")
        accelerator.print(f"  Loaded {len(data)} examples from {args.dataset}")
        train_dataset = data.map(
            partial(tokenize_chat, tokenizer=tokenizer, max_length=args.max_length),
            remove_columns=data.column_names,
            num_proc=min(os.cpu_count() or 4, 8),
        )

    _num_workers = min(os.cpu_count() or 4, 8)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=partial(collate_sft, pad_token_id=tokenizer.pad_token_id),
        pin_memory=True,
        num_workers=_num_workers,
        persistent_workers=True,
        prefetch_factor=4,
    )

    orig_config_dict, _ = PretrainedConfig.get_config_dict(args.model_name)

    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        attn_implementation="kernels-community/flash-attn3",
        use_cache=False,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if args.lora_r > 0:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    no_decay = ["bias", "norm"]
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in trainable_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in trainable_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = math.ceil(total_train_steps * args.warmup_ratio)
    log_steps = max(math.floor(total_train_steps * args.logging_steps), 1)
    model_num_parameters = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_train_steps,
        scheduler_specific_kwargs={"min_lr_rate": 0.1},
    )

    eval_history = []

    @torch.no_grad()
    def evaluate_healthbench(questions, label=""):
        model.eval()
        torch.cuda.empty_cache()
        gen_model = accelerator.unwrap_model(model)
        gen_model.config.use_cache = True
        total = correct = not_found_count = 0
        local_questions = questions[accelerator.process_index::accelerator.num_processes]
        with left_padding(tokenizer):
            for i in tqdm(range(0, len(local_questions), args.eval_batch_size),
                          disable=not accelerator.is_local_main_process, desc=f"HealthBench-BR [{label}]"):
                batch = local_questions[i:i + args.eval_batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["question"]}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    ) for ex in batch
                ]
                answers = [ex["answer"] for ex in batch]
                tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                outputs = gen_model.generate(**tokens, max_new_tokens=1000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                for j, ex in enumerate(batch):
                    _, is_correct, nf = check_correct(
                        tokenizer.decode(outputs[j]).split("<|im_start|>assistant\n")[1], answers[j])
                    correct += is_correct
                    total += 1
                    not_found_count += int(nf)
        global_correct = accelerator.reduce(torch.tensor([correct], device=accelerator.device, dtype=torch.int64), reduction="sum").item()
        global_total = accelerator.reduce(torch.tensor([total], device=accelerator.device, dtype=torch.int64), reduction="sum").item()
        global_nf = accelerator.reduce(torch.tensor([not_found_count], device=accelerator.device, dtype=torch.int64), reduction="sum").item()
        accuracy = global_correct / global_total
        accelerator.print(f"\nHealthBench-BR [{label}] Accuracy: {accuracy * 100:.1f} ({global_correct}/{global_total}) - no answer: {global_nf}\n")
        torch.cuda.empty_cache()
        model.train()
        gen_model.config.use_cache = False
        return accuracy

    @torch.no_grad()
    def evaluate_qa_loss(questions, label=""):
        model.eval()
        torch.cuda.empty_cache()
        local_questions = questions[accelerator.process_index::accelerator.num_processes]
        loss_tok_sum = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
        ntok_sum = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)
        with left_padding(tokenizer):
            for i in tqdm(range(0, len(local_questions), args.eval_batch_size),
                          disable=not accelerator.is_local_main_process, desc=f"Dev QA Loss [{label}]"):
                prompts = local_questions[i:i + args.eval_batch_size]
                tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
                offsets = tokens.pop("offset_mapping")
                tokens = tokens.to(model.device)
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                for b, text in enumerate(prompts):
                    ans_pos = text.rfind("Resposta:")
                    if ans_pos == -1:
                        labels[b, :] = -100
                        continue
                    answer_start_char = ans_pos + len("Resposta:")
                    if answer_start_char < len(text) and text[answer_start_char] == " ":
                        answer_start_char += 1
                    before = offsets[b, :, 1] <= answer_start_char
                    labels[b, before] = -100
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ntok = (labels != -100).sum().to(torch.float32)
                loss_tok_sum += out.loss.detach() * ntok
                ntok_sum += ntok
        total_loss_tok = accelerator.reduce(loss_tok_sum, reduction="sum").item()
        total_ntok = accelerator.reduce(ntok_sum, reduction="sum").item()
        avg_loss = total_loss_tok / max(total_ntok, 1.0)
        accelerator.print(f"\nDev QA loss [{label}] (answer-only): {avg_loss:.3f}\n")
        torch.cuda.empty_cache()
        model.train()
        return avg_loss

    def run_all_evals(epoch):
        results = {"epoch": epoch}
        for split in ["train", "test"]:
            results[f"healthbench_br_{split}"] = evaluate_healthbench(hb_questions[split], label=split)
            results[f"dev_qa_{split}"] = evaluate_qa_loss(dev_questions[split], label=split)
        if accelerator.is_main_process:
            eval_history.append(results)

    if not args.skip_eval:
        run_all_evals(epoch=0)

    accelerator.print("***** Running SFT *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {args.num_train_epochs}")
    accelerator.print(f"  Micro batch size = {args.train_batch_size}")
    accelerator.print(f"  Total batch size = {args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    accelerator.print(f"  Total optimization steps = {total_train_steps}")
    accelerator.print(f"  Learning rate = {args.learning_rate:.0e}")

    total_flops = args.gpu_flops * accelerator.num_processes
    interval_tokens = 0
    global_step = 0
    iter_start = time.perf_counter()

    with tqdm(total=total_train_steps, disable=not accelerator.is_local_main_process) as pbar:
        for epoch in range(1, args.num_train_epochs + 1):
            model.train()

            for step, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(accelerator.device, non_blocking=True)
                labels = batch["labels"].to(accelerator.device, non_blocking=True)
                interval_tokens += input_ids.shape[0] * input_ids.shape[1]

                with accelerator.accumulate(model):
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    ).loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        pbar.update(1)

                        if global_step % log_steps == 0:
                            torch.cuda.synchronize()
                            local_tokens = torch.tensor([interval_tokens], device=accelerator.device, dtype=torch.int64)
                            global_tokens = accelerator.reduce(local_tokens, reduction="sum").item()
                            total_time = time.perf_counter() - iter_start
                            tokens_per_second = global_tokens / total_time
                            mfu = (tokens_per_second * 6 * model_num_parameters) / total_flops
                            lr = optimizer.param_groups[0]["lr"]
                            interval_tokens = 0
                            iter_start = time.perf_counter()
                            accelerator.print(
                                f"Loss: {loss.detach().float().cpu().item():.3f} | "
                                f"Tokens/s: {tokens_per_second:.0f} | "
                                f"MFU: {mfu * 100:.1f}% | "
                                f"Grad norm: {grad_norm.item():.2f} | LR: {lr:.1e}"
                            )

            if not args.skip_eval:
                run_all_evals(epoch=epoch)

            if not args.save_only_last_epoch or epoch == args.num_train_epochs:
                accelerator.print(f"Saving epoch {epoch}...")
                save_dir = f"{args.output_dir}/checkpoint-{epoch}"
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = accelerator.get_state_dict(model, unwrap=False)
                unwrapped_model.save_pretrained(
                    save_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=state_dict,
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_dir)
                    tc_path = f"{save_dir}/tokenizer_config.json"
                    tc = json.load(open(tc_path))
                    if isinstance(tc.get("extra_special_tokens"), list):
                        tc["extra_special_tokens"] = {}
                        json.dump(tc, open(tc_path, "w"), indent=2)
                    json.dump(orig_config_dict, open(f"{save_dir}/config.json", "w"), indent=2)
                    args_dump = {
                        **vars(args),
                        "num_examples": len(train_dataset),
                        "total_batch_size": args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
                        "total_train_steps": total_train_steps,
                    }
                    json.dump(args_dump, open(f"{save_dir}/args.json", "w"), indent=2)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(eval_history, open(os.path.join(args.output_dir, "eval_history.json"), "w"), indent=2)
