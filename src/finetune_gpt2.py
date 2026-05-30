"""
Fine-tune GPT-2 small on our processed corpus for autocomplete.

This is the strongest of the three models. The pretrained GPT-2 already
writes fluent English; fine-tuning on a curated, well-edited corpus
(WikiText / BBC / arXiv) nudges its style toward "good written English",
which is what we want for a writing-assist autocomplete.

Usage:
    python -m src.finetune_gpt2 train  --epochs 2
    python -m src.finetune_gpt2 predict --prefix "the quick brown"
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "gpt2-finetuned"
MODEL_NAME = "gpt2"


def train(epochs=2, batch_size=8, lr=5e-5, block_size=128):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    data_files = {
        "train": str(PROC / "train_small.txt"),
        "validation": str(PROC / "val_small.txt"),
    }
    raw = load_dataset("text", data_files=data_files)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=block_size)

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Trainer auto-uses all visible GPUs with DataParallel.
    # To restrict to one GPU: export CUDA_VISIBLE_DEVICES=0
    # For DistributedDataParallel (faster), launch with:
    #   torchrun --nproc_per_node=2 -m src.finetune_gpt2 train ...
    # Eval disabled inside training to avoid a CUDA assert seen with
    # eval_strategy="epoch" + fp16 on this transformers version. We
    # save the final model manually below and run evaluation separately.
    args = TrainingArguments(
        output_dir=str(CKPT),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(CKPT))
    tokenizer.save_pretrained(str(CKPT))
    print(f"Saved -> {CKPT}")


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
    model = AutoModelForCausalLM.from_pretrained(str(CKPT)).to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def predict_next(model, tokenizer, device, prefix: str, top_k=5):
    ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, top_k * 4)
    results, seen = [], set()
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        # decode and strip the leading space GPT-2 BPE adds
        word = tokenizer.decode([idx]).strip()
        if not word or word in seen or not any(c.isalpha() for c in word):
            continue
        seen.add(word)
        results.append((word, p))
        if len(results) >= top_k:
            break
    return results


def cli_predict(prefix: str, top_k: int = 5):
    model, tokenizer, device = load_model()
    preds = predict_next(model, tokenizer, device, prefix, top_k)
    print(f"\nPrefix: {prefix!r}")
    print(f"Top-{top_k} next-word predictions:")
    for w, p in preds:
        print(f"  {w:<20} {p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--block-size", type=int, default=128)

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.epochs, args.batch_size, args.lr, args.block_size)
    else:
        cli_predict(args.prefix, args.top_k)
