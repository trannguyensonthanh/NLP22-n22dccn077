"""
Evaluate models on LAMBADA — the standard benchmark for next-word prediction.

LAMBADA contains passages where the goal is to predict the LAST word given
the entire preceding context. Crucially, the target word can almost never
be guessed from local context alone — you need to read the whole passage.
This is the gold standard for "how well does the model capture discourse
for autocomplete?" and is what major LM papers (GPT-2, GPT-3, ...) report.

Reference: Paperno et al. 2016, https://arxiv.org/abs/1606.06031

Usage:  python -m src.eval_lambada
"""
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def load_lambada(limit=None):
    """LAMBADA test split. Each ex: {'text': '<passage>'} — predict last word."""
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def split_context_target(text: str):
    parts = text.rstrip().rsplit(" ", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1].rstrip(".!?,;:\"'")


# --- N-gram -----------------------------------------------------------------

def eval_ngram_lambada(limit=500, order=4):
    from . import ngram_model
    ngram_model._ensure_nltk()
    from nltk.tokenize import word_tokenize

    print(f"\n=== N-gram (order={order}) on LAMBADA ===")
    model = ngram_model.load_model(order)
    ds = load_lambada(limit=limit)
    top1 = top5 = 0
    for ex in tqdm(ds):
        ctx, tgt = split_context_target(ex["text"])
        if not tgt:
            continue
        tokens = word_tokenize(ctx.lower())
        preds = ngram_model.predict_next(model, tokens, top_k=5)
        words = [w for w, _ in preds]
        if words and words[0] == tgt.lower():
            top1 += 1
        if tgt.lower() in words:
            top5 += 1
    n = len(ds)
    print(f"  top-1: {top1 / n:.4f}   top-5: {top5 / n:.4f}   (n={n})")
    return top1 / n, top5 / n


# --- LSTM -------------------------------------------------------------------

@torch.no_grad()
def eval_lstm_lambada(limit=500):
    from . import neural_model
    print("\n=== LSTM on LAMBADA ===")
    model, vocab, device = neural_model.load_model()
    ds = load_lambada(limit=limit)
    top1 = top5 = 0
    for ex in tqdm(ds):
        ctx, tgt = split_context_target(ex["text"])
        if not tgt:
            continue
        preds = neural_model.predict_next(model, vocab, device,
                                          ctx.lower(), top_k=5)
        words = [w for w, _ in preds]
        if words and words[0] == tgt.lower():
            top1 += 1
        if tgt.lower() in words:
            top5 += 1
    n = len(ds)
    print(f"  top-1: {top1 / n:.4f}   top-5: {top5 / n:.4f}   (n={n})")
    return top1 / n, top5 / n


# --- GPT-2 ------------------------------------------------------------------

@torch.no_grad()
def eval_gpt2_lambada(limit=500):
    from . import finetune_gpt2
    print("\n=== GPT-2 fine-tuned on LAMBADA ===")
    model, tokenizer, device = finetune_gpt2.load_model()
    ds = load_lambada(limit=limit)
    top1 = top5 = 0
    for ex in tqdm(ds):
        ctx, tgt = split_context_target(ex["text"])
        if not tgt:
            continue
        # No trailing space: GPT-2 BPE expects to predict the space-prefixed
        # next-word token ("Ġword"). Appending " " inserts a dangling space
        # token that pushes the model off-distribution and yields subword junk.
        preds = finetune_gpt2.predict_next(model, tokenizer, device,
                                           ctx, top_k=5)
        words = [w.lower() for w, _ in preds]
        if words and words[0] == tgt.lower():
            top1 += 1
        if tgt.lower() in words:
            top5 += 1
    n = len(ds)
    print(f"  top-1: {top1 / n:.4f}   top-5: {top5 / n:.4f}   (n={n})")
    return top1 / n, top5 / n


def main():
    results = {}
    for name, fn in [("ngram", eval_ngram_lambada),
                     ("lstm",  eval_lstm_lambada),
                     ("gpt2",  eval_gpt2_lambada)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[skip] {name}: {e}")

    print("\n=== LAMBADA Summary ===")
    print(f"{'Model':<10} {'Top-1':>8} {'Top-5':>8}")
    for name, (t1, t5) in results.items():
        print(f"{name:<10} {t1:>8.4f} {t5:>8.4f}")


if __name__ == "__main__":
    main()
