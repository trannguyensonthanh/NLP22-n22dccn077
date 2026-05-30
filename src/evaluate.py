"""
Evaluate all three models on the test set.

Metrics:
  - Perplexity      : how well the model assigns probability to test data
  - Top-k accuracy  : fraction of next-word predictions where the true word
                      appears in the model's top-k suggestions (k=1, 3, 5).

Usage:
    python -m src.evaluate
"""
import math
from pathlib import Path

import torch
from tqdm import tqdm

from . import finetune_gpt2, neural_model, ngram_model
# Make Vocab resolvable at unpickle time
from .neural_model import Vocab  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "processed" / "test.txt"


def load_test_sentences(limit=None):
    with TEST.open(encoding="utf-8") as f:
        sents = [line.strip().split() for line in f if line.strip()]
    return sents[:limit] if limit else sents


# --- N-gram -----------------------------------------------------------------

def eval_ngram(order=4, limit=200):
    """Perplexity only — top-k accuracy with KN over 200k vocab is both
    slow and uninformative for our setting."""
    print(f"\n=== N-gram (order={order}) ===", flush=True)
    model = ngram_model.load_model(order)
    sents = load_test_sentences(limit=limit)

    total_logp, total_tokens = 0.0, 0
    for sent in tqdm(sents):
        for i in range(1, len(sent)):
            context = tuple(sent[max(0, i - (order - 1)):i])
            p = model.score(sent[i], context)
            if p > 0:
                total_logp += math.log(p)
                total_tokens += 1

    ppl = math.exp(-total_logp / max(total_tokens, 1))
    print(f"  perplexity: {ppl:.2f}  (on {total_tokens} tokens)", flush=True)
    return {"ppl": ppl, "top1": None, "top3": None, "top5": None}


# --- LSTM -------------------------------------------------------------------

@torch.no_grad()
def eval_lstm(limit=500, ks=(1, 3, 5)):
    print("\n=== LSTM ===", flush=True)
    model, vocab, device = neural_model.load_model()
    sents = load_test_sentences(limit=limit)
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total_loss, total_tokens, preds_total = 0.0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    for sent in tqdm(sents):
        tokens = [neural_model.BOS] + sent + [neural_model.EOS]
        ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)
        inp, tgt = ids[:, :-1], ids[:, 1:]
        logits, _ = model(inp)
        total_loss += loss_fn(logits.reshape(-1, logits.size(-1)),
                              tgt.reshape(-1)).item()
        total_tokens += (tgt != 0).sum().item()

        # top-k from same forward pass (positions correspond to next-word preds)
        topk = torch.topk(logits[0], max_k, dim=-1).indices.tolist()
        for pos in range(min(len(sent), logits.size(1))):
            true_id = tgt[0, pos].item()
            if true_id == 0:
                continue
            top_ids = topk[pos]
            for k in ks:
                if true_id in top_ids[:k]:
                    hits[k] += 1
            preds_total += 1

    ppl = math.exp(total_loss / max(total_tokens, 1))
    accs = {k: hits[k] / max(preds_total, 1) for k in ks}
    print(f"  perplexity: {ppl:.2f}", flush=True)
    for k in ks:
        print(f"  top-{k} acc: {accs[k]:.4f}", flush=True)
    return {"ppl": ppl, "top1": accs[1], "top3": accs[3], "top5": accs[5]}


# --- GPT-2 ------------------------------------------------------------------

@torch.no_grad()
def eval_gpt2(limit=300, ks=(1, 3, 5)):
    """Top-k done by matching first-token-of-true-word in BPE space.
    This is much more reliable than decoding subword predictions back to
    whitespace-tokenised words."""
    print("\n=== GPT-2 (fine-tuned) ===", flush=True)
    model, tokenizer, device = finetune_gpt2.load_model()
    sents = load_test_sentences(limit=limit)
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total_loss, total_tokens, preds_total = 0.0, 0, 0

    for sent in tqdm(sents):
        text = " ".join(sent)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if ids.size(1) < 2:
            continue
        out = model(ids, labels=ids)
        total_loss += out.loss.item() * (ids.size(1) - 1)
        total_tokens += ids.size(1) - 1

        # Token-level top-k: for each true BPE token in the sequence, is it in
        # the model's top-k? This matches what an autocomplete UI surfaces.
        logits = out.logits[0]  # (T, V)
        topk = torch.topk(logits[:-1], max_k, dim=-1).indices  # predicts ids[1:]
        target = ids[0, 1:]
        for k in ks:
            hits[k] += (topk[:, :k] == target.unsqueeze(-1)).any(-1).sum().item()
        preds_total += target.size(0)

    ppl = math.exp(total_loss / max(total_tokens, 1))
    accs = {k: hits[k] / max(preds_total, 1) for k in ks}
    print(f"  perplexity: {ppl:.2f}", flush=True)
    for k in ks:
        print(f"  top-{k} acc: {accs[k]:.4f}", flush=True)
    return {"ppl": ppl, "top1": accs[1], "top3": accs[3], "top5": accs[5]}


def main():
    results = {}
    for name, fn in [("ngram", eval_ngram),
                     ("lstm", eval_lstm),
                     ("gpt2", eval_gpt2)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[skip] {name}: {e}", flush=True)
            import traceback; traceback.print_exc()

    print("\n=== Summary ===", flush=True)
    print(f"{'Model':<10} {'PPL':>10} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}",
          flush=True)
    for name, r in results.items():
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "  n/a "
        print(f"{name:<10} {r['ppl']:>10.2f} "
              f"{fmt(r['top1']):>8} {fmt(r['top3']):>8} {fmt(r['top5']):>8}",
              flush=True)


if __name__ == "__main__":
    main()
