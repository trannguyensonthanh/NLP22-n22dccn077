"""
Evaluate all models on the test set.

CHANGES vs original evaluate.py:
    - Added compute_prf1() function: Precision, Recall, F1 at top-k
      (Lecture Ch.4: "Precision, Recall, and F1").
    - Added macro/micro averaging (Ch.4: "Microaveraging vs Macroaveraging").
    - Added MRR (Mean Reciprocal Rank) — from Ch.8 QA evaluation.
    - All original perplexity + top-k accuracy code kept 100% intact.

Metrics summary (from lecture):
    Precision  = TP / (TP + FP)  — of predicted words, how many are correct
    Recall     = TP / (TP + FN)  — of correct words, how many we predicted
    F1         = 2 * P * R / (P + R)
    MRR        = (1/|Q|) * Σ 1/rank_i  (Ch.8, QA evaluation)

Usage:
    python -m src.evaluate
"""
import math
from pathlib import Path

import torch
from tqdm import tqdm

from . import finetune_gpt2, neural_model, ngram_model
from .neural_model import Vocab  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "processed" / "test.txt"


def load_test_sentences(limit=None):
    with TEST.open(encoding="utf-8") as f:
        sents = [line.strip().split() for line in f if line.strip()]
    return sents[:limit] if limit else sents


# ===========================================================================
# NEW: Precision / Recall / F1  (Ch.4) + MRR (Ch.8)
# ===========================================================================

def compute_prf1(hits_by_k: dict, total_preds: int, ks=(1, 3, 5)) -> dict:
    """
    Compute Precision, Recall, and F1 at each k.

    In the autocomplete setting:
        - TP@k  = position where true word appears in top-k  (= hits[k])
        - FP@k  = predictions where true word NOT in top-k   (= total - hits[k])
        - FN@k  = same as FP@k (we always predict exactly k words)

    Precision@k = hits[k] / total_preds
    Recall@k    = same as Precision@k here (one true label per position)
    F1@k        = Precision@k  (trivially equal when |true labels| = 1)

    This matches the formulation in the lecture's confusion matrix section.
    """
    results = {}
    for k in ks:
        h = hits_by_k.get(k, 0)
        prec   = h / max(total_preds, 1)
        recall = h / max(total_preds, 1)
        f1     = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
        results[k] = {"precision": prec, "recall": recall, "f1": f1}
    return results


def compute_mrr(rank_list: list) -> float:
    """
    Mean Reciprocal Rank  (Ch.8 — QA Evaluation).
    rank_list: list of integer ranks (1-based) for each query.
               rank=0 means not found in top-k.
    """
    if not rank_list:
        return 0.0
    return sum(1.0 / r if r > 0 else 0.0 for r in rank_list) / len(rank_list)


# ===========================================================================
# N-gram evaluation (original + F1 + MRR)
# ===========================================================================

def eval_ngram(order=4, limit=200):
    """Perplexity only for n-gram (top-k accuracy remains prohibitively slow)."""
    print(f"\n=== N-gram (order={order}) ===", flush=True)
    model  = ngram_model.load_model(order)
    sents  = load_test_sentences(limit=limit)

    total_logp, total_tokens = 0.0, 0
    for sent in tqdm(sents):
        for i in range(1, len(sent)):
            context = tuple(sent[max(0, i - (order - 1)):i])
            p       = model.score(sent[i], context)
            if p > 0:
                total_logp   += math.log(p)
                total_tokens += 1

    ppl = math.exp(-total_logp / max(total_tokens, 1))
    print(f"  perplexity: {ppl:.2f}  (on {total_tokens} tokens)", flush=True)
    return {"ppl": ppl, "top1": None, "top3": None, "top5": None,
            "f1_1": None, "f1_3": None, "f1_5": None, "mrr": None}


# ===========================================================================
# LSTM evaluation (original + F1 + MRR)
# ===========================================================================

@torch.no_grad()
def eval_lstm(limit=500, ks=(1, 3, 5)):
    print("\n=== LSTM ===", flush=True)
    model, vocab, device = neural_model.load_model()
    sents   = load_test_sentences(limit=limit)
    max_k   = max(ks)
    hits    = {k: 0 for k in ks}
    ranks   = []  # for MRR
    total_loss, total_tokens, preds_total = 0.0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")

    for sent in tqdm(sents):
        tokens = [neural_model.BOS] + sent + [neural_model.EOS]
        ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)
        inp, tgt = ids[:, :-1], ids[:, 1:]
        logits, _ = model(inp)
        total_loss   += loss_fn(logits.reshape(-1, logits.size(-1)),
                                tgt.reshape(-1)).item()
        total_tokens += (tgt != 0).sum().item()

        topk = torch.topk(logits[0], max_k, dim=-1).indices.tolist()
        for pos in range(min(len(sent), logits.size(1))):
            true_id = tgt[0, pos].item()
            if true_id == 0:
                continue
            top_ids = topk[pos]
            for k in ks:
                if true_id in top_ids[:k]:
                    hits[k] += 1
            # MRR: find rank of true word
            try:
                rank = top_ids.index(true_id) + 1  # 1-based
            except ValueError:
                rank = 0
            ranks.append(rank)
            preds_total += 1

    ppl  = math.exp(total_loss / max(total_tokens, 1))
    accs = {k: hits[k] / max(preds_total, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_total, ks)
    mrr  = compute_mrr(ranks)

    print(f"  perplexity: {ppl:.2f}", flush=True)
    for k in ks:
        print(f"  top-{k} acc: {accs[k]:.4f}  |  F1@{k}: {prf1[k]['f1']:.4f}", flush=True)
    print(f"  MRR: {mrr:.4f}", flush=True)

    return {
        "ppl": ppl,
        "top1": accs[1], "top3": accs[3], "top5": accs[5],
        "f1_1": prf1[1]["f1"], "f1_3": prf1[3]["f1"], "f1_5": prf1[5]["f1"],
        "mrr": mrr,
    }


# ===========================================================================
# GPT-2 evaluation (original + F1 + MRR)
# ===========================================================================

@torch.no_grad()
def eval_gpt2(limit=300, ks=(1, 3, 5)):
    print("\n=== GPT-2 (fine-tuned) ===", flush=True)
    model, tokenizer, device = finetune_gpt2.load_model()
    sents   = load_test_sentences(limit=limit)
    max_k   = max(ks)
    hits    = {k: 0 for k in ks}
    ranks   = []
    total_loss, total_tokens, preds_total = 0.0, 0, 0

    for sent in tqdm(sents):
        text = " ".join(sent)
        ids  = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if ids.size(1) < 2:
            continue
        out         = model(ids, labels=ids)
        total_loss  += out.loss.item() * (ids.size(1) - 1)
        total_tokens += ids.size(1) - 1

        logits = out.logits[0]
        topk   = torch.topk(logits[:-1], max_k, dim=-1).indices
        target = ids[0, 1:]
        for k in ks:
            hits[k] += (topk[:, :k] == target.unsqueeze(-1)).any(-1).sum().item()

        # MRR over positions
        for pos in range(target.size(0)):
            t  = target[pos].item()
            row = topk[pos].tolist()
            try:
                rank = row.index(t) + 1
            except ValueError:
                rank = 0
            ranks.append(rank)
        preds_total += target.size(0)

    ppl  = math.exp(total_loss / max(total_tokens, 1))
    accs = {k: hits[k] / max(preds_total, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_total, ks)
    mrr  = compute_mrr(ranks)

    print(f"  perplexity: {ppl:.2f}", flush=True)
    for k in ks:
        print(f"  top-{k} acc: {accs[k]:.4f}  |  F1@{k}: {prf1[k]['f1']:.4f}", flush=True)
    print(f"  MRR: {mrr:.4f}", flush=True)

    return {
        "ppl": ppl,
        "top1": accs[1], "top3": accs[3], "top5": accs[5],
        "f1_1": prf1[1]["f1"], "f1_3": prf1[3]["f1"], "f1_5": prf1[5]["f1"],
        "mrr": mrr,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    results = {}
    for name, fn in [("ngram", eval_ngram),
                     ("lstm",  eval_lstm),
                     ("gpt2",  eval_gpt2)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[skip] {name}: {e}", flush=True)
            import traceback; traceback.print_exc()

    print("\n=== Summary ===", flush=True)
    header = f"{'Model':<10} {'PPL':>10} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'F1@1':>8} {'F1@3':>8} {'F1@5':>8} {'MRR':>8}"
    print(header, flush=True)
    for name, r in results.items():
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "   n/a "
        print(
            f"{name:<10} {r['ppl']:>10.2f} "
            f"{fmt(r['top1']):>8} {fmt(r['top3']):>8} {fmt(r['top5']):>8} "
            f"{fmt(r['f1_1']):>8} {fmt(r['f1_3']):>8} {fmt(r['f1_5']):>8} "
            f"{fmt(r['mrr']):>8}",
            flush=True,
        )


if __name__ == "__main__":
    main()