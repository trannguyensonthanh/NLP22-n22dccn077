"""
src/evaluate_2.py  —  Đánh giá toàn bộ mô hình trên test set (v2)
==================================================================

Models được đánh giá:
    1. N-gram KN-4          (ngram_model.py)
    2. HMM-LM               (hmm_model.py)
    3. LSTM                 (neural_model.py)
    4. RNN Basic            (rnn_model.py)
    5. MaxEnt               (maxent_model.py)
    6. LSTM-AWD             (neural_model_2.py)
    7. GPT-2 fine-tuned     (finetune_gpt2.py)

Metrics (theo bài giảng):
    - Perplexity (Ch.3)
    - Top-1/3/5 Accuracy
    - Precision / Recall / F1@k  (Ch.4)
    - MRR — Mean Reciprocal Rank (Ch.8 QA evaluation)
    - Latency (ms/query)

Usage:
    python -m src.evaluate_2
    python -m src.evaluate_2 --models ngram rnn lstm maxent lstm_awd
"""

import argparse
import math
import time
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "processed" / "test.txt"


def load_test_sentences(limit=None):
    with TEST.open(encoding="utf-8") as f:
        sents = [l.strip().split() for l in f if l.strip()]
    return sents[:limit] if limit else sents


# ============================================================
# METRICS HELPERS
# ============================================================

def compute_prf1(hits_by_k: dict, total: int, ks=(1, 3, 5)) -> dict:
    """
    Precision / Recall / F1 tại k (Ch.4 — Sentiment Analysis).
    Trong autocomplete với 1 nhãn đúng/vị trí:
      Precision@k = Recall@k = F1@k = hits[k] / total
    """
    results = {}
    for k in ks:
        h = hits_by_k.get(k, 0)
        p = h / max(total, 1)
        results[k] = {"precision": p, "recall": p,
                      "f1": (2 * p * p / (2 * p)) if p > 0 else 0.0}
    return results


def compute_mrr(ranks: list) -> float:
    """
    Mean Reciprocal Rank (Ch.8 — QA Evaluation).
    ranks: list of 1-based ranks; 0 = not found.
    """
    if not ranks:
        return 0.0
    return sum(1.0 / r if r > 0 else 0.0 for r in ranks) / len(ranks)


def measure_latency(predict_fn, sample_prefix: str, n_runs: int = 10) -> float:
    """
    Đo latency trung bình (ms/query) sau warmup.
    """
    # Warmup
    for _ in range(3):
        try:
            predict_fn(sample_prefix)
        except Exception:
            return None
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            predict_fn(sample_prefix)
        except Exception:
            return None
        times.append((time.perf_counter() - t0) * 1000)
    return round(sum(times) / len(times), 2)


def _fmt(v):
    if v is None:
        return "   n/a  "
    return f"{v:.4f}"


# ============================================================
# 1. N-GRAM KN-4
# ============================================================

def eval_ngram(order: int = 4, limit: int = 300, ks=(1, 3, 5)):
    print(f"\n=== N-gram KN-{order} ===", flush=True)
    try:
        from . import ngram_model
        from .ngram_model import KneserNeyLM
        import sys

        # Fix pickle checkpoint saved as __main__.KneserNeyLM
        sys.modules["__main__"].KneserNeyLM = KneserNeyLM

        # Fix checkpoint saved as src.evaluate_all.KneserNeyLM nếu có
        sys.modules[__name__].KneserNeyLM = KneserNeyLM
        model = ngram_model.load_model(order)
    except Exception as e:
        print(f"  [debug error] {e}") # In lỗi thực tế để kiểm tra
        print(f"  [skip] checkpoint not found — run: python -m src.ngram_model train")
        return None
    sents = load_test_sentences(limit)  

    # PPL
    total_logp, total_tok = 0.0, 0
    hits = {k: 0 for k in ks}
    ranks = []
    preds_n = 0

    for sent in tqdm(sents, desc="Evaluating N-gram"):
        for i in range(1, len(sent)):
            ctx = tuple(sent[max(0, i - (order - 1)):i])
            p   = model.score(sent[i], ctx)
            if p > 0:
                total_logp += math.log(p)
                total_tok  += 1
        # Top-k accuracy (on first 10 positions per sentence)
        for i in range(2, min(len(sent), 12)):
            prefix_toks = sent[:i]
            true_word   = sent[i]
            preds       = ngram_model.predict_next(model, prefix_toks, top_k=max(ks))
            words       = [w for w, _ in preds]
            for k in ks:
                if true_word in words[:k]:
                    hits[k] += 1
            try:
                ranks.append(words.index(true_word) + 1)
            except ValueError:
                ranks.append(0)
            preds_n += 1

    ppl  = math.exp(-total_logp / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_n, ks)
    mrr  = compute_mrr(ranks)

    # Latency
    lat = measure_latency(
        lambda pr: ngram_model.predict_next(model, pr.split(), 5),
        "the quick brown fox"
    )

    print(f"  PPL:     {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}:  {accs[k]:.4f}  F1@{k}: {prf1[k]['f1']:.4f}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Latency: {lat} ms/query")
    return dict(ppl=ppl, top1=accs[1], top3=accs[3], top5=accs[5],
                f1_1=prf1[1]["f1"], f1_3=prf1[3]["f1"], f1_5=prf1[5]["f1"],
                mrr=mrr, latency_ms=lat)




# ============================================================
# 3. HMM-LM
# ============================================================

def eval_hmm(limit: int = 300, ks=(1, 3, 5)):
    import sys
    from src.hmm_model import HMMLangModel
    sys.modules['__main__'].HMMLangModel = HMMLangModel
    print("\n=== HMM Language Model ===", flush=True)
    try:
        from . import hmm_model
        model = hmm_model.load_model()
    except FileNotFoundError:
        print("  [skip] checkpoint not found — run: python -m src.hmm_model train")
        return None

    sents   = load_test_sentences(limit)
    hits    = {k: 0 for k in ks}
    ranks   = []
    preds_n = 0
    total_lp = 0.0
    total_tok = 0

    for sent in tqdm(sents, desc="Evaluating HMM"):
        if len(sent) < 2:
            continue
        lp = model.sentence_log_prob(sent)
        total_lp  += lp
        total_tok += len(sent)

        for i in range(2, min(len(sent), 12)):
            prefix_toks = sent[:i]
            true_word   = sent[i]
            preds       = model.predict_next(prefix_toks, top_k=max(ks))
            words       = [w for w, _ in preds]
            for k in ks:
                if true_word in words[:k]:
                    hits[k] += 1
            try:
                ranks.append(words.index(true_word) + 1)
            except ValueError:
                ranks.append(0)
            preds_n += 1

    ppl  = math.exp(-total_lp / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_n, ks)
    mrr  = compute_mrr(ranks)

    lat = measure_latency(
        lambda pr: model.predict_next(pr.split(), 5),
        "the quick brown fox"
    )

    print(f"  PPL:     {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}:  {accs[k]:.4f}  F1@{k}: {prf1[k]['f1']:.4f}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Latency: {lat} ms/query")
    return dict(ppl=ppl, top1=accs[1], top3=accs[3], top5=accs[5],
                f1_1=prf1[1]["f1"], f1_3=prf1[3]["f1"], f1_5=prf1[5]["f1"],
                mrr=mrr, latency_ms=lat)


# ============================================================
# 4. LSTM
# ============================================================

def eval_lstm(limit: int = 500, ks=(1, 3, 5)):
    print("\n=== LSTM ===", flush=True)
    try:
        import torch
        from . import neural_model
        model, vocab, device = neural_model.load_model()
    except FileNotFoundError:
        print("  [skip] checkpoint not found — run: python -m src.neural_model train")
        return None

    import torch
    import torch.nn as nn
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    sents     = load_test_sentences(limit)
    hits      = {k: 0 for k in ks}
    ranks     = []
    total_loss = 0.0
    total_tok  = 0
    preds_n    = 0

    with torch.no_grad():
        for sent in tqdm(sents, desc="Evaluating LSTM"):
            tokens = [neural_model.BOS] + sent + [neural_model.EOS]
            ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long,
                                  device=device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            logits, _ = model(inp)
            total_loss += loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1)
            ).item()
            total_tok += (tgt != 0).sum().item()

            topk_ids = torch.topk(logits[0], max(ks), dim=-1).indices.tolist()
            for pos in range(min(len(sent), logits.size(1))):
                true_id = tgt[0, pos].item()
                if true_id == 0:
                    continue
                row = topk_ids[pos]
                for k in ks:
                    if true_id in row[:k]:
                        hits[k] += 1
                try:
                    ranks.append(row.index(true_id) + 1)
                except ValueError:
                    ranks.append(0)
                preds_n += 1

    ppl  = math.exp(total_loss / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_n, ks)
    mrr  = compute_mrr(ranks)

    lat = measure_latency(
        lambda pr: neural_model.predict_next(model, vocab, device, pr, 5),
        "the quick brown fox"
    )

    print(f"  PPL:     {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}:  {accs[k]:.4f}  F1@{k}: {prf1[k]['f1']:.4f}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Latency: {lat} ms/query")
    return dict(ppl=ppl, top1=accs[1], top3=accs[3], top5=accs[5],
                f1_1=prf1[1]["f1"], f1_3=prf1[3]["f1"], f1_5=prf1[5]["f1"],
                mrr=mrr, latency_ms=lat)


# ============================================================
# 5. RNN BASIC  ← NEW
# ============================================================

def eval_rnn(limit: int = 500, ks=(1, 3, 5)):
    """
    Đánh giá RNN Basic — so sánh trực tiếp với LSTM.

    Tại sao RNN kém hơn LSTM?
      - Không có cell state → không nhớ được phụ thuộc xa
      - Vanishing gradient sau ~10-15 bước
      - PPL dự kiến: ~200-300 (so với LSTM ~118)
    Tại sao RNN nhanh hơn LSTM?
      - 1 phép tính/bước (tanh) thay vì 4 gates
      - Ít tham số hơn ~4x
    """
    print("\n=== RNN Basic ===", flush=True)
    try:
        import torch
        import torch.nn as nn
        from . import rnn_model
        model, vocab, device = rnn_model.load_model()
    except FileNotFoundError:
        print("  [skip] checkpoint not found — run: python -m src.rnn_model train")
        return None

    import torch
    import torch.nn as nn
    loss_fn    = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    sents      = load_test_sentences(limit)
    hits       = {k: 0 for k in ks}
    ranks      = []
    total_loss = 0.0
    total_tok  = 0
    preds_n    = 0

    with torch.no_grad():
        for sent in tqdm(sents, desc="Evaluating RNN"):
            from .neural_model import BOS, EOS
            tokens = [BOS] + sent + [EOS]
            ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long,
                                  device=device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            logits, _ = model(inp)
            total_loss += loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1)
            ).item()
            total_tok += (tgt != 0).sum().item()

            topk_ids = torch.topk(logits[0], max(ks), dim=-1).indices.tolist()
            for pos in range(min(len(sent), logits.size(1))):
                true_id = tgt[0, pos].item()
                if true_id == 0:
                    continue
                row = topk_ids[pos]
                for k in ks:
                    if true_id in row[:k]:
                        hits[k] += 1
                try:
                    ranks.append(row.index(true_id) + 1)
                except ValueError:
                    ranks.append(0)
                preds_n += 1

    ppl  = math.exp(total_loss / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_n, ks)
    mrr  = compute_mrr(ranks)

    lat = measure_latency(
        lambda pr: rnn_model.predict_next(model, vocab, device, pr, 5),
        "the quick brown fox"
    )

    print(f"  PPL:     {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}:  {accs[k]:.4f}  F1@{k}: {prf1[k]['f1']:.4f}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Latency: {lat} ms/query")
    return dict(ppl=ppl, top1=accs[1], top3=accs[3], top5=accs[5],
                f1_1=prf1[1]["f1"], f1_3=prf1[3]["f1"], f1_5=prf1[5]["f1"],
                mrr=mrr, latency_ms=lat)


# ============================================================
# 6. MAXENT (standalone)
# ============================================================

def eval_maxent(limit: int = 300, ks=(1, 3, 5)):
    """
    Đánh giá MaxEnt Language Model — standalone (không phải enhancer).

    PPL được tính bằng cách score từng token qua _softmax_over_candidates():
      log P(w_i | context) → sum → PPL = exp(-avg_logp)
    """
    print("\n=== MaxEnt Language Model ===", flush=True)
    try:
        import sys
        from src.maxent_model import MaxEntLM, FeatureExtractor
        sys.modules['__main__'].MaxEntLM = MaxEntLM
        sys.modules['__main__'].FeatureExtractor = FeatureExtractor
        from . import maxent_model
        model = maxent_model.load_model()
    except (FileNotFoundError, Exception) as e:
        print(f"  [skip] checkpoint not found — run: python -m src.maxent_model train")
        return None

    sents   = load_test_sentences(limit)
    hits    = {k: 0 for k in ks}
    ranks   = []
    preds_n = 0
    total_logp = 0.0
    total_tok  = 0

    # Use top-500 vocab for scoring (same as model.predict_next default)
    vocab_cands = model.vocab_list[:500] if model.vocab_list else []
    if not vocab_cands:
        print("  [skip] model has empty vocab")
        return None

    for sent in tqdm(sents, desc="Evaluating MaxEnt"):
        for i in range(1, len(sent)):
            context  = sent[max(0, i - 3): i]
            true_w   = sent[i]

            # PPL: compute log prob
            probs = model._softmax_over_candidates(context, vocab_cands)
            p = probs.get(true_w, 1e-10)
            total_logp += math.log(max(p, 1e-10))
            total_tok  += 1

        # Top-k accuracy (positions 2..11 in each sentence)
        for i in range(2, min(len(sent), 12)):
            prefix_toks = sent[:i]
            true_word   = sent[i]
            preds       = model.predict_next(prefix_toks, top_k=max(ks))
            words       = [w for w, _ in preds]
            for k in ks:
                if true_word in words[:k]:
                    hits[k] += 1
            try:
                ranks.append(words.index(true_word) + 1)
            except ValueError:
                ranks.append(0)
            preds_n += 1

    ppl  = math.exp(-total_logp / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_n, ks)
    mrr  = compute_mrr(ranks)

    lat = measure_latency(
        lambda pr: model.predict_next(pr.split(), 5),
        "the quick brown fox"
    )

    print(f"  PPL:     {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}:  {accs[k]:.4f}  F1@{k}: {prf1[k]['f1']:.4f}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Latency: {lat} ms/query")
    return dict(ppl=ppl, top1=accs[1], top3=accs[3], top5=accs[5],
                f1_1=prf1[1]["f1"], f1_3=prf1[3]["f1"], f1_5=prf1[5]["f1"],
                mrr=mrr, latency_ms=lat)


# ============================================================
# 7. LSTM-AWD (neural_model_2 — AWD variant)
# ============================================================

def eval_lstm_awd(limit: int = 500, ks=(1, 3, 5)):
    """
    Đánh giá AWD-LSTM — Merity et al. 2018.
    Regularization: DropConnect + Variational Dropout + Weight Tying + ASGD.
    """
    print("\n=== AWD-LSTM ===", flush=True)
    try:
        import torch
        import torch.nn as nn
        from . import neural_model_2
        model, vocab, device = neural_model_2.load_model("awd")
    except FileNotFoundError:
        print("  [skip] checkpoint not found — run: python -m src.neural_model_2 train --variant awd")
        return None
    except Exception as e:
        print(f"  [skip] {e}")
        return None

    import torch
    import torch.nn as nn
    loss_fn    = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    sents      = load_test_sentences(limit)
    hits       = {k: 0 for k in ks}
    ranks      = []
    total_loss = 0.0
    total_tok  = 0
    preds_n    = 0

    with torch.no_grad():
        for sent in tqdm(sents, desc="Evaluating AWD-LSTM"):
            tokens = [neural_model_2.BOS] + sent + [neural_model_2.EOS]
            ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long,
                                  device=device)
            inp, tgt = ids[:, :-1], ids[:, 1:]
            logits, _ = model(inp)
            total_loss += loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1)
            ).item()
            total_tok += (tgt != 0).sum().item()

            topk_ids = torch.topk(logits[0], max(ks), dim=-1).indices.tolist()
            for pos in range(min(len(sent), logits.size(1))):
                true_id = tgt[0, pos].item()
                if true_id == 0:
                    continue
                row = topk_ids[pos]
                for k in ks:
                    if true_id in row[:k]:
                        hits[k] += 1
                try:
                    ranks.append(row.index(true_id) + 1)
                except ValueError:
                    ranks.append(0)
                preds_n += 1

    ppl  = math.exp(total_loss / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_n, ks)
    mrr  = compute_mrr(ranks)

    lat = measure_latency(
        lambda pr: neural_model_2.predict_next(model, vocab, device, pr, 5),
        "the quick brown fox"
    )

    print(f"  PPL:     {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}:  {accs[k]:.4f}  F1@{k}: {prf1[k]['f1']:.4f}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Latency: {lat} ms/query")
    return dict(ppl=ppl, top1=accs[1], top3=accs[3], top5=accs[5],
                f1_1=prf1[1]["f1"], f1_3=prf1[3]["f1"], f1_5=prf1[5]["f1"],
                mrr=mrr, latency_ms=lat)


# ============================================================
# 8. GPT-2
# ============================================================

def eval_gpt2(limit: int = 300, ks=(1, 3, 5)):
    print("\n=== GPT-2 (fine-tuned) ===", flush=True)
    try:
        import torch
        import torch.nn as nn
        from . import finetune_gpt2
        model, tokenizer, device = finetune_gpt2.load_model()
    except (FileNotFoundError, ImportError, Exception) as e:
        print(f"  [skip] {e}")
        return None

    import torch
    sents      = load_test_sentences(limit)
    hits       = {k: 0 for k in ks}
    ranks      = []
    total_loss = 0.0
    total_tok  = 0
    preds_n    = 0

    with torch.no_grad():
        for sent in tqdm(sents, desc="Evaluating GPT-2"):
            text = " ".join(sent)
            ids  = tokenizer(text, return_tensors="pt").input_ids.to(device)
            if ids.size(1) < 2:
                continue
            out = model(ids, labels=ids)
            total_loss  += out.loss.item() * (ids.size(1) - 1)
            total_tok   += ids.size(1) - 1

            logits = out.logits[0]
            topk   = torch.topk(logits[:-1], max(ks), dim=-1).indices
            target = ids[0, 1:]
            for k in ks:
                hits[k] += (topk[:, :k] == target.unsqueeze(-1)).any(-1).sum().item()
            for pos in range(target.size(0)):
                t   = target[pos].item()
                row = topk[pos].tolist()
                try:
                    ranks.append(row.index(t) + 1)
                except ValueError:
                    ranks.append(0)
            preds_n += target.size(0)

    ppl  = math.exp(total_loss / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    prf1 = compute_prf1(hits, preds_n, ks)
    mrr  = compute_mrr(ranks)

    lat = measure_latency(
        lambda pr: finetune_gpt2.predict_next(model, tokenizer, device, pr + " ", 5),
        "the quick brown fox"
    )

    print(f"  PPL:     {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}:  {accs[k]:.4f}  F1@{k}: {prf1[k]['f1']:.4f}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Latency: {lat} ms/query")
    return dict(ppl=ppl, top1=accs[1], top3=accs[3], top5=accs[5],
                f1_1=prf1[1]["f1"], f1_3=prf1[3]["f1"], f1_5=prf1[5]["f1"],
                mrr=mrr, latency_ms=lat)


# ============================================================
# MAIN — chạy tất cả hoặc subset
# ============================================================

MODEL_MAP = {
    "ngram":        eval_ngram,
    "hmm":          eval_hmm,
    "lstm":         eval_lstm,
    "rnn":          eval_rnn,
    "maxent":       eval_maxent,
    "lstm_awd":     eval_lstm_awd,
    "gpt2":         eval_gpt2,
}

MODEL_ORDER = ["ngram", "hmm", "lstm", "rnn", "maxent", "lstm_awd", "gpt2"]


def main(models_to_run=None):
    to_run  = models_to_run or MODEL_ORDER
    results = {}

    for name in to_run:
        fn = MODEL_MAP.get(name)
        if fn is None:
            print(f"[skip] unknown model: {name}")
            continue
        try:
            r = fn()
            if r:
                results[name] = r
        except Exception as e:
            import traceback
            print(f"[error] {name}: {e}")
            traceback.print_exc()

    # ── Summary table ──────────────────────────────────────────
    print("\n" + "=" * 95)
    print("SUMMARY — All Models")
    print("=" * 95)
    hdr = (f"{'Model':<16} {'PPL':>8} {'Top-1':>7} {'Top-3':>7} "
           f"{'Top-5':>7} {'F1@1':>7} {'F1@5':>7} {'MRR':>7} {'ms/q':>7}")
    print(hdr)
    print("-" * 95)

    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:<16} "
            f"{r['ppl']:>8.2f} "
            f"{_fmt(r.get('top1')):>7} "
            f"{_fmt(r.get('top3')):>7} "
            f"{_fmt(r.get('top5')):>7} "
            f"{_fmt(r.get('f1_1')):>7} "
            f"{_fmt(r.get('f1_5')):>7} "
            f"{_fmt(r.get('mrr')):>7} "
            f"{r.get('latency_ms', 'n/a'):>7}",
            flush=True,
        )
    print("=" * 95)

    # ── Save results to JSON ───────────────────────────────────
    import json
    APP_KEYS_MAP = {
        "ngram": "4-gram KN",
        "hmm": "HMM-LM",
        "lstm": "LSTM Standard",
        "rnn": "RNN Basic",
        "maxent": "MaxEnt LM",
        "lstm_awd": "AWD-LSTM",
        "gpt2": "GPT-2 fine-tuned",
    }
    
    # Đọc kết quả cũ (nếu có)
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    json_path = log_dir / "eval_results.json"
    
    saved_results = {}
    if json_path.exists():
        try:
            with json_path.open() as f:
                saved_results = json.load(f)
        except Exception:
            pass

    # Cập nhật kết quả mới
    for name, r in results.items():
        app_key = APP_KEYS_MAP.get(name)
        if app_key:
            if app_key not in saved_results:
                saved_results[app_key] = {}
            saved_results[app_key].update({
                "ppl": r.get("ppl"),
                "top1": r.get("top1"),
                "top5": r.get("top5"),
                "mrr": r.get("mrr"),
                "latency_ms": r.get("latency_ms"),
                "trained": True,
            })

    # Lưu lại
    with json_path.open("w") as f:
        json.dump(saved_results, f, indent=2)
    print(f"\n[Info] Results saved to {json_path}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_MAP.keys()),
        default=None,
        help="Chọn models cần evaluate (mặc định: tất cả)",
    )
    args = ap.parse_args()
    main(args.models)