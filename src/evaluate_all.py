"""
Unified evaluation for ALL models — fills logs/eval_results.json for the
Streamlit "Score Comparison" dashboard (demo/app_2.py, Page 2).

For every model that has a checkpoint we report:
    ppl         — perplexity            (lower is better; None if not applicable)
    top1, top5  — top-k next-word accuracy
    mrr         — mean reciprocal rank
    latency_ms  — avg time per predict_next() call
    trained     — whether a checkpoint exists

Metrics methodology
    • Neural LMs (LSTM/AWD/Mamba/GPT-2): teacher-forced forward pass gives
      exact PPL + top-k + MRR efficiently (one pass per sentence).
    • Statistical / retrieval / ensemble models: top-k + MRR + latency are
      measured by calling predict_next() position-by-position. PPL is reported
      only where a normalised next-word probability is cheaply available
      (n-gram KN, interpolated n-gram).

Usage
    python -m src.evaluate_all                       # all models w/ checkpoints
    python -m src.evaluate_all --models ngram lstm gpt2
    python -m src.evaluate_all --limit 50            # quick smoke
    python -m src.evaluate_all --show-only           # print saved JSON, no run
    python -m src.evaluate_all --update --models mamba   # refresh only some
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT  = Path(__file__).resolve().parents[1]
PROC  = ROOT / "data" / "processed"
CKPT  = ROOT / "checkpoints"
TEST  = PROC / "test.txt"
OUT   = ROOT / "logs" / "eval_results.json"

KS = (1, 5)
# NLTK KneserNeyInterpolated.score() is ~2.9s/call (pure-Python, non-optimizable).
# So PPL is estimated over a small token sample to keep runtime sane.
PPL_TOKEN_CAP        = 40   # 4-gram KN
PPL_TOKEN_CAP_INTERP = 15   # interpolated KN (~4x slower per score)


# ===========================================================================
# Data
# ===========================================================================

def _register_main(*classes):
    """
    Checkpoints were pickled while their training script ran as ``__main__``
    (``python -m src.X``), so the pickled classes are bound to ``__main__``.
    Re-expose them on the current ``__main__`` (this module) so pickle.load can
    resolve them. Call right before loading, per model (handles the Vocab name
    clash between neural_model_2 and mamba_model by registering sequentially).
    """
    import __main__
    for c in classes:
        setattr(__main__, c.__name__, c)


def load_test_sentences(limit=None):
    with TEST.open(encoding="utf-8") as f:
        sents = [line.strip().split() for line in f if line.strip()]
    return sents[:limit] if limit else sents


# ===========================================================================
# Generic metric helpers
# ===========================================================================

def _empty():
    return {"ppl": None, "top1": None, "top5": None, "mrr": None,
            "latency_ms": None, "trained": True}


def _predict_eval(predict_fn, sents, max_queries=400, max_pos_per_sent=12):
    """
    Top-1/Top-5/MRR/latency by calling predict_fn(context_tokens, k) which
    returns a list of (word, score). Context is the prefix; target is the
    next word. Bounded by max_queries to stay tractable for slow models.
    """
    max_k = max(KS)
    hits  = {k: 0 for k in KS}
    ranks = []
    n_q   = 0
    t0    = time.perf_counter()

    for sent in tqdm(sents, leave=False):
        for i in range(1, min(len(sent), max_pos_per_sent + 1)):
            context = sent[:i]
            true_w  = sent[i].lower()
            preds   = predict_fn(context, max_k + 5) or []
            words   = [w.lower() for w, _ in preds][:max_k + 5]
            for k in KS:
                if true_w in words[:k]:
                    hits[k] += 1
            ranks.append(words.index(true_w) + 1 if true_w in words else 0)
            n_q += 1
            if n_q >= max_queries:
                break
        if n_q >= max_queries:
            break

    elapsed = time.perf_counter() - t0
    m = _empty()
    m["top1"]       = hits[1] / max(n_q, 1)
    m["top5"]       = hits[5] / max(n_q, 1)
    m["mrr"]        = sum(1.0 / r for r in ranks if r > 0) / max(len(ranks), 1)
    m["latency_ms"] = (elapsed / max(n_q, 1)) * 1000.0
    return m


@torch.no_grad()
def _neural_eval(forward_logits, encode_fn, bos_id, sents, device, ignore_pad=0):
    """
    Exact PPL + top-k + MRR for a neural LM via teacher forcing.
    forward_logits(ids) -> logits (1, T, V)   (handles tuple-return models)
    encode_fn(token_list) -> list[int]        (already includes specials? no)
    The caller wraps BOS/EOS via encode_fn.
    """
    max_k = max(KS)
    hits  = {k: 0 for k in KS}
    ranks = []
    total_loss, total_tok, n_pred = 0.0, 0, 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_pad, reduction="sum")
    t0 = time.perf_counter()

    for sent in tqdm(sents, leave=False):
        ids = torch.tensor([encode_fn(sent)], dtype=torch.long, device=device)
        if ids.size(1) < 2:
            continue
        out    = forward_logits(ids)
        logits = out[0] if isinstance(out, tuple) else out
        inp_logits = logits[:, :-1, :]
        tgt        = ids[:, 1:]
        total_loss += loss_fn(inp_logits.reshape(-1, inp_logits.size(-1)),
                              tgt.reshape(-1)).item()
        total_tok  += (tgt != ignore_pad).sum().item()

        topk = torch.topk(inp_logits[0], max_k, dim=-1).indices.tolist()
        for pos in range(tgt.size(1)):
            true_id = tgt[0, pos].item()
            if true_id == ignore_pad:
                continue
            row = topk[pos]
            for k in KS:
                if true_id in row[:k]:
                    hits[k] += 1
            ranks.append(row.index(true_id) + 1 if true_id in row else 0)
            n_pred += 1

    elapsed = time.perf_counter() - t0
    m = _empty()
    m["ppl"]        = math.exp(total_loss / max(total_tok, 1))
    m["top1"]       = hits[1] / max(n_pred, 1)
    m["top5"]       = hits[5] / max(n_pred, 1)
    m["mrr"]        = sum(1.0 / r for r in ranks if r > 0) / max(len(ranks), 1)
    m["latency_ms"] = (elapsed / max(len(sents), 1)) * 1000.0
    return m


# ===========================================================================
# Per-model evaluators  (return metrics dict, or raise to skip)
# ===========================================================================

def eval_ngram(limit):
    from . import ngram_model
    ngram_model._ensure_nltk()
    model = ngram_model.load_model(4)
    sents = load_test_sentences(limit)

    # PPL via KN score()
    # PPL only — KN predict_next top-k is prohibitively slow (see evaluate_2).
    # NLTK KN .score() is itself slow, so cap the number of scored tokens.
    total_logp, n_tok = 0.0, 0
    for sent in tqdm(sents, leave=False):
        for i in range(1, len(sent)):
            ctx = tuple(sent[max(0, i - 3):i])
            p   = model.score(sent[i], ctx)
            if p > 0:
                total_logp += math.log(p); n_tok += 1
        if n_tok >= PPL_TOKEN_CAP:
            break
    m = _empty()
    m["ppl"] = math.exp(-total_logp / max(n_tok, 1)) if n_tok else None
    return m


def eval_ngram_interp(limit):
    from src.ngram_model_2 import InterpolatedNgramModel
    _register_main(InterpolatedNgramModel)
    model = InterpolatedNgramModel.load(4)
    sents = load_test_sentences(limit)

    # PPL only — interpolated KN predict_next is prohibitively slow.
    total_logp, n_tok = 0.0, 0
    for sent in tqdm(sents, leave=False):
        for i in range(1, len(sent)):
            ctx = [w.lower() for w in sent[max(0, i - 3):i]]
            p   = model.score(sent[i].lower(), ctx)
            if p > 0:
                total_logp += math.log(p); n_tok += 1
        if n_tok >= PPL_TOKEN_CAP_INTERP:
            break
    m = _empty()
    m["ppl"] = math.exp(-total_logp / max(n_tok, 1)) if n_tok else None
    return m


def eval_hmm(limit):  # noqa: E302
    from . import hmm_model
    _register_main(hmm_model.HMMLangModel, hmm_model.Tagger)
    model = hmm_model.load_model()
    sents = load_test_sentences(limit)
    return _predict_eval(lambda ctx, k: hmm_model.predict_next(model, [w.lower() for w in ctx], k), sents, max_queries=80)


def eval_maxent(limit):
    from . import maxent_model
    _register_main(maxent_model.MaxEntLM, maxent_model.FeatureExtractor)
    model = maxent_model.load_model()
    sents = load_test_sentences(limit)
    return _predict_eval(lambda ctx, k: maxent_model.predict_next(model, [w.lower() for w in ctx], k), sents, max_queries=80)


def _eval_neural_v2(variant, limit):
    from src import neural_model_2 as nm
    _register_main(nm.Vocab)
    model, vocab, device = nm.load_model(variant)
    model.eval()
    sents = load_test_sentences(limit)
    enc = lambda toks: vocab.encode([nm.BOS] + [w.lower() for w in toks] + [nm.EOS])
    return _neural_eval(lambda ids: model(ids), enc, vocab.stoi[nm.BOS], sents, device)


def eval_lstm(limit):
    return _eval_neural_v2("standard", limit)


def eval_lstm_awd(limit):
    return _eval_neural_v2("awd", limit)


def eval_lstm_beam(limit):
    from src import neural_model_2 as nm
    _register_main(nm.Vocab)
    model, vocab, device = nm.load_model("standard")
    model.eval()
    sents = load_test_sentences(limit)
    # Beam affects ranking only; PPL not meaningful here.
    def predict(ctx, k):
        return nm.beam_search_predict(model, vocab, device, " ".join(ctx), beam_size=max(k, 8))
    m = _predict_eval(predict, sents, max_queries=80)
    return m


def eval_mamba(limit):
    from src import mamba_model as mm
    _register_main(mm.Vocab)
    model, vocab, device = mm.load_model()
    model.eval()
    sents = load_test_sentences(limit)
    enc = lambda toks: vocab.encode([mm.BOS] + [w.lower() for w in toks] + [mm.EOS])
    return _neural_eval(lambda ids: model(ids), enc, vocab.stoi[mm.BOS], sents, device)


@torch.no_grad()
def eval_gpt2(limit):
    from . import finetune_gpt2
    model, tokenizer, device = finetune_gpt2.load_model()
    sents = load_test_sentences(limit)
    max_k = max(KS)
    hits  = {k: 0 for k in KS}
    ranks = []
    total_loss, total_tok, n_pred = 0.0, 0, 0
    t0 = time.perf_counter()

    for sent in tqdm(sents, leave=False):
        ids = tokenizer(" ".join(sent), return_tensors="pt").input_ids.to(device)
        if ids.size(1) < 2:
            continue
        out = model(ids, labels=ids)
        total_loss += out.loss.item() * (ids.size(1) - 1)
        total_tok  += ids.size(1) - 1
        logits = out.logits[0]
        topk   = torch.topk(logits[:-1], max_k, dim=-1).indices
        target = ids[0, 1:]
        for k in KS:
            hits[k] += (topk[:, :k] == target.unsqueeze(-1)).any(-1).sum().item()
        for pos in range(target.size(0)):
            row = topk[pos].tolist(); t = target[pos].item()
            ranks.append(row.index(t) + 1 if t in row else 0)
        n_pred += target.size(0)

    elapsed = time.perf_counter() - t0
    m = _empty()
    m["ppl"]        = math.exp(total_loss / max(total_tok, 1))
    m["top1"]       = hits[1] / max(n_pred, 1)
    m["top5"]       = hits[5] / max(n_pred, 1)
    m["mrr"]        = sum(1.0 / r for r in ranks if r > 0) / max(len(ranks), 1)
    m["latency_ms"] = (elapsed / max(len(sents), 1)) * 1000.0
    return m


def eval_rag(limit):
    from src import rag_dense
    _register_main(rag_dense.RAGDenseModel, rag_dense.HybridRetriever,
                   rag_dense.BM25Retriever, rag_dense.DenseRetriever)
    model = rag_dense.RAGDenseModel.load()
    sents = load_test_sentences(limit)
    return _predict_eval(lambda ctx, k: model.predict_next(" ".join(ctx), k), sents, max_queries=80)


def eval_rrf(limit):
    from src.moe_ensemble import reciprocal_rank_fusion, MoEEnsemble
    from src import mamba_model
    _register_main(mamba_model.Vocab)     # mamba expert vocab pickle
    experts = ["gpt2", "mamba"]           # vocab-clash-free, fast GPU experts
    moe = MoEEnsemble(active_experts=experts, mode="heuristic")
    sents = load_test_sentences(limit)

    def predict(ctx, k):
        prefix = " ".join(ctx)
        lists = []
        for exp in experts:
            d = moe._expert_predict(exp, prefix, k)
            if d:
                lists.append(sorted(d.items(), key=lambda x: -x[1]))
        return reciprocal_rank_fusion(lists)[:k + 5] if lists else []

    return _predict_eval(predict, sents, max_queries=120)


def eval_moe(limit):
    from src.moe_ensemble import MoEEnsemble
    from src import mamba_model
    _register_main(mamba_model.Vocab)
    moe = MoEEnsemble(active_experts=["gpt2", "mamba"], mode="heuristic")
    try:
        moe.load_gate()
    except Exception:
        pass
    sents = load_test_sentences(limit)
    return _predict_eval(lambda ctx, k: moe.predict(" ".join(ctx), k), sents, max_queries=120)


# ===========================================================================
# Registry:  cli_name -> (display_name, [checkpoint paths], evaluator)
# ===========================================================================

EVALUATORS = {
    "ngram":        ("4-gram KN",           [CKPT / "ngram_4.pkl"],                 eval_ngram),
    "ngram_interp": ("N-gram Interpolated", [CKPT / "ngram_interp_4.pkl"],          eval_ngram_interp),
    "hmm":          ("HMM-LM",              [CKPT / "hmm_lm.pkl"],                   eval_hmm),
    "maxent":       ("MaxEnt LM",           [CKPT / "maxent_model.pkl"],            eval_maxent),
    "lstm":         ("LSTM Standard",       [CKPT / "lstm_best.pt", CKPT / "lstm2_standard_best.pt"], eval_lstm),
    "lstm_beam":    ("LSTM + Beam Search",  [CKPT / "lstm_best.pt", CKPT / "lstm2_standard_best.pt"], eval_lstm_beam),
    "lstm_awd":     ("AWD-LSTM",            [CKPT / "lstm2_awd_best.pt"],            eval_lstm_awd),
    "gpt2":         ("GPT-2 fine-tuned",    [CKPT / "gpt2-finetuned"],               eval_gpt2),
    "mamba":        ("Mamba SSM",           [CKPT / "mamba_best.pt"],                eval_mamba),
    "rag":          ("RAG (Hybrid)",        [CKPT / "rag_hybrid_index.pkl"],         eval_rag),
    "rrf":          ("RRF Ensemble",        [CKPT / "ngram_4.pkl"],                  eval_rrf),
    "moe":          ("Gated Ensemble",      [CKPT / "ngram_4.pkl"],                  eval_moe),
}


def _checkpoint_exists(paths):
    """True if ANY of the candidate checkpoint paths exists."""
    return any(p.exists() for p in paths)


def _load_saved():
    if OUT.exists():
        try:
            with OUT.open() as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save(results):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Evaluate all models → logs/eval_results.json")
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"subset to evaluate (default: all). choices: {list(EVALUATORS)}")
    ap.add_argument("--limit", type=int, default=300, help="test sentences to use")
    ap.add_argument("--show-only", action="store_true", help="print saved results, do not run")
    ap.add_argument("--update", action="store_true",
                    help="merge into existing eval_results.json (keep other models)")
    args = ap.parse_args()

    results = _load_saved()

    if args.show_only:
        if not results:
            print("No saved results at", OUT)
        else:
            print(json.dumps(results, indent=2))
        return

    targets = args.models or list(EVALUATORS)
    unknown = [m for m in targets if m not in EVALUATORS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Valid: {list(EVALUATORS)}")

    # When not --update and evaluating the full set, start fresh.
    if not args.update and args.models is None:
        results = {}

    for name in targets:
        display, ckpts, fn = EVALUATORS[name]
        if not _checkpoint_exists(ckpts):
            print(f"[skip] {display}: no checkpoint ({', '.join(p.name for p in ckpts)})", flush=True)
            results.setdefault(display, _empty())
            results[display]["trained"] = False
            continue
        print(f"\n=== {display} ===", flush=True)
        try:
            metrics = fn(args.limit)
            metrics["trained"] = True
            results[display] = metrics
            def _fmt(v, suf=""):
                return f"{v:.4f}{suf}" if isinstance(v, (int, float)) else "n/a"
            print(f"  ppl={_fmt(metrics['ppl'])}  top1={_fmt(metrics['top1'])}  "
                  f"top5={_fmt(metrics['top5'])}  mrr={_fmt(metrics['mrr'])}  "
                  f"lat={_fmt(metrics['latency_ms'],'ms')}", flush=True)
        except Exception as e:
            import traceback
            print(f"[skip] {display}: {e}", flush=True)
            traceback.print_exc()
            results.setdefault(display, _empty())
            results[display]["trained"] = _checkpoint_exists(ckpts)

    _save(results)
    print(f"\nSaved → {OUT}", flush=True)

    # Summary table
    print(f"\n{'Model':<22}{'PPL':>10}{'Top-1':>8}{'Top-5':>8}{'MRR':>8}{'ms':>9}")
    for display, r in results.items():
        def f(v, p=4):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  n/a"
        print(f"{display:<22}{f(r.get('ppl'),2):>10}{f(r.get('top1')):>8}"
              f"{f(r.get('top5')):>8}{f(r.get('mrr')):>8}{f(r.get('latency_ms'),1):>9}")


if __name__ == "__main__":
    main()