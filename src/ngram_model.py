"""
src/ngram_model.py  —  Kneser-Ney N-gram (Custom, không dùng NLTK)
====================================================================

Vấn đề của phiên bản cũ (dùng NLTK KneserNeyInterpolated):
  - Checkpoint 450 MB vì NLTK lưu Python objects (nested dicts + FreqDist)
    với overhead rất lớn (~10x so với dữ liệu thực sự cần thiết).
  - Load chậm 20-30 giây vì pickle phải deserialize hàng triệu objects.

Giải pháp (phiên bản này):
  - Custom KN: chỉ lưu Counter dicts (int keys/values, không overhead).
  - Checkpoint: ~15-40 MB thay vì 450 MB  → giảm ~10-15x.
  - Load time: ~1-2 giây.
  - Train time: ~30-60 giây (không dùng NLTK overhead).
  - min_count=2: bỏ n-gram xuất hiện 1 lần → thêm ~80% nhỏ hơn.
  - CÙNG interface: predict_next(), train(), load_model() — không đổi.

Kneser-Ney (Kneser & Ney 1995):
  P_KN(w|ctx) = max(C(ctx,w)-d, 0) / C(ctx) + lambda(ctx) * P_KN(w|ctx[1:])
  KN unigram dùng continuation probability thay vì frequency.
  d = 0.75 (Church & Gale 1991).

Usage:
    python -m src.ngram_model train              # 4-gram, min_count=2
    python -m src.ngram_model train --order 3   # trigram nếu muốn nhỏ hơn
    python -m src.ngram_model predict --prefix "the quick brown"
    python -m src.ngram_model evaluate
"""

from networkx.generators import spectral_graph_forge
from networkx.algorithms.assortativity import neighbor_degree
import argparse
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

KN_D        = 0.75   # discount constant (Church & Gale 1991)
SKIP        = {"<s>", "</s>"}


# ============================================================
# KNESER-NEY LANGUAGE MODEL
# ============================================================

class KneserNeyLM:
    """
    Custom Kneser-Ney N-gram Language Model — không phụ thuộc NLTK.

    Checkpoint size (train_small.txt, 240k sents):
      order=4, min_count=2: ~15-20 MB   (NLTK: 450 MB)
      order=4, min_count=1: ~50-60 MB
      order=3, min_count=2: ~8-12 MB

    Load time: ~1-2 giây (NLTK: ~20-30 giây).
    """

    def __init__(self, order: int = 4, d: float = KN_D, min_count: int = 2):
        self.order     = order
        self.d         = d
        self.min_count = min_count

        # _counts[n-1] = Counter { (w1,...,wn): count }
        # _counts[0] = unigram Counter { (w,): count }
        # _counts[1] = bigram  Counter { (w1,w2): count }
        # ...
        self._counts: list = []

        # Continuation count cho KN unigram
        # _cont[w] = số loại từ khác nhau xuất hiện TRƯỚC w (trong bigrams)
        self._cont: Counter = Counter()
        # Tổng số bigram types (mẫu số cho KN unigram)
        self._n_bi_types: int = 1

        self.vocab: set = set()
        self._trained = False
        self._ctx_totals = []
        self._ctx_unique_followers = []


    # ----------------------------------------------------------
    def fit(self, sentences: list, verbose: bool = True) -> None:
        """
        Đếm n-gram. sentences: list of raw sentence strings.
        """
        t0 = time.perf_counter()
        if verbose:
            print(f"[KN-{self.order}] Counting {len(sentences):,} sentences…")

        counts = [Counter() for _ in range(self.order)]
        cont_before: defaultdict = defaultdict(set)

        for sent in tqdm(sentences, desc="Training Kneser-Ney"):   
            tokens = sent.strip().split()
            if not tokens:
                continue
            self.vocab.update(tokens)
            # BOS padding: (order-1) <s> tokens
            padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]

            T = len(padded)
            for n in range(1, self.order + 1):
                for i in range(T - n + 1):
                    key = tuple(padded[i:i+n])
                    counts[n-1][key] += 1

            # Continuation: w2 xuất hiện sau bao nhiêu loại w1?
            for i in range(T - 1):
                w1, w2 = padded[i], padded[i+1]
                if w2 not in SKIP:
                    cont_before[w2].add(w1)

        t_count = time.perf_counter() - t0
        if verbose:
            for n in range(1, self.order + 1):
                print(f"  {n}-gram: {len(counts[n-1]):,} types")
            print(f"  Counting: {t_count:.1f}s")

        # Prune low-count n-grams để giảm checkpoint
        if self.min_count > 1 and verbose:
            print(f"  Pruning count < {self.min_count}…")
        for n in tqdm(range(self.order), desc="Pruning n-grams"):
            if self.min_count > 1:
                orig = len(counts[n])
                counts[n] = Counter({k: v for k, v in counts[n].items()
                                     if v >= self.min_count})
                if verbose:
                    print(f"    {n+1}-gram: {orig:,} → {len(counts[n]):,}")

        self._counts = counts

        # Build follower index để predict nhanh
        self._followers = defaultdict(set)

        for key in counts[self.order - 1]:
            ctx = key[:-1]
            nxt = key[-1]
            self._followers[ctx].add(nxt)

        self._cont = Counter({w: len(ws) for w, ws in cont_before.items()})
        self._n_bi_types = max(len(counts[1]) if len(counts) > 1 else 1, 1)
        self._ctx_totals = [Counter() for _ in range(self.order)]
        self._ctx_unique_followers = [Counter() for _ in range(self.order)]

        for n in range(2, self.order + 1):
            for key, count in counts[n - 1].items():
                ctx = key[:-1]
                self._ctx_totals[n - 1][ctx] += count
                self._ctx_unique_followers[n - 1][ctx] += 1

        self._trained = True

        total = time.perf_counter() - t0
        if verbose:
            print(f"  Total: {total:.1f}s")

    # ----------------------------------------------------------
    def _kn(self, word: str, context: tuple) -> float:
        """
        P_KN(word | context) — đệ quy theo Kneser-Ney.
        """
        # Base: KN unigram (continuation probability)
        if not context:
            return max(self._cont.get(word, 0), 1e-10) / self._n_bi_types

        n = min(len(context) + 1, self.order)
        ctx = context[-(n-1):]   # lấy đúng n-1 từ context

        counts_n = self._counts[n-1] if n-1 < len(self._counts) else Counter()
        key = ctx + (word,)

        # C(ctx, w)
        c_ctx_w = counts_n.get(key, 0)

        # C(ctx) từ level n-1
        c_ctx = self._ctx_totals[n - 1].get(ctx, 0)

        if c_ctx == 0:
            return self._kn(word, ctx[1:])   # back-off

        p_disc = max(c_ctx_w - self.d, 0.0) / c_ctx

        # N+(ctx, *) = số loại từ khác nhau theo sau ctx
        n_types = self._ctx_unique_followers[n - 1].get(ctx, 0)
        lambda_ctx = (self.d * n_types) / c_ctx

        return p_disc + lambda_ctx * self._kn(word, ctx[1:])

    def score(self, word: str, context: tuple = ()) -> float:
        """Compatible với NLTK model.score(word, context)."""
        return self._kn(word.lower(), tuple(w.lower() for w in context))

    # ----------------------------------------------------------
    def predict_next(self, prefix_tokens: list, top_k: int = 5) -> list:
        """Top-k (word, prob) cho từ tiếp theo."""
        if not self._trained:
            return []

        n_ctx   = self.order - 1
        context = tuple(w.lower() for w in prefix_tokens[-n_ctx:])

        # Lấy candidates: từ có C(ctx, w) > 0 ở bất kỳ bậc nào
        candidates: set = set()
        for back in range(n_ctx + 1):
            ctx_try = context[back:]

            if ctx_try in self._followers:
                candidates.update(
                    w for w in self._followers[ctx_try]
                    if w not in SKIP
                )

            if len(candidates) >= 300:
                break

        # Fallback: unigram vocab
        if len(candidates) < top_k and self._counts:
            for key, _ in self._counts[0].most_common(500):
                if key[0] not in SKIP:
                    candidates.add(key[0])

        scored = [(w, self._kn(w, context)) for w in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ----------------------------------------------------------
    def sentence_log_prob(self, tokens: list) -> float:
        """Log P(sentence) — dùng cho PPL."""
        lp = 0.0
        for i in range(1, len(tokens)):
            ctx = tuple(tokens[max(0, i - self.order + 1):i])
            p   = self._kn(tokens[i].lower(), ctx)
            lp += math.log(max(p, 1e-10))
        return lp

    # ----------------------------------------------------------
    def save(self, path: Path = None) -> None:
        path = path or CKPT / f"ngram_{self.order}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        mb = Path(path).stat().st_size / 1e6
        print(f"[KN-{self.order}] Saved → {path}  ({mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path = None, order: int = 4) -> "KneserNeyLM":
        path = path or CKPT / f"ngram_{order}.pkl"
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            obj = pickle.load(f)
        elapsed = time.perf_counter() - t0
        mb = Path(path).stat().st_size / 1e6
        print(f"[KN-{obj.order}] Loaded ← {path}  ({mb:.1f} MB, {elapsed:.1f}s)")
        return obj


# ============================================================
# PUBLIC API
# ============================================================

def train(
    order:           int = 4,
    train_file_name: str = "train_small.txt",
    min_count:       int = 2,
) -> KneserNeyLM:
    t_path = PROC / train_file_name
    if not t_path.exists():
        raise SystemExit(f"Missing {t_path}")
    with t_path.open(encoding="utf-8") as f:
        sents = [
            l.strip()
            for l in tqdm(
                f,
                desc="Loading train_small data",
                unit="line"
            )
            if l.strip()
        ]
    print(f"[KN-{order}] {len(sents):,} sentences from {t_path.name}")
    model = KneserNeyLM(order=order, min_count=min_count)
    model.fit(sents, verbose=True)
    model.save()
    return model


def load_model(order: int = 4) -> KneserNeyLM:
    return KneserNeyLM.load(order=order)


def predict_next(model, prefix_tokens: list, top_k: int = 5) -> list:
    return model.predict_next(prefix_tokens, top_k)


def evaluate(order: int = 4, limit: int = 300) -> float:
    model = load_model(order)
    with (PROC / "test.txt").open(encoding="utf-8") as f:
        sents = [l.strip().split() for l in f if l.strip()][:limit]
    total_lp = total_tok = 0
    for sent in tqdm(sents, desc="Evaluating Kneser-Ney"):
        if len(sent) < 2:
            continue
        total_lp  += model.sentence_log_prob(sent)
        total_tok += len(sent)
    ppl = math.exp(-total_lp / max(total_tok, 1))
    print(f"[KN-{order}] PPL on {limit} test sents: {ppl:.2f}")
    return ppl


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kneser-Ney N-gram LM")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--order",     type=int, default=4,
                   help="N-gram order: 3 (nhỏ hơn) hoặc 4 (tốt hơn). Mặc định: 4")
    t.add_argument("--file",      default="train_small.txt")
    t.add_argument("--min-count", type=int, default=2,
                   help="Bỏ n-gram có count < min_count. 2 = giảm 80%% size. Mặc định: 2")

    p = sub.add_parser("predict")
    p.add_argument("--prefix",  required=True)
    p.add_argument("--order",   type=int, default=4)
    p.add_argument("--top-k",   type=int, default=5)

    e = sub.add_parser("evaluate")
    e.add_argument("--order",   type=int, default=4)
    e.add_argument("--limit",   type=int, default=300)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.order, args.file, args.min_count)
    elif args.cmd == "predict":
        m = load_model(args.order)
        for w, pr in m.predict_next(args.prefix.lower().split(), args.top_k):
            print(f"  {w:<22} {pr:.6f}")
    elif args.cmd == "evaluate":
        evaluate(args.order, args.limit)