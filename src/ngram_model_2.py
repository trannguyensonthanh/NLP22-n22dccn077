"""
N-gram baseline language model with Kneser-Ney smoothing  +  Linear Interpolation.

CHANGES vs original ngram_model.py:
    1.  InterpolatedNgramModel class — wraps unigram/bigram/trigram/4-gram
        models trained on the same corpus and combines them with learned λ
        weights (Lecture Ch.3: Backoff and Interpolation).
    2.  train_interpolated()  — trains all 4 orders in one pass.
    3.  predict_next_interpolated() — uses the weighted mixture for scoring.
    4.  tune_lambdas() — EM-style search on a held-out validation set.
    5.  Original train/load/predict_next kept 100% intact for backward
        compatibility.

Why interpolation matters (lecture Ch.3 quote):
    "In interpolation, we always mix the probability estimates from all the
     n-gram estimators, weighing and combining the trigram, bigram, and
     unigram counts."

Usage:
    # train once
    python -m src.ngram_model train_interpolated --order 4

    # then predict
    python -m src.ngram_model predict_interp --prefix "the quick brown"
"""

import argparse
import pickle
from pathlib import Path

import nltk
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================

def _ensure_nltk():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def load_sentences(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield word_tokenize(line)


# ===========================================================================
# Original code — kept unchanged for backward compatibility
# ===========================================================================

def train(order: int = 4, train_file_name: str = "train_small.txt"):
    _ensure_nltk()
    train_file = PROC / train_file_name
    if not train_file.exists():
        raise SystemExit(f"Missing {train_file}. Run preprocess.py first.")

    print(f"Loading training sentences from {train_file.name}...")
    sents = list(tqdm(load_sentences(train_file)))
    print(f"  {len(sents):,} sentences")

    print(f"Building {order}-gram pipeline...")
    train_data, vocab = padded_everygram_pipeline(order, sents)

    print(f"Fitting Kneser-Ney {order}-gram model...")
    model = KneserNeyInterpolated(order)
    model.fit(train_data, vocab)
    print(f"  vocab size: {len(model.vocab):,}")

    out = CKPT / f"ngram_{order}.pkl"
    with out.open("wb") as f:
        pickle.dump(model, f)
    print(f"Saved -> {out}")


def load_model(order: int = 4):
    path = CKPT / f"ngram_{order}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def predict_next(model, prefix_tokens, top_k: int = 5):
    """Return top_k (word, prob) candidates — original fast-path implementation."""
    order = model.order
    skip = {"<s>", "</s>", "<UNK>"}
    for n in range(order - 1, 0, -1):
        context = tuple(prefix_tokens[-n:]) if n > 0 else ()
        try:
            cfd = model.counts[n + 1][context]
        except (KeyError, IndexError):
            cfd = None
        if cfd:
            candidates = [w for w in cfd if w not in skip]
            if candidates:
                if len(candidates) > 50:
                    candidates.sort(key=lambda w: -cfd[w])
                    candidates = candidates[:50]
                scored = [(w, model.score(w, context)) for w in candidates]
                scored.sort(key=lambda x: -x[1])
                return scored[:top_k]
    unigrams = model.counts[1][()]
    candidates = [w for w in unigrams if w not in skip]
    scored = [(w, model.score(w, ())) for w in candidates[:5000]]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ===========================================================================
# NEW: Interpolated N-gram model (Ch.3 — Linear Interpolation)
# ===========================================================================

class InterpolatedNgramModel:
    """
    Combines KN models of order 1..max_order with learnable λ weights.

    P_interp(w | ctx) = λ1*P1(w) + λ2*P2(w|w_{-1}) + λ3*P3(w|w_{-2,-1}) + …
    where Σλ = 1 and all λ ≥ 0.

    Default λ are set heuristically from the lecture's Table 3.x observation
    that higher-order models are generally better. They can be tuned with
    tune_lambdas().
    """

    def __init__(self, max_order: int = 4):
        self.max_order = max_order
        self.models: dict[int, KneserNeyInterpolated] = {}
        # Default λ: more weight to higher-order models
        self.lambdas: list[float] = self._default_lambdas(max_order)

    # ------------------------------------------------------------------
    @staticmethod
    def _default_lambdas(order: int) -> list[float]:
        """Exponentially increasing weights toward higher-order models."""
        raw = [2 ** i for i in range(order)]
        total = sum(raw)
        return [r / total for r in raw]

    # ------------------------------------------------------------------
    def fit(self, sents: list, verbose: bool = True):
        """Train one KN model per order on the same sentence list."""
        for order in range(1, self.max_order + 1):
            if verbose:
                print(f"  Fitting {order}-gram KN model…")
            train_data, vocab = padded_everygram_pipeline(order, sents)
            m = KneserNeyInterpolated(order)
            m.fit(train_data, vocab)
            self.models[order] = m
            if verbose:
                print(f"    vocab size: {len(m.vocab):,}")

    # ------------------------------------------------------------------
    def score(self, word: str, context_tokens: list) -> float:
        """
        Interpolated probability for word given context.
        P_interp = Σ_n  λ_n * KN_n(word | context[-n+1:])
        """
        total = 0.0
        for n, lam in enumerate(self.lambdas, start=1):
            m = self.models.get(n)
            if m is None:
                continue
            ctx = tuple(context_tokens[-(n - 1):]) if n > 1 else ()
            try:
                p = m.score(word, ctx)
            except Exception:
                p = 0.0
            total += lam * p
        return total

    # ------------------------------------------------------------------
    def predict_next(self, prefix_tokens: list, top_k: int = 5) -> list:
        """Return top-k (word, interpolated_prob) pairs."""
        skip = {"<s>", "</s>", "<UNK>"}

        # Collect candidate words from highest-order model first
        candidates = set()
        for order in range(self.max_order, 0, -1):
            m = self.models.get(order)
            if m is None:
                continue
            n = order - 1
            ctx = tuple(prefix_tokens[-n:]) if n > 0 else ()
            try:
                cfd = m.counts[order][ctx]
                candidates.update(w for w in cfd if w not in skip)
            except (KeyError, IndexError):
                pass
            if len(candidates) >= 200:
                break

        if not candidates:
            # fallback: unigram candidates
            m1 = self.models.get(1)
            if m1:
                candidates = {w for w in list(m1.counts[1][()])[:2000] if w not in skip}

        scored = [
            (w, self.score(w, prefix_tokens))
            for w in candidates
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ------------------------------------------------------------------
    def tune_lambdas(
        self,
        val_sents: list,
        n_iter: int = 20,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> list[float]:
        """
        Simple gradient-free lambda tuning on a validation set.
        Uses coordinate ascent: perturb each λ, keep change if perplexity
        improves (matches Ch.3: "Choose λs to maximise probability of
        held-out data").
        """
        import math

        def _ppl(lambdas):
            self.lambdas = lambdas
            log_sum, n_tok = 0.0, 0
            for sent in val_sents[:500]:          # cap for speed
                tokens = sent.split()
                for i in range(1, len(tokens)):
                    ctx = tokens[max(0, i - self.max_order + 1):i]
                    p = self.score(tokens[i], ctx)
                    if p > 0:
                        log_sum += math.log(p)
                        n_tok += 1
            return math.exp(-log_sum / max(n_tok, 1))

        current = list(self.lambdas)
        best_ppl = _ppl(current)
        if verbose:
            print(f"  Initial λ={[f'{l:.3f}' for l in current]}  PPL={best_ppl:.2f}")

        for iteration in range(n_iter):
            improved = False
            for i in range(len(current)):
                for delta in [lr, -lr]:
                    candidate = list(current)
                    candidate[i] = max(0.01, candidate[i] + delta)
                    # Normalise to sum=1
                    total = sum(candidate)
                    candidate = [c / total for c in candidate]
                    ppl = _ppl(candidate)
                    if ppl < best_ppl:
                        best_ppl = ppl
                        current = candidate
                        improved = True
            if verbose:
                print(f"  iter {iteration+1:2d}: λ={[f'{l:.3f}' for l in current]}  PPL={best_ppl:.2f}")
            if not improved:
                break

        self.lambdas = current
        return current

    # ------------------------------------------------------------------
    def save(self, path: Path = None):
        path = path or CKPT / f"ngram_interp_{self.max_order}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved interpolated model -> {path}")

    @classmethod
    def load(cls, max_order: int = 4, path: Path = None):
        path = path or CKPT / f"ngram_interp_{max_order}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# Training entry point for interpolated model
# ===========================================================================

def train_interpolated(
    max_order: int = 4,
    train_file: str = "train_small.txt",
    val_file: str = "val_small.txt",
    tune: bool = True,
):
    _ensure_nltk()
    t_path = PROC / train_file
    v_path = PROC / val_file
    if not t_path.exists():
        raise SystemExit(f"Missing {t_path}. Run preprocess.py first.")

    print(f"Loading training sentences from {t_path.name}…")
    sents = list(tqdm(load_sentences(t_path)))
    print(f"  {len(sents):,} sentences")

    model = InterpolatedNgramModel(max_order=max_order)
    model.fit(sents)

    if tune and v_path.exists():
        print("Tuning λ weights on validation set…")
        with v_path.open(encoding="utf-8") as f:
            val_lines = [l.strip() for l in f if l.strip()]
        model.tune_lambdas(val_lines)

    model.save()
    print("Done.")
    return model


def predict_next_interpolated(prefix: str, top_k: int = 5, max_order: int = 4):
    _ensure_nltk()
    model = InterpolatedNgramModel.load(max_order)
    tokens = word_tokenize(prefix.lower())
    return model.predict_next(tokens, top_k)


# ===========================================================================
# CLI
# ===========================================================================

def cli_predict(prefix: str, order: int = 4, top_k: int = 5):
    _ensure_nltk()
    model = load_model(order)
    tokens = word_tokenize(prefix.lower())
    preds = predict_next(model, tokens, top_k)
    print(f"\nPrefix: {prefix!r}")
    print(f"Top-{top_k} next-word predictions (KN-{order}):")
    for w, p in preds:
        print(f"  {w:<20} {p:.4f}")


def cli_predict_interp(prefix: str, order: int = 4, top_k: int = 5):
    _ensure_nltk()
    preds = predict_next_interpolated(prefix, top_k, order)
    print(f"\nPrefix: {prefix!r}")
    print(f"Top-{top_k} next-word predictions (Interpolated 1..{order}-gram):")
    for w, p in preds:
        print(f"  {w:<20} {p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--order", type=int, default=4)
    t.add_argument("--file", default="train_small.txt")

    ti = sub.add_parser("train_interpolated")
    ti.add_argument("--order", type=int, default=4)
    ti.add_argument("--file", default="train_small.txt")
    ti.add_argument("--val",  default="val_small.txt")
    ti.add_argument("--no-tune", action="store_true")

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--top-k", type=int, default=5)

    pi = sub.add_parser("predict_interp")
    pi.add_argument("--prefix", required=True)
    pi.add_argument("--order", type=int, default=4)
    pi.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.order, args.file)
    elif args.cmd == "train_interpolated":
        train_interpolated(args.order, args.file, args.val, not args.no_tune)
    elif args.cmd == "predict":
        cli_predict(args.prefix, args.order, args.top_k)
    elif args.cmd == "predict_interp":
        cli_predict_interp(args.prefix, args.order, args.top_k)