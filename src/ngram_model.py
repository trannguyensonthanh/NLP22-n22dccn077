"""
N-gram baseline language model with Kneser-Ney smoothing.

Why this model: it's the classic statistical baseline. Cheap to train, fast
to query, gives an interpretable lower bound for the neural models. Reports
in a school project look much stronger with this comparison.

Usage:
    python -m src.ngram_model train --order 4
    python -m src.ngram_model predict --prefix "the quick brown"
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
    """Return top_k (word, prob) candidates for the next word.

    Fast path: only score words that have been seen following the longest
    matching context (backing off to shorter contexts if needed). This is
    orders of magnitude faster than scoring the full vocab.
    """
    order = model.order
    skip = {"<s>", "</s>", "<UNK>"}
    # back off until we find a context with continuations
    for n in range(order - 1, 0, -1):
        context = tuple(prefix_tokens[-n:]) if n > 0 else ()
        try:
            cfd = model.counts[n + 1][context]
        except (KeyError, IndexError):
            cfd = None
        if cfd:
            candidates = [w for w in cfd if w not in skip]
            if candidates:
                # cap by raw count first (cheap) then score only the survivors,
                # since model.score is expensive and we only need the top_k
                if len(candidates) > 50:
                    candidates.sort(key=lambda w: -cfd[w])
                    candidates = candidates[:50]
                scored = [(w, model.score(w, context)) for w in candidates]
                scored.sort(key=lambda x: -x[1])
                return scored[:top_k]
    # unigram fallback
    unigrams = model.counts[1][()]
    candidates = [w for w in unigrams if w not in skip]
    scored = [(w, model.score(w, ())) for w in candidates[:5000]]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def cli_predict(prefix: str, order: int = 4, top_k: int = 5):
    _ensure_nltk()
    model = load_model(order)
    tokens = word_tokenize(prefix.lower())
    preds = predict_next(model, tokens, top_k)
    print(f"\nPrefix: {prefix!r}")
    print(f"Top-{top_k} next-word predictions:")
    for w, p in preds:
        print(f"  {w:<20} {p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--order", type=int, default=4)
    t.add_argument("--file", default="train_small.txt")

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.order, args.file)
    else:
        cli_predict(args.prefix, args.order, args.top_k)
