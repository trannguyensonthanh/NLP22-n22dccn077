"""
maxent_model.py — Maximum Entropy Language Model

Maximum Entropy (MaxEnt) / Log-Linear Language Model.
Rosenfeld 1996 — "A Maximum Entropy Approach to Adaptive Statistical LM"

MaxEnt as STANDALONE model:
    P(w | context) ∝ exp(Σ_i λ_i * f_i(context, w))
    where f_i are binary feature functions.

MaxEnt as ENHANCER for HMM / N-gram:
    In Streamlit, user can toggle "Use MaxEnt enhancement" for HMM or N-gram.
    When enabled, MaxEnt re-scores the candidate list produced by HMM/N-gram
    using richer features (POS context, skip-grams, topic words).

Features used:
    - n-gram indicator (bigram, trigram, skip-bigram)
    - POS-tag of previous word(s)
    - Document topic words (TF-IDF top words)
    - Sentence length class
    - Word class (number / punctuation / capitalized)

Usage:
    # standalone
    python -m src.maxent_model train
    python -m src.maxent_model predict --prefix "the quick brown"

    # as enhancer
    from src.maxent_model import MaxEntEnhancer
    enh = MaxEntEnhancer()
    enh.load()
    reranked = enh.enhance(prefix, predictions)   # works on top of any model
"""

import argparse
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

MAXENT_CKPT = CKPT / "maxent_model.pkl"


# ===========================================================================
# Feature Extractor
# ===========================================================================

class FeatureExtractor:
    """
    Extracts binary/real-valued features for MaxEnt model.

    Features for P(w | context):
        1. Bigram (prev_word, w)
        2. Trigram (prev2, prev1, w)
        3. Skip-bigram (prev2, w)
        4. POS class of prev word (crude heuristic: all-caps, ends-s, etc.)
        5. Word class of w (digit, punct, capitalized, common)
        6. Prev word is comma/period (punctuation context)
        7. Prefix length class (short/medium/long sentence)
    """

    def __init__(self, vocab: set = None):
        self.vocab = vocab or set()

    def _pos_class(self, word: str) -> str:
        """Crude POS class without a tagger."""
        if not word:
            return "none"
        if word in ("<s>", "</s>"):
            return "boundary"
        if re.fullmatch(r"\d+(\.\d+)?", word):
            return "number"
        if re.fullmatch(r"[,\.;:!?\-\"'(){}\[\]]", word):
            return "punct"
        if word.endswith("ing"):
            return "verb_ing"
        if word.endswith("ed"):
            return "verb_ed"
        if word.endswith("ly"):
            return "adverb"
        if word.endswith("tion") or word.endswith("ness") or word.endswith("ment"):
            return "noun_suffix"
        if word in {"the", "a", "an"}:
            return "det"
        if word in {"is", "are", "was", "were", "be", "been", "being"}:
            return "aux"
        if word in {"in", "on", "at", "of", "for", "to", "with", "from", "by"}:
            return "prep"
        return "other"

    def features(self, context: List[str], candidate: str) -> Dict[str, float]:
        """
        Returns feature dict for (context, candidate).
        Keys are feature names, values are 1.0 (indicator) or real.
        """
        feats: Dict[str, float] = {}
        prev1 = context[-1] if context else "<s>"
        prev2 = context[-2] if len(context) >= 2 else "<s>"
        w = candidate

        # ---- N-gram features ----
        feats[f"bg:{prev1}_{w}"]        = 1.0
        feats[f"tg:{prev2}_{prev1}_{w}"] = 1.0
        feats[f"skip:{prev2}_{w}"]       = 1.0

        # ---- POS class features ----
        feats[f"pos1:{self._pos_class(prev1)}_{self._pos_class(w)}"] = 1.0
        feats[f"wclass:{self._pos_class(w)}"] = 1.0

        # ---- Punctuation context ----
        if prev1 in {",", ".", ":", ";"}:
            feats["after_punct"] = 1.0

        # ---- Candidate word features ----
        feats[f"cand:{w}"] = 1.0
        if len(w) <= 3:
            feats["short_word"] = 1.0
        if w[0].isupper() if w else False:
            feats["capitalized"] = 1.0

        # ---- Context length ----
        ctx_len = len(context)
        feats[f"ctx_len:{min(ctx_len, 10)}"] = 1.0

        return feats


# ===========================================================================
# MaxEnt Model (Log-Linear, trained with SGD)
# ===========================================================================

class MaxEntLM:
    """
    Maximum Entropy Language Model.

    Training: online SGD with L2 regularisation (Gaussian prior on weights).
    Inference: score all candidate words using learned weights.
    """

    def __init__(
        self,
        top_words: int = 5_000,
        l2: float = 1e-4,
        lr: float = 0.01,
        n_epochs: int = 3,
    ):
        self.top_words = top_words
        self.l2 = l2
        self.lr = lr
        self.n_epochs = n_epochs

        self.weights: Dict[str, float] = defaultdict(float)
        self.vocab_list: List[str] = []
        self.feat_extractor: FeatureExtractor = None

    # ------------------------------------------------------------------
    def _score(self, feats: Dict[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * v for k, v in feats.items())

    def _softmax_over_candidates(
        self, context: List[str], candidates: List[str]
    ) -> Dict[str, float]:
        """Compute normalised P(w | context) for a list of candidates."""
        raw = {w: math.exp(min(self._score(
            self.feat_extractor.features(context, w)
        ), 500)) for w in candidates}
        total = sum(raw.values()) or 1.0
        return {w: v / total for w, v in raw.items()}

    # ------------------------------------------------------------------
    def fit(self, sentences: List[List[str]], verbose: bool = True):
        """
        Online SGD training on tokenised sentences.
        Each (context, next_word) pair is one training example.
        """
        # Build vocabulary from top_words most frequent
        counter: Counter = Counter()
        for sent in sentences:
            counter.update(sent)
        self.vocab_list = [w for w, _ in counter.most_common(self.top_words)]
        vocab_set = set(self.vocab_list)
        self.feat_extractor = FeatureExtractor(vocab=vocab_set)

        if verbose:
            print(f"[MaxEnt] vocab: {len(self.vocab_list):,}  "
                  f"sentences: {len(sentences):,}")

        for epoch in range(1, self.n_epochs + 1):
            total_loss = 0.0
            n_examples = 0
            for sent in sentences:
                for i in range(1, len(sent)):
                    context  = sent[max(0, i - 3): i]
                    true_w   = sent[i]
                    if true_w not in vocab_set:
                        continue

                    # Sample negative candidates (subset of vocab)
                    neg_sample = list(set(self.vocab_list[:200]) - {true_w})[:49]
                    candidates = [true_w] + neg_sample

                    probs = self._softmax_over_candidates(context, candidates)
                    p_true = probs.get(true_w, 1e-10)
                    total_loss -= math.log(p_true + 1e-10)
                    n_examples += 1

                    # SGD update
                    true_feats = self.feat_extractor.features(context, true_w)
                    # gradient: f(ctx, true_w) - E[f(ctx, w)]
                    expected: Dict[str, float] = defaultdict(float)
                    for w, p_w in probs.items():
                        for k, v in self.feat_extractor.features(context, w).items():
                            expected[k] += p_w * v

                    for k, v in true_feats.items():
                        grad = v - expected[k] - self.l2 * self.weights[k]
                        self.weights[k] += self.lr * grad

            avg_loss = total_loss / max(n_examples, 1)
            if verbose:
                print(f"[MaxEnt] Epoch {epoch}: avg_loss={avg_loss:.4f}  "
                      f"active_features={len(self.weights):,}")

    # ------------------------------------------------------------------
    def predict_next(
        self, context_tokens: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k (word, prob) from MaxEnt model."""
        if not self.vocab_list:
            return []
        probs = self._softmax_over_candidates(context_tokens, self.vocab_list[:500])
        scored = sorted(probs.items(), key=lambda x: -x[1])
        return scored[:top_k]

    # ------------------------------------------------------------------
    def save(self, path: Path = None):
        path = path or MAXENT_CKPT
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[MaxEnt] Saved → {path}")

    @classmethod
    def load(cls, path: Path = None) -> "MaxEntLM":
        path = path or MAXENT_CKPT
        with open(path, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# MaxEnt Enhancer — wraps any model's output and re-ranks it
# ===========================================================================

class MaxEntEnhancer:
    """
    Plug-in re-ranker that can be added to HMM or N-gram predictions.

    Usage in Streamlit:
        enhancer = MaxEntEnhancer()
        enhancer.load()
        # given predictions from HMM or n-gram:
        enhanced = enhancer.enhance("the quick brown", base_predictions)

    This matches the Streamlit toggle:
        "HMM" / "HMM + MaxEnt" / "N-gram" / "N-gram + MaxEnt"
    """

    def __init__(self, alpha: float = 0.4):
        """
        alpha: weight of MaxEnt score in final combination.
               final_score = alpha * maxent_prob + (1 - alpha) * base_prob
        """
        self.alpha = alpha
        self._model: MaxEntLM = None

    def load(self, path: Path = None):
        self._model = MaxEntLM.load(path)
        return self

    def is_loaded(self) -> bool:
        return self._model is not None

    def enhance(
        self,
        prefix: str,
        base_predictions: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Re-rank base_predictions using MaxEnt scores.
        base_predictions: [(word, base_prob), ...]
        Returns: [(word, combined_score), ...] sorted best-first.
        """
        if self._model is None or not base_predictions:
            return base_predictions

        context = prefix.lower().split()
        candidates = [w for w, _ in base_predictions]

        # MaxEnt scores for the same candidates
        me_scores = {}
        for w in candidates:
            feats = self._model.feat_extractor.features(context, w)
            me_scores[w] = math.exp(min(self._model._score(feats), 500))
        total = sum(me_scores.values()) or 1.0
        me_norm = {w: s / total for w, s in me_scores.items()}

        # Interpolate
        base_max = max(p for _, p in base_predictions) or 1.0
        combined = []
        for w, bp in base_predictions:
            me_p   = me_norm.get(w, 0.0)
            score  = self.alpha * me_p + (1 - self.alpha) * (bp / base_max)
            combined.append((w, score))

        total_c = sum(s for _, s in combined) or 1.0
        normed  = [(w, s / total_c) for w, s in combined]
        normed.sort(key=lambda x: -x[1])
        return normed


# ===========================================================================
# Training entry point
# ===========================================================================

def train_model(
    train_file: str = "train_small.txt",
    top_words: int = 5_000,
    epochs: int = 3,
    lr: float = 0.01,
):
    t_path = PROC / train_file
    if not t_path.exists():
        raise SystemExit(f"Missing {t_path}")

    print(f"[MaxEnt] Loading {t_path.name}…")
    sentences = []
    with t_path.open(encoding="utf-8") as f:
        for line in f:
            toks = line.strip().lower().split()
            if toks:
                sentences.append(toks)
    print(f"[MaxEnt] {len(sentences):,} sentences")

    model = MaxEntLM(top_words=top_words, lr=lr, n_epochs=epochs)
    model.fit(sentences)
    model.save()
    return model


def load_model():
    return MaxEntLM.load()


def predict_next(model: MaxEntLM, prefix_tokens: List[str], top_k: int = 5):
    return model.predict_next(prefix_tokens, top_k)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--file",       default="train_small.txt")
    t.add_argument("--top-words",  type=int, default=5_000)
    t.add_argument("--epochs",     type=int, default=3)
    t.add_argument("--lr",         type=float, default=0.01)

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--top-k",  type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train_model(args.file, args.top_words, args.epochs, args.lr)
    else:
        m = load_model()
        tokens = args.prefix.lower().split()
        preds  = m.predict_next(tokens, args.top_k)
        print(f"\nPrefix: {args.prefix!r}")
        for w, p in preds:
            print(f"  {w:<20} {p:.4f}")