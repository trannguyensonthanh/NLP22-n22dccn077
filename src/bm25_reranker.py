"""
BM25-based suggestion re-ranker.

Lecture Ch.8 (Question Answering / IR) introduces BM25 as an improved
retrieval scoring function over TF-IDF:

    BM25(t, d, q) = IDF(t) * [ tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl)) ]

Here we adapt BM25 to re-rank autocomplete suggestions:
    - "Query"   = the prefix typed by the user.
    - "Document"= a candidate next word plus its local context from training
                  data (approximated via co-occurrence statistics).
    - The candidate that is most "relevant" to the prefix gets bumped up.

Two modes are provided:
    1. Prefix-BM25 (fast): score each candidate word against the prefix
       itself, weighting words that co-occur more with the prefix.
    2. Context-BM25 (richer): uses a pre-built inverted index from the
       training corpus to find which candidate words appear most often in
       contexts similar to the prefix.

Usage:
    from src.bm25_reranker import BM25Reranker
    br = BM25Reranker()
    br.build_index("data/processed/train_small.txt")
    reranked = br.rerank("the experimental results show", predictions)
"""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# BM25 core  (Ch.8)
# ---------------------------------------------------------------------------

class BM25:
    """
    Okapi BM25 scorer.

    Parameters follow standard defaults from the lecture / literature:
        k1 = 1.5   (term saturation)
        b  = 0.75  (length normalisation)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size   = 0
        self.avgdl         = 0.0
        self.doc_freqs     : dict[str, int]       = {}  # df(t) over docs
        self.doc_lengths   : list[int]             = []
        self.term_freqs    : list[dict[str, int]]  = []  # tf(t, d) per doc
        self.idf_cache     : dict[str, float]      = {}

    # ------------------------------------------------------------------
    def fit(self, documents: List[List[str]]):
        """
        documents: list of tokenised documents (each doc = list of words).
        """
        self.corpus_size = len(documents)
        self.term_freqs  = []
        self.doc_lengths = []

        for doc in documents:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(doc))
            for word in tf:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avgdl = sum(self.doc_lengths) / max(self.corpus_size, 1)
        self._compute_idf()

    # ------------------------------------------------------------------
    def _compute_idf(self):
        N = self.corpus_size
        for word, df in self.doc_freqs.items():
            # IDF formula from Ch.8:  log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf_cache[word] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    # ------------------------------------------------------------------
    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """BM25 score for document doc_idx given query_tokens."""
        tf_doc = self.term_freqs[doc_idx]
        dl     = self.doc_lengths[doc_idx]
        score  = 0.0
        for qt in query_tokens:
            if qt not in tf_doc:
                continue
            idf  = self.idf_cache.get(qt, 0.0)
            tf   = tf_doc[qt]
            norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += idf * norm
        return score

    # ------------------------------------------------------------------
    def get_top_docs(self, query_tokens: List[str], top_n: int = 10) -> List[Tuple[int, float]]:
        """Return (doc_idx, score) sorted descending."""
        scores = [
            (i, self.score(query_tokens, i))
            for i in range(self.corpus_size)
        ]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]


# ---------------------------------------------------------------------------
# BM25 Reranker
# ---------------------------------------------------------------------------

class BM25Reranker:
    """
    Re-ranks autocomplete predictions using BM25 context similarity.

    Strategy:
        1. Build a sliding-window corpus: each "document" is a small window
           (size=window_size) of tokens from the training corpus.
        2. For each candidate word, find windows that contain that word.
        3. Score each candidate by how similar those containing windows are
           to the current prefix (using BM25).
        4. Combine BM25 score with the LM probability.
    """

    def __init__(
        self,
        window_size: int = 6,
        alpha: float = 0.4,
        k1: float = 1.5,
        b: float = 0.75,
        max_windows: int = 50_000,
    ):
        """
        Args:
            window_size:  tokens per context window.
            alpha:        BM25 contribution in final score (1-alpha = LM weight).
            max_windows:  cap on corpus size to keep indexing fast.
        """
        self.window_size = window_size
        self.alpha       = alpha
        self.max_windows = max_windows
        self.bm25        = BM25(k1=k1, b=b)
        self.windows     : List[List[str]] = []
        # inverted index: word → list of window indices containing it
        self.inverted    : dict[str, List[int]] = defaultdict(list)
        self._built      = False

    # ------------------------------------------------------------------
    def build_index(self, corpus_path, verbose: bool = True):
        """
        Build BM25 index from a pre-processed text file (one sentence per line).
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        if verbose:
            print(f"Building BM25 index from {corpus_path.name}…")

        windows: List[List[str]] = []
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                tokens = re.findall(r"[a-z']+", line.lower())
                for i in range(len(tokens) - self.window_size + 1):
                    windows.append(tokens[i: i + self.window_size])
                if len(windows) >= self.max_windows:
                    break

        self.windows = windows
        if verbose:
            print(f"  {len(windows):,} context windows")

        # Build inverted index
        for idx, win in enumerate(windows):
            for w in set(win):
                self.inverted[w].append(idx)

        # Fit BM25
        self.bm25.fit(windows)
        self._built = True
        if verbose:
            print("  BM25 index ready.")

    # ------------------------------------------------------------------
    def _candidate_score(self, candidate: str, prefix_tokens: List[str]) -> float:
        """
        BM25 relevance of candidate given prefix context.

        We average the BM25 scores of the top-10 windows containing
        the candidate word.
        """
        if not self._built:
            return 0.0

        win_ids = self.inverted.get(candidate.lower(), [])
        if not win_ids:
            return 0.0

        scores = [self.bm25.score(prefix_tokens, i) for i in win_ids[:200]]
        scores.sort(reverse=True)
        top_n = scores[:10]
        return sum(top_n) / len(top_n) if top_n else 0.0

    # ------------------------------------------------------------------
    def rerank(
        self,
        prefix: str,
        predictions: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Re-rank predictions with BM25 context relevance.

        final_score = (1 - alpha) * lm_prob + alpha * bm25_norm
        """
        if not predictions:
            return predictions
        if not self._built:
            return predictions  # graceful fallback if index not built

        prefix_tokens = re.findall(r"[a-z']+", prefix.lower())

        # BM25 raw scores per candidate
        bm25_scores = [
            self._candidate_score(w, prefix_tokens)
            for w, _ in predictions
        ]

        max_bm25 = max(bm25_scores) or 1.0

        combined = []
        for (w, lm_p), bs in zip(predictions, bm25_scores):
            bm25_norm = bs / max_bm25
            score = (1 - self.alpha) * lm_p + self.alpha * bm25_norm
            combined.append((w, score))

        # Normalise
        total = sum(s for _, s in combined) or 1.0
        normed = [(w, s / total) for w, s in combined]
        normed.sort(key=lambda x: -x[1])
        return normed

    # ------------------------------------------------------------------
    def topk(
        self,
        prefix: str,
        predictions: List[Tuple[str, float]],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        return self.rerank(prefix, predictions)[:k]

    # ------------------------------------------------------------------
    def save(self, path: Path = None):
        import pickle
        path = path or ROOT / "checkpoints" / "bm25_reranker.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved BM25 reranker -> {path}")

    @classmethod
    def load(cls, path: Path = None):
        import pickle
        path = path or ROOT / "checkpoints" / "bm25_reranker.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus", default="data/processed/train_small.txt")
    b.add_argument("--max-windows", type=int, default=50_000)

    t = sub.add_parser("test")
    t.add_argument("--prefix", required=True)

    args = ap.parse_args()

    if args.cmd == "build":
        br = BM25Reranker(max_windows=args.max_windows)
        br.build_index(ROOT / args.corpus)
        br.save()

    elif args.cmd == "test":
        br = BM25Reranker.load()
        dummy_preds = [
            ("the", 0.20), ("results", 0.18), ("method", 0.15),
            ("system", 0.12), ("data", 0.10), ("model", 0.09),
            ("approach", 0.08), ("analysis", 0.08),
        ]
        print(f"\nPrefix: '{args.prefix}'")
        ranked = br.rerank(args.prefix, dummy_preds)
        print("Top-5 after BM25 re-ranking:")
        for w, p in ranked[:5]:
            print(f"  {w:<20} {p:.4f}")