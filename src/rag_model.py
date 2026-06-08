"""
Retrieval-Augmented Generation (RAG) for autocomplete.

Implements the RAG pattern from Lecture Ch.8 (Question Answering):
    1. Retrieve relevant passages from the training corpus (BM25 / dense).
    2. Prepend retrieved context to the prefix before feeding GPT-2.
    3. GPT-2 generates next-word predictions conditioned on both the
       user's prefix AND the retrieved context → better domain accuracy.

Lecture reference (Ch.8):
    "Instead of asking the LM to memorise everything, can we provide the
     LM with relevant and useful content just-in-time? Retrieval / search
     is a common mechanism for identifying such relevant information."

Pipeline:
    [prefix]
        ↓  BM25 retrieval (fast)
    [top-1 passage]
        ↓  Prepend as context
    [context + prefix]
        ↓  GPT-2 (fine-tuned)
    [next-word suggestions]

Usage:
    # 1. Build retrieval index once
    python -m src.rag_model build --corpus data/processed/train_small.txt

    # 2. Predict
    python -m src.rag_model predict --prefix "the experimental results show"
"""

import argparse
import pickle
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

RAG_INDEX_PATH = CKPT / "rag_index.pkl"


# ---------------------------------------------------------------------------
# BM25 retriever  (lightweight, no external dependency)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 passage retriever (Ch.8 — Information Retrieval with BM25).

    Stores sentence-level passages from the training corpus and scores
    them against a query (the user's prefix).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        import math
        from collections import Counter, defaultdict
        self._math = math
        self._Counter = Counter
        self.k1 = k1
        self.b  = b
        self.passages    : list[str]             = []
        self.tok_passages: list[list[str]]       = []
        self.doc_freqs   : dict[str, int]        = {}
        self.term_freqs  : list[dict[str, int]]  = []
        self.doc_lengths : list[int]             = []
        self.avgdl       : float                 = 0.0
        self.idf         : dict[str, float]      = {}
        self.N           : int                   = 0

    def _tokenise(self, text: str) -> list[str]:
        return re.findall(r"[a-z']+", text.lower())

    def fit(self, passages: list[str]):
        import math
        from collections import Counter
        self.passages     = passages
        self.tok_passages = [self._tokenise(p) for p in passages]
        self.term_freqs   = [Counter(t) for t in self.tok_passages]
        self.doc_lengths  = [len(t) for t in self.tok_passages]
        self.avgdl        = sum(self.doc_lengths) / max(len(passages), 1)
        self.N            = len(passages)

        # Document frequencies
        for tf in self.term_freqs:
            for w in tf:
                self.doc_freqs[w] = self.doc_freqs.get(w, 0) + 1

        # IDF (Ch.8 formula)
        for w, df in self.doc_freqs.items():
            self.idf[w] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def get_top_k(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (passage, score) pairs for the query."""
        import math
        query_tokens = self._tokenise(query)
        results = []
        for i, (tf, dl) in enumerate(zip(self.term_freqs, self.doc_lengths)):
            score = 0.0
            for qt in query_tokens:
                if qt not in tf:
                    continue
                idf  = self.idf.get(qt, 0.0)
                f    = tf[qt]
                norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                score += idf * norm
            if score > 0:
                results.append((self.passages[i], score))
        results.sort(key=lambda x: -x[1])
        return results[:k]


# ---------------------------------------------------------------------------
# RAG model
# ---------------------------------------------------------------------------

class RAGModel:
    """
    Retrieval-Augmented Generation for next-word autocomplete.

    Architecture:
        Retriever (BM25) → Context selection → GPT-2 conditional generation
    """

    def __init__(
        self,
        max_context_tokens: int = 50,
        top_k_passages:     int = 1,
        context_prefix:     str = "Context: ",
        sep:                str = " | ",
    ):
        """
        Args:
            max_context_tokens: truncate retrieved passage to this many words.
            top_k_passages:     how many passages to retrieve and concatenate.
            context_prefix:     prepended label for the context block.
            sep:                separator between context and user prefix.
        """
        self.max_context_tokens = max_context_tokens
        self.top_k_passages     = top_k_passages
        self.context_prefix     = context_prefix
        self.sep                = sep
        self.retriever          = None
        self._gpt2_mod          = None
        self._gpt2_model        = None
        self._gpt2_tok          = None
        self._gpt2_dev          = None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, corpus_path, max_passages: int = 100_000, verbose=True):
        corpus_path = Path(corpus_path)
        if verbose:
            print(f"Building RAG index from {corpus_path.name}…")
        passages = []
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and len(s.split()) >= 5:
                    passages.append(s)
                if len(passages) >= max_passages:
                    break
        if verbose:
            print(f"  {len(passages):,} passages")
        self.retriever = BM25Retriever()
        self.retriever.fit(passages)
        if verbose:
            print("  Index built.")

    def save_index(self, path=None):
        path = Path(path or RAG_INDEX_PATH)
        with open(path, "wb") as f:
            pickle.dump(self.retriever, f)
        print(f"Saved RAG index -> {path}")

    def load_index(self, path=None):
        path = Path(path or RAG_INDEX_PATH)
        with open(path, "rb") as f:
            self.retriever = pickle.load(f)

    # ------------------------------------------------------------------
    # GPT-2 lazy loader
    # ------------------------------------------------------------------

    def _ensure_gpt2(self):
        if self._gpt2_model is not None:
            return
        from . import finetune_gpt2
        self._gpt2_mod   = finetune_gpt2
        self._gpt2_model, self._gpt2_tok, self._gpt2_dev = finetune_gpt2.load_model()

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_augmented_prefix(self, prefix: str) -> str:
        """
        Retrieve relevant passage and prepend to prefix.
        Returns: "Context: <passage> | <prefix>"
        """
        if self.retriever is None:
            return prefix  # no index → fall through to plain GPT-2

        top = self.retriever.get_top_k(prefix, k=self.top_k_passages)
        if not top:
            return prefix

        # Truncate passage to max_context_tokens words
        ctx_words = []
        for passage, _ in top:
            words = passage.split()[: self.max_context_tokens // max(self.top_k_passages, 1)]
            ctx_words.extend(words)

        ctx_str  = " ".join(ctx_words)
        augmented = f"{self.context_prefix}{ctx_str}{self.sep}{prefix}"
        return augmented

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_next(self, prefix: str, top_k: int = 5) -> list:
        """
        Retrieve context, augment prefix, run GPT-2, return top-k words.
        Falls back to plain GPT-2 if retriever not loaded.
        """
        self._ensure_gpt2()
        augmented = self._build_augmented_prefix(prefix)
        # Use same predict_next as finetune_gpt2 but with augmented input
        preds = self._gpt2_mod.predict_next(
            self._gpt2_model,
            self._gpt2_tok,
            self._gpt2_dev,
            augmented + " ",
            top_k,
        )
        return preds

    def explain_retrieval(self, prefix: str) -> dict:
        """Show what context gets retrieved for a prefix."""
        if self.retriever is None:
            return {"prefix": prefix, "retrieved": []}
        top = self.retriever.get_top_k(prefix, k=3)
        return {
            "prefix":    prefix,
            "augmented": self._build_augmented_prefix(prefix),
            "retrieved": [{"passage": p[:120], "score": round(s, 4)} for p, s in top],
        }


# ---------------------------------------------------------------------------
# Module-level helpers to match other models' interface
# ---------------------------------------------------------------------------

_rag_instance: RAGModel = None


def load_model(index_path=None):
    global _rag_instance
    _rag_instance = RAGModel()
    _rag_instance.load_index(index_path)
    return _rag_instance


def predict_next(model: RAGModel, prefix: str, top_k: int = 5):
    return model.predict_next(prefix, top_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_build(corpus: str, max_passages: int = 100_000):
    rag = RAGModel()
    rag.build_index(ROOT / corpus, max_passages=max_passages)
    rag.save_index()


def cli_predict(prefix: str, top_k: int = 5):
    rag = RAGModel()
    try:
        rag.load_index()
    except FileNotFoundError:
        print("[warn] RAG index not found — run `python -m src.rag_model build` first.")
        print("       Falling back to plain GPT-2 predictions.")

    print(f"\nRetrieval explanation:")
    info = rag.explain_retrieval(prefix)
    for r in info.get("retrieved", []):
        print(f"  [score={r['score']}] {r['passage']}")
    print(f"  Augmented prefix: {info['augmented'][:120]}…")

    preds = rag.predict_next(prefix, top_k)
    print(f"\nTop-{top_k} RAG next-word predictions:")
    for w, p in preds:
        print(f"  {w:<20} {p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus",        default="data/processed/train_small.txt")
    b.add_argument("--max-passages",  type=int, default=100_000)

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--top-k",  type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "build":
        cli_build(args.corpus, args.max_passages)
    elif args.cmd == "predict":
        cli_predict(args.prefix, args.top_k)