"""
rag_dense.py — RAG with Dense Retrieval + BM25 Hybrid

Extends rag_model.py with dense (embedding-based) retrieval,
and supports three retrieval modes:
    1. BM25 only        (sparse, keyword matching)
    2. Dense only       (semantic, embedding similarity)
    3. Hybrid BM25+Dense (reciprocal rank fusion of both)

Dense retrieval uses a lightweight sentence encoder
(average word embeddings — no heavy transformers needed).
If sentence-transformers is installed, uses it for better quality.

Architecture:
    Dense index: sentences → embeddings (avg-word-emb or SentenceTransformer)
    Query: prefix → embedding → cosine similarity → top-k passages
    Hybrid: RRF(BM25_ranks, Dense_ranks)
    Generation: GPT-2 fine-tuned conditioned on retrieved passage + prefix

Usage:
    python -m src.rag_dense build --mode hybrid
    python -m src.rag_dense predict --prefix "the experimental results" --mode hybrid
"""

import argparse
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

DENSE_IDX_PATH  = CKPT / "rag_dense_index.pkl"
BM25_IDX_PATH   = CKPT / "rag_bm25_index.pkl"
HYBRID_IDX_PATH = CKPT / "rag_hybrid_index.pkl"


# ===========================================================================
# BM25 Retriever (same as rag_model.py — self-contained for clarity)
# ===========================================================================

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.passages: List[str] = []
        self.tok_passages: List[List[str]] = []
        self.term_freqs: List[Counter] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.N: int = 0

    def _tok(self, text: str) -> List[str]:
        return re.findall(r"[a-z']+", text.lower())

    def fit(self, passages: List[str]):
        self.passages     = passages
        self.tok_passages = [self._tok(p) for p in passages]
        self.term_freqs   = [Counter(t) for t in self.tok_passages]
        self.doc_lengths  = [len(t) for t in self.tok_passages]
        self.N            = len(passages)
        self.avgdl        = sum(self.doc_lengths) / max(self.N, 1)
        for tf in self.term_freqs:
            for w in tf:
                self.doc_freqs[w] = self.doc_freqs.get(w, 0) + 1
        for w, df in self.doc_freqs.items():
            self.idf[w] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        tf = self.term_freqs[doc_idx]
        dl = self.doc_lengths[doc_idx]
        s  = 0.0
        for qt in query_tokens:
            if qt not in tf: continue
            idf_  = self.idf.get(qt, 0.0)
            f     = tf[qt]
            norm  = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            s    += idf_ * norm
        return s

    def get_top_k(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        qtoks = self._tok(query)
        scored = [(i, self.score(qtoks, i)) for i in range(self.N)]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ===========================================================================
# Dense Retriever (lightweight word-embedding cosine similarity)
# ===========================================================================

class DenseRetriever:
    """
    Dense passage retrieval using averaged word embeddings.

    If `sentence_transformers` is installed, uses a pretrained model
    for better semantic matching. Otherwise falls back to fast
    TF-IDF-weighted average embeddings (built from corpus statistics).
    """

    def __init__(self, emb_dim: int = 128):
        self.emb_dim    = emb_dim
        self.passages   : List[str]            = []
        self.embeddings : Optional[torch.Tensor] = None  # (N, emb_dim)
        self.vocab_emb  : Dict[str, torch.Tensor] = {}  # word → embedding
        self.idf        : Dict[str, float]     = {}
        self._use_sbert = False

    # ------------------------------------------------------------------
    def _try_sbert(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_sbert = True
            self.emb_dim    = 384
            print("[Dense] Using sentence-transformers (all-MiniLM-L6-v2)")
        except ImportError:
            print("[Dense] sentence-transformers not found — using word avg embeddings")

    # ------------------------------------------------------------------
    def _build_word_embeddings(self, passages: List[str]):
        """Build TF-IDF-weighted random word embeddings (reproducible)."""
        import hashlib
        words: Counter = Counter()
        for p in passages:
            words.update(re.findall(r"[a-z']+", p.lower()))
        N = len(passages)
        doc_freqs: Counter = Counter()
        for p in passages:
            for w in set(re.findall(r"[a-z']+", p.lower())):
                doc_freqs[w] += 1
        for w in words:
            self.idf[w] = math.log((N + 1) / (doc_freqs.get(w, 0) + 1))

        # Deterministic random embeddings per word (hash-based)
        for w in words:
            seed = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % (2**31)
            gen  = torch.Generator()
            gen.manual_seed(seed)
            self.vocab_emb[w] = torch.randn(self.emb_dim, generator=gen)

    def _embed_text(self, text: str) -> torch.Tensor:
        """Average TF-IDF weighted word embeddings for a text."""
        toks = re.findall(r"[a-z']+", text.lower())
        if not toks:
            return torch.zeros(self.emb_dim)
        vecs = []
        for t in toks:
            if t in self.vocab_emb:
                vecs.append(self.vocab_emb[t] * self.idf.get(t, 1.0))
        if not vecs:
            return torch.zeros(self.emb_dim)
        emb = torch.stack(vecs).mean(0)
        return emb / (emb.norm() + 1e-8)

    # ------------------------------------------------------------------
    def fit(self, passages: List[str], verbose: bool = True):
        self.passages = passages
        self._try_sbert()

        if verbose:
            print(f"[Dense] Encoding {len(passages):,} passages…")

        if self._use_sbert:
            all_embs = self._sbert.encode(
                passages, batch_size=128, show_progress_bar=verbose,
                convert_to_tensor=True,
            )
            # Normalise
            all_embs = all_embs / (all_embs.norm(dim=1, keepdim=True) + 1e-8)
            self.embeddings = all_embs.cpu()
        else:
            self._build_word_embeddings(passages)
            embs = torch.stack([self._embed_text(p) for p in passages])
            self.embeddings = embs

        if verbose:
            print(f"[Dense] Index shape: {self.embeddings.shape}")

    # ------------------------------------------------------------------
    def get_top_k(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        if self.embeddings is None:
            return []

        if self._use_sbert:
            q_emb = self._sbert.encode([query], convert_to_tensor=True).cpu()
            q_emb = q_emb / (q_emb.norm() + 1e-8)
        else:
            q_emb = self._embed_text(query).unsqueeze(0)

        sims  = (self.embeddings @ q_emb.T).squeeze(-1)  # (N,)
        topk  = torch.topk(sims, min(k, len(self.passages)))
        return [(int(i), float(s)) for i, s in zip(topk.indices.tolist(), topk.values.tolist())]


# ===========================================================================
# Hybrid Retriever (BM25 + Dense via Reciprocal Rank Fusion)
# ===========================================================================

class HybridRetriever:
    """
    Combines BM25 (sparse) and Dense (semantic) retrieval using RRF.

    RRF ensures neither retriever dominates — relevant passages
    surface if they rank highly in either system.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25  = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def get_top_k(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        bm25_ranked  = self.bm25.get_top_k(query, k=k * 3)    # (idx, score)
        dense_ranked = self.dense.get_top_k(query, k=k * 3)

        # RRF fusion
        rrf_scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranked, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (idx, _) in enumerate(dense_ranked, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return ranked[:k]

    @property
    def passages(self):
        return self.bm25.passages


# ===========================================================================
# RAG Dense Model (main class)
# ===========================================================================

class RAGDenseModel:
    """
    RAG with configurable retrieval: BM25 | Dense | Hybrid.

    Drop-in replacement for rag_model.RAGModel with richer retrieval.
    """

    def __init__(
        self,
        retrieval_mode: str = "hybrid",   # "bm25" | "dense" | "hybrid"
        max_context_tokens: int = 60,
        top_k_passages: int = 2,
    ):
        self.retrieval_mode     = retrieval_mode
        self.max_context_tokens = max_context_tokens
        self.top_k_passages     = top_k_passages

        self._bm25:    Optional[BM25Retriever]    = None
        self._dense:   Optional[DenseRetriever]   = None
        self._hybrid:  Optional[HybridRetriever]  = None
        self._gpt2     = None
        self._gpt2_tok = None
        self._gpt2_dev = None

    # ------------------------------------------------------------------
    def build_index(self, corpus_path, max_passages: int = 100_000, verbose: bool = True):
        corpus_path = Path(corpus_path)
        if verbose:
            print(f"[RAG-Dense] Building {self.retrieval_mode} index from {corpus_path.name}…")

        passages = []
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and len(s.split()) >= 5:
                    passages.append(s)
                if len(passages) >= max_passages:
                    break
        if verbose:
            print(f"[RAG-Dense]   {len(passages):,} passages")

        self._bm25 = BM25Retriever()
        self._bm25.fit(passages)

        if self.retrieval_mode in ("dense", "hybrid"):
            self._dense = DenseRetriever()
            self._dense.fit(passages, verbose=verbose)

        if self.retrieval_mode == "hybrid":
            self._hybrid = HybridRetriever(self._bm25, self._dense)

        if verbose:
            print("[RAG-Dense]   Index ready.")

    # ------------------------------------------------------------------
    def save(self, path: Path = None):
        path = path or HYBRID_IDX_PATH
        with open(path, "wb") as f:
            pickle.dump({
                "bm25":   self._bm25,
                "dense":  self._dense,
                "mode":   self.retrieval_mode,
                "max_ctx": self.max_context_tokens,
                "top_k":  self.top_k_passages,
            }, f)
        print(f"[RAG-Dense] Saved → {path}")

    @classmethod
    def load(cls, path: Path = None) -> "RAGDenseModel":
        path = path or HYBRID_IDX_PATH
        if not path.exists():
            raise FileNotFoundError(f"RAG-Dense index not found at {path}. Run build first.")
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(retrieval_mode=d["mode"],
                  max_context_tokens=d["max_ctx"],
                  top_k_passages=d["top_k"])
        obj._bm25  = d.get("bm25")
        obj._dense = d.get("dense")
        if obj._bm25 and obj._dense:
            obj._hybrid = HybridRetriever(obj._bm25, obj._dense)
        return obj

    # ------------------------------------------------------------------
    def _retrieve(self, query: str) -> List[str]:
        k = self.top_k_passages
        retriever = None
        if self.retrieval_mode == "hybrid" and self._hybrid:
            retriever = self._hybrid
        elif self.retrieval_mode == "dense" and self._dense:
            retriever = self._dense
        elif self._bm25:
            retriever = self._bm25

        if retriever is None:
            return []

        hits = retriever.get_top_k(query, k=k)
        passages = []
        for idx, score in hits:
            if score > 0:
                p = retriever.passages[idx]
                words = p.split()[:self.max_context_tokens // max(k, 1)]
                passages.append(" ".join(words))
        return passages

    # ------------------------------------------------------------------
    def _augmented_prefix(self, prefix: str) -> str:
        passages = self._retrieve(prefix)
        if not passages:
            return prefix
        ctx = " | ".join(passages)
        return f"[Context: {ctx}] {prefix}"

    # ------------------------------------------------------------------
    def _ensure_gpt2(self):
        if self._gpt2 is not None:
            return
        from src import finetune_gpt2
        self._gpt2, self._gpt2_tok, self._gpt2_dev = finetune_gpt2.load_model()

    @torch.no_grad()
    def predict_next(self, prefix: str, top_k: int = 5) -> List[Tuple[str, float]]:
        self._ensure_gpt2()
        augmented = self._augmented_prefix(prefix)

        from src import finetune_gpt2
        return finetune_gpt2.predict_next(
            self._gpt2, self._gpt2_tok, self._gpt2_dev,
            augmented + " ", top_k,
        )

    def explain(self, prefix: str) -> dict:
        passages = self._retrieve(prefix)
        return {
            "mode":      self.retrieval_mode,
            "prefix":    prefix,
            "retrieved": [p[:120] for p in passages],
            "augmented": self._augmented_prefix(prefix)[:200],
        }


# ===========================================================================
# Module-level API
# ===========================================================================

def load_model(mode: str = "hybrid") -> RAGDenseModel:
    return RAGDenseModel.load()


def predict_next(model: RAGDenseModel, prefix: str, top_k: int = 5):
    return model.predict_next(prefix, top_k)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus",       default="data/processed/train_small.txt")
    b.add_argument("--mode",         choices=["bm25", "dense", "hybrid"], default="hybrid")
    b.add_argument("--max-passages", type=int, default=100_000)

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--mode",   choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.add_argument("--top-k",  type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "build":
        model = RAGDenseModel(retrieval_mode=args.mode)
        model.build_index(ROOT / args.corpus, max_passages=args.max_passages)
        model.save()

    elif args.cmd == "predict":
        model = RAGDenseModel.load()
        info  = model.explain(args.prefix)
        print(f"\nMode: {info['mode']}")
        print(f"Retrieved: {info['retrieved']}")
        preds = model.predict_next(args.prefix, args.top_k)
        print(f"\nTop-{args.top_k} predictions:")
        for w, prob in preds:
            print(f"  {w:<20} {prob:.4f}")