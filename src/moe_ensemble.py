"""
moe_ensemble.py — Mixture of Experts Ensemble

Shazeer et al. 2017 — "Outrageously Large Neural Networks: Sparsely-Gated MoE"

Instead of averaging all models equally, MoE uses a LEARNED gating network
that decides how much weight to give each model based on the current prefix.

Key idea:
    "When the prefix is academic → weight GPT-2 more"
    "When the prefix is short/informal → weight n-gram more"
    "When the prefix has rare words → weight LSTM+ELMo more"

Architecture:
    Gating features:
        - Prefix length (bucketed)
        - Vocab frequency of last word (log scale)
        - POS tag distribution in prefix
        - Domain signals (academic/news/general word counts)

    Gating network:
        features → Linear → Softmax → weights per expert

    Final prediction:
        P(w) = Σ_k gate_k * P_k(w | prefix)

    The gating weights are LEARNABLE (trained on held-out val set)
    or can be heuristic-set (no training needed).

    Also supports static (fixed weight) ensemble for comparison.

Experts (all optional — only loaded ones are used):
    0: N-gram KN-4         (ngram_model)
    1: N-gram Interpolated (ngram_model_2)
    2: HMM-LM              (hmm_model)
    3: LSTM Standard       (neural_model / neural_model_2)
    4: LSTM AWD            (neural_model_2)
    5: GPT-2 fine-tuned    (finetune_gpt2)
    6: Mamba               (mamba_model)

Usage:
    from src.moe_ensemble import MoEEnsemble

    # Simple (static weights)
    moe = MoEEnsemble(weights={"ngram": 0.1, "lstm": 0.4, "gpt2": 0.5})
    preds = moe.predict("the quick brown", top_k=5)

    # Learned gating
    moe = MoEEnsemble()
    moe.train_gate("data/processed/val_small.txt")
    preds = moe.predict("the quick brown", top_k=5)

    python -m src.moe_ensemble predict --prefix "the quick brown"
    python -m src.moe_ensemble train_gate
"""

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

MOE_CKPT = CKPT / "moe_gate.pkl"

# Registered expert names
ALL_EXPERTS = ["ngram", "ngram_interp", "hmm", "lstm", "lstm_awd", "gpt2", "mamba"]


# ===========================================================================
# Domain / Feature signals for gating
# ===========================================================================

# Crude academic word list (for domain detection)
ACADEMIC_WORDS = {
    "the", "of", "in", "this", "paper", "we", "propose", "method",
    "results", "show", "using", "data", "model", "based", "approach",
    "system", "used", "analysis", "new", "can", "also", "first",
    "two", "however", "also", "our", "study", "research", "experiment",
    "hypothesis", "significant", "furthermore", "moreover", "therefore",
    "abstract", "introduction", "conclusion", "evaluation", "performance",
}
NEWS_WORDS = {
    "said", "says", "told", "according", "reported", "government", "minister",
    "president", "official", "announced", "tuesday", "wednesday", "monday",
    "police", "court", "year", "percent", "million", "billion", "party",
}

# Token frequency bins (approximation — no corpus needed)
COMMON_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "must", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "it", "its", "that", "this", "these", "those", "not",
    "or", "and", "but", "if", "or", "than", "then", "so",
}


def extract_gate_features(prefix: str) -> List[float]:
    """
    Extract 12 gating features from prefix text.
    These are fed into the gating network (or used as heuristics).

    Features:
        0: prefix length bucket (0-1)
        1: last word is common (0/1)
        2: academic word ratio (0-1)
        3: news word ratio (0-1)
        4: avg word length bucket (0-1)
        5: has numbers (0/1)
        6: has punctuation context (0/1)
        7: starts with capital (0/1)
        8: long words ratio (0-1)
        9: unique word ratio (0-1)
        10: last_word ends -ing (0/1)
        11: last_word ends -ed (0/1)
    """
    words = prefix.lower().split()
    if not words:
        return [0.0] * 12

    n = len(words)
    last = words[-1]

    # Feature 0: length bucket (clipped to [0, 20] → [0, 1])
    f0 = min(n, 20) / 20

    # Feature 1: last word common
    f1 = 1.0 if last in COMMON_WORDS else 0.0

    # Feature 2: academic ratio
    acad = sum(1 for w in words if w in ACADEMIC_WORDS)
    f2 = acad / n

    # Feature 3: news ratio
    news = sum(1 for w in words if w in NEWS_WORDS)
    f3 = news / n

    # Feature 4: avg word length (norm to 0-1, max 15)
    f4 = min(sum(len(w) for w in words) / n, 15) / 15

    # Feature 5: has digit
    f5 = 1.0 if any(c.isdigit() for w in words for c in w) else 0.0

    # Feature 6: last word is punctuation
    f6 = 1.0 if words[-1] in {",", ".", ":", ";", "?", "!"} else 0.0

    # Feature 7: original starts with capital
    f7 = 1.0 if prefix and prefix[0].isupper() else 0.0

    # Feature 8: long words (>= 8 chars) ratio
    f8 = sum(1 for w in words if len(w) >= 8) / n

    # Feature 9: unique word ratio (diversity)
    f9 = len(set(words)) / n

    # Feature 10: last ends -ing
    f10 = 1.0 if last.endswith("ing") else 0.0

    # Feature 11: last ends -ed
    f11 = 1.0 if last.endswith("ed") else 0.0

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]


# ===========================================================================
# Gating Network
# ===========================================================================

class GatingNetwork(nn.Module):
    """
    Small MLP that maps gating features → expert weights (softmax).
    """

    def __init__(self, n_features: int = 12, n_experts: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) → (B, n_experts) softmax weights"""
        return F.softmax(self.net(x), dim=-1)


# ===========================================================================
# Heuristic gate (no training needed)
# ===========================================================================

def heuristic_weights(features: List[float], expert_names: List[str]) -> Dict[str, float]:
    """
    Compute expert weights using hand-crafted rules.
    Used as fallback when gating network is not trained.

    Rules:
        - Academic text  → boost GPT-2
        - News text      → boost GPT-2 + LSTM
        - Short prefix   → boost N-gram (less context to use anyway)
        - Long prefix    → boost LSTM / GPT-2 / Mamba
        - Seen last word → boost N-gram
        - Unseen last   → boost GPT-2 / LSTM (via ELMo)
    """
    f = features
    w = {e: 1.0 for e in expert_names}

    # Academic: boost gpt2
    if f[2] > 0.3:
        for k in w:
            if "gpt2" in k:
                w[k] *= 2.5
            if "ngram" in k and "interp" not in k:
                w[k] *= 0.6

    # News: balanced gpt2 + lstm
    elif f[3] > 0.2:
        for k in w:
            if "gpt2" in k or "lstm" in k:
                w[k] *= 1.5

    # Short prefix → n-gram shines
    if f[0] < 0.2:
        for k in w:
            if "ngram" in k:
                w[k] *= 1.8
            if "mamba" in k or "lstm_awd" in k:
                w[k] *= 0.7

    # Long prefix → neural models
    elif f[0] > 0.5:
        for k in w:
            if "ngram" in k:
                w[k] *= 0.5
            if "mamba" in k or "gpt2" in k:
                w[k] *= 1.5

    # Normalise
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


# ===========================================================================
# MoE Ensemble
# ===========================================================================

class MoEEnsemble:
    """
    Mixture of Experts ensemble for next-word prediction.

    Supports:
        - Static weights (fixed, from constructor)
        - Heuristic weights (hand-crafted rules based on prefix features)
        - Learned gating (small MLP trained on validation data)
    """

    def __init__(
        self,
        active_experts: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        mode: str = "heuristic",   # "static" | "heuristic" | "learned"
    ):
        """
        active_experts: which experts to use (default: all available).
        weights: static weights dict (only for mode="static").
        mode: how gating is computed.
        """
        self.active_experts = active_experts or []
        self.static_weights = weights or {}
        self.mode = mode
        self.gate_net: Optional[GatingNetwork] = None
        self._loaded_models: Dict = {}

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_expert(self, name: str):
        """Lazy load an expert model. Returns None if not available."""
        if name in self._loaded_models:
            return self._loaded_models[name]

        try:
            if name == "ngram":
                from src import ngram_model
                ngram_model._ensure_nltk()
                m = ngram_model.load_model(4)
                self._loaded_models[name] = ("ngram", ngram_model, m)

            elif name == "ngram_interp":
                from src.ngram_model_2 import InterpolatedNgramModel
                m = InterpolatedNgramModel.load(4)
                self._loaded_models[name] = ("ngram_interp", None, m)

            elif name == "hmm":
                from src import hmm_model
                m = hmm_model.load_model()
                self._loaded_models[name] = ("hmm", hmm_model, m)

            elif name == "lstm":
                from src import neural_model
                m, vocab, device = neural_model.load_model()
                self._loaded_models[name] = ("lstm", neural_model, m, vocab, device)

            elif name == "lstm_awd":
                from src import neural_model_2
                m, vocab, device = neural_model_2.load_model("awd")
                self._loaded_models[name] = ("lstm_awd", neural_model_2, m, vocab, device)

            elif name == "gpt2":
                from src import finetune_gpt2
                m, tok, device = finetune_gpt2.load_model()
                self._loaded_models[name] = ("gpt2", finetune_gpt2, m, tok, device)

            elif name == "mamba":
                from src import mamba_model
                m, vocab, device = mamba_model.load_model()
                self._loaded_models[name] = ("mamba", mamba_model, m, vocab, device)

        except Exception as e:
            print(f"[MoE] Could not load '{name}': {e}")
            self._loaded_models[name] = None

        return self._loaded_models[name]

    # ------------------------------------------------------------------
    # Per-expert prediction
    # ------------------------------------------------------------------

    def _expert_predict(self, name: str, prefix: str, top_k: int) -> Dict[str, float]:
        """Get {word: prob} from one expert. Returns {} on failure."""
        entry = self._load_expert(name)
        if entry is None:
            return {}

        try:
            kind = entry[0]
            if kind == "ngram":
                _, mod, m = entry
                from nltk.tokenize import word_tokenize
                preds = mod.predict_next(m, word_tokenize(prefix.lower()), top_k * 4)
            elif kind == "ngram_interp":
                _, _, m = entry
                from nltk.tokenize import word_tokenize
                preds = m.predict_next(word_tokenize(prefix.lower()), top_k * 4)
            elif kind == "hmm":
                _, mod, m = entry
                preds = mod.predict_next(m, prefix.lower().split(), top_k * 4)
            elif kind in ("lstm", "lstm_awd"):
                _, mod, m, vocab, device = entry
                preds = mod.predict_next(m, vocab, device, prefix, top_k * 4)
            elif kind == "gpt2":
                _, mod, m, tok, device = entry
                preds = mod.predict_next(m, tok, device, prefix + " ", top_k * 4)
            elif kind == "mamba":
                _, mod, m, vocab, device = entry
                preds = mod.predict_next(m, vocab, device, prefix, top_k * 4)
            else:
                return {}

            return {w: float(p) for w, p in preds}

        except Exception as e:
            print(f"[MoE] Expert '{name}' failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Gating weights
    # ------------------------------------------------------------------

    def _get_weights(self, prefix: str) -> Dict[str, float]:
        names = self.active_experts or list(self._loaded_models.keys())

        if self.mode == "static":
            if not self.static_weights:
                return {n: 1.0 / len(names) for n in names}
            total = sum(self.static_weights.get(n, 0) for n in names) or 1.0
            return {n: self.static_weights.get(n, 0) / total for n in names}

        feats = extract_gate_features(prefix)

        if self.mode == "learned" and self.gate_net is not None:
            with torch.no_grad():
                x = torch.tensor([feats], dtype=torch.float)
                w = self.gate_net(x)[0].tolist()
            expert_list = ALL_EXPERTS
            return {n: w[i] for i, n in enumerate(expert_list) if n in names}

        # Heuristic fallback
        return heuristic_weights(feats, names)

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------

    def predict(
        self,
        prefix: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict next words by combining all active experts.
        Returns (word, combined_prob) list, sorted best-first.
        """
        names = [n for n in (self.active_experts or ALL_EXPERTS)
                 if self._load_expert(n) is not None]

        if not names:
            return []

        weights = self._get_weights(prefix)

        # Collect all candidate words
        expert_preds: Dict[str, Dict[str, float]] = {}
        all_words: set = set()
        for name in names:
            preds = self._expert_predict(name, prefix, top_k)
            expert_preds[name] = preds
            all_words.update(preds.keys())
        combined = {}
        normed_expert_preds = {}
        for name in names:
            preds = expert_preds[name]
            if preds:
                sum_p = sum(preds.values()) or 1.0
                normed_expert_preds[name] = {w: p / sum_p for w, p in preds.items()}
            else:
                normed_expert_preds[name] = {}

        for word in all_words:
            score = 0.0
            for name in names:
                w = weights.get(name, 0.0)
                p = normed_expert_preds[name].get(word, 0.0) 
                score += w * p
            combined[word] = score

        # Sort and return top-k
        ranked = sorted(combined.items(), key=lambda x: -x[1])
        total  = sum(s for _, s in ranked[:top_k]) or 1.0
        return [(w, s / total) for w, s in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Gate training (optional — trains on val-set next-word accuracy)
    # ------------------------------------------------------------------

    def train_gate(
        self,
        val_file: str = "val_small.txt",
        epochs: int = 5,
        lr: float = 1e-2,
        top_k: int = 5,
        max_sents: int = 500,
        verbose: bool = True,
    ):
        """
        Train gating network to maximise top-k accuracy on validation set.
        Uses simple CE loss: gate should weight experts that get it right.
        """
        val_path = PROC / val_file
        if not val_path.exists():
            print("[MoE] Val file not found — using heuristic mode.")
            return

        names  = self.active_experts or ALL_EXPERTS
        n_exp  = len(ALL_EXPERTS)
        net    = GatingNetwork(n_features=12, n_experts=n_exp)
        opt    = torch.optim.Adam(net.parameters(), lr=lr)

        if verbose:
            print(f"[MoE] Training gate on {val_path.name}…")

        with val_path.open() as f:
            sents = [l.strip().split() for l in f if l.strip()][:max_sents]

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_examples = 0
            for sent in sents:
                for i in range(3, min(len(sent), 15)):
                    prefix   = " ".join(sent[:i])
                    true_w   = sent[i]
                    feats    = extract_gate_features(prefix)
                    x        = torch.tensor([feats], dtype=torch.float)
                    gate_w   = net(x)[0]                          # (n_experts,)

                    # Reward: which experts got the right answer?
                    rewards  = []
                    for name in ALL_EXPERTS:
                        preds = self._expert_predict(name, prefix, top_k)
                        hit = 1.0 if true_w in preds.keys() else 0.0
                        rewards.append(hit)

                    reward_t = torch.tensor(rewards, dtype=torch.float)
                    # If no expert got it, skip (avoid degenerate updates)
                    if reward_t.sum() < 1e-6:
                        continue

                    # Loss: negative log-prob of rewarded experts
                    reward_norm = reward_t / reward_t.sum()
                    loss = -(reward_norm * torch.log(gate_w + 1e-8)).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_loss += loss.item()
                    n_examples += 1

            avg = total_loss / max(n_examples, 1)
            if verbose:
                print(f"[MoE] Gate epoch {epoch}: avg_loss={avg:.4f}")

        self.gate_net = net
        self.mode     = "learned"
        self.save_gate()
        if verbose:
            print("[MoE] Gate trained and saved.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_gate(self, path: Path = None):
        path = path or MOE_CKPT
        with open(path, "wb") as f:
            pickle.dump({"net": self.gate_net, "mode": self.mode}, f)

    def load_gate(self, path: Path = None):
        path = path or MOE_CKPT
        if not path.exists():
            return
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.gate_net = d.get("net")


# ===========================================================================
# RRF — Reciprocal Rank Fusion (no-training ensemble)
# ===========================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al. 2009).

    RRF_score(d) = Σ_i  1 / (k + rank_i(d))

    Arguments:
        ranked_lists: list of [(word, prob), …] from each model (already sorted)
        k: smoothing constant (60 is standard)

    Returns: merged [(word, rrf_score), …] sorted best-first.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (word, _) in enumerate(ranked, start=1):
            scores[word] = scores.get(word, 0.0) + 1.0 / (k + rank)

    total = sum(scores.values()) or 1.0
    ranked_out = sorted(scores.items(), key=lambda x: -x[1])
    return [(w, s / total) for w, s in ranked_out]


# ===========================================================================
# Module-level helpers
# ===========================================================================

_global_moe: Optional[MoEEnsemble] = None


def get_or_create(active_experts=None, mode="heuristic") -> MoEEnsemble:
    global _global_moe
    if _global_moe is None:
        _global_moe = MoEEnsemble(active_experts=active_experts, mode=mode)
        _global_moe.load_gate()
    return _global_moe


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--prefix",   required=True)
    p.add_argument("--top-k",    type=int, default=5)
    p.add_argument("--experts",  nargs="+", default=None,
                   help="Experts to use. Default: all available.")
    p.add_argument("--mode",     choices=["static", "heuristic", "learned"],
                   default="heuristic")

    tg = sub.add_parser("train_gate")
    tg.add_argument("--val",     default="val_small.txt")
    tg.add_argument("--epochs",  type=int, default=5)
    tg.add_argument("--experts", nargs="+", default=None)

    args = ap.parse_args()

    if args.cmd == "predict":
        moe = MoEEnsemble(active_experts=args.experts, mode=args.mode)
        moe.load_gate()
        preds = moe.predict(args.prefix, args.top_k)
        print(f"\nMoE ({args.mode}) predictions for: {args.prefix!r}")
        for w, p in preds:
            print(f"  {w:<20} {p:.4f}")

    elif args.cmd == "train_gate":
        moe = MoEEnsemble(active_experts=args.experts)
        moe.train_gate(args.val, args.epochs)