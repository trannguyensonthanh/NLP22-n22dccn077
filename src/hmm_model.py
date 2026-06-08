"""
HMM Language Model — 4th model for comparison.

Implements a Hidden Markov Model language model as taught in Ch.2
(HMM Part-of-Speech Tagging) and Ch.3 (Statistical Language Models).

Architecture:
    - Hidden states  = POS tags (from Penn Treebank tagset, Ch.2)
    - Observations   = words
    - Transition A   = P(tag_t | tag_{t-1})   — bigram POS transitions
    - Emission    B  = P(word  | tag)          — word likelihood given tag
    - Viterbi decoding for finding best tag sequence (Ch.2)
    - Next-word prediction: marginalise over tags

    P(next_word | context) = Σ_tag P(next_word | tag) * P(tag | last_tag)

This is the textbook HMM-LM described in:
    - Ch.2: "HMM part-of-speech tagging"
    - Ch.3: "N-grams and Markov Models"

Compared to the n-gram baseline, HMM-LM:
    - Captures POS-level structure explicitly
    - Generalises better to unseen words via tag emissions
    - PPL expected to be between n-gram (high) and LSTM (low)

Usage:
    python -m src.hmm_model train
    python -m src.hmm_model predict --prefix "the quick brown"
"""

import argparse
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

# Penn Treebank POS tags (Ch.2 Table)
PENN_TAGS = [
    "CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS",
    "MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$",
    "RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG",
    "VBN","VBP","VBZ","WDT","WP","WP$","WRB",
    "<s>","</s>",
]

BOS_TAG = "<s>"
EOS_TAG = "</s>"


# ---------------------------------------------------------------------------
# Tagger wrapper — uses spaCy if available, otherwise NLTK
# ---------------------------------------------------------------------------

class Tagger:
    def __init__(self):
        self._nlp = None
        self._nltk = False
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm",
                                   disable=["parser", "ner", "lemmatizer"])
        except Exception:
            try:
                import nltk
                nltk.download("averaged_perceptron_tagger", quiet=True)
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                self._nltk = True
            except Exception:
                pass

    def tag(self, tokens):
        """Return list of (word, tag) pairs."""
        if self._nlp:
            doc = self._nlp(" ".join(tokens))
            return [(t.text.lower(), t.tag_) for t in doc]
        if self._nltk:
            from nltk import pos_tag
            return [(w.lower(), t) for w, t in pos_tag(tokens)]
        # fallback: everything is NN
        return [(t.lower(), "NN") for t in tokens]


# ---------------------------------------------------------------------------
# HMM Language Model
# ---------------------------------------------------------------------------

class HMMLangModel:
    """
    Bigram Hidden Markov Model language model.

    Training:
        - Count transition bigrams  C(tag_i, tag_{i+1})
        - Count emission pairs      C(tag, word)
        - Smooth with add-k (Ch.3: Add-k smoothing)

    Inference (next-word prediction):
        For each candidate word w, compute:
            P(w | ctx) = Σ_{tag} P(w | tag) * P(tag | last_ctx_tag)
        Take top-k by this marginalised probability.
    """

    def __init__(self, add_k: float = 0.01):
        """
        Args:
            add_k: smoothing constant for both A and B matrices (Ch.3 add-k).
        """
        self.add_k = add_k
        self.tags  = PENN_TAGS

        # Count tables (filled during training)
        self._trans_counts  : dict[str, Counter] = defaultdict(Counter)
        self._emit_counts   : dict[str, Counter] = defaultdict(Counter)
        self._tag_counts    : Counter = Counter()
        self._word_counts   : Counter = Counter()
        self._vocab         : set = set()

        # Smoothed probability tables (filled after training)
        self.trans : dict[str, dict[str, float]] = {}  # A[tag1][tag2]
        self.emit  : dict[str, dict[str, float]] = {}  # B[tag][word]
        self.n_tags : int = 0
        self.n_words: int = 0
        self._tagger = Tagger()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, sentences, verbose=True):
        """
        sentences: list of whitespace-tokenised sentence strings.
        """
        if verbose:
            print("Counting HMM statistics…")
        n = 0
        for sent in sentences:
            tokens = sent.strip().split()
            if not tokens:
                continue
            tagged = self._tagger.tag(tokens)
            # Prepend/append BOS/EOS
            seq = [(None, BOS_TAG)] + tagged + [(None, EOS_TAG)]
            for i in range(len(seq) - 1):
                t1 = seq[i][1]
                t2 = seq[i + 1][1]
                self._trans_counts[t1][t2] += 1
                self._tag_counts[t1] += 1
            for word, tag in tagged:
                self._emit_counts[tag][word] += 1
                self._word_counts[word] += 1
                self._vocab.add(word)
            n += 1
        if verbose:
            print(f"  Processed {n:,} sentences, vocab={len(self._vocab):,}")
        self._build_tables(verbose)

    def _build_tables(self, verbose=True):
        """Convert counts to smoothed log-prob tables (Ch.3 Add-k)."""
        if verbose:
            print("Building smoothed probability tables…")

        all_tags  = set(self._trans_counts.keys()) | set(self._emit_counts.keys())
        all_tags.update(PENN_TAGS)
        self.n_tags  = len(all_tags)
        self.n_words = len(self._vocab)
        k = self.add_k

        # Transition probabilities  A[t1][t2]
        self.trans = {}
        for t1 in all_tags:
            denom = self._tag_counts.get(t1, 0) + k * self.n_tags
            self.trans[t1] = {}
            for t2 in all_tags:
                count = self._trans_counts[t1].get(t2, 0)
                self.trans[t1][t2] = (count + k) / denom

        # Emission probabilities  B[tag][word]
        self.emit = {}
        for tag in all_tags:
            denom = self._tag_counts.get(tag, 0) + k * (self.n_words + 1)
            self.emit[tag] = {}
            for word in self._vocab:
                count = self._emit_counts[tag].get(word, 0)
                self.emit[tag][word] = (count + k) / denom
            # UNK
            self.emit[tag]["<UNK>"] = k / denom

        if verbose:
            print("  Done.")

    # ------------------------------------------------------------------
    # Viterbi (Ch.2) — finds best tag sequence for a token list
    # ------------------------------------------------------------------

    def viterbi(self, tokens):
        """
        Viterbi algorithm (Ch.2) — returns best tag sequence for tokens.
        Also computes sentence probability (for perplexity).
        """
        words = [t.lower() for t in tokens]
        T     = len(words)
        if T == 0:
            return [], 0.0

        all_tags = [t for t in self.trans.keys() if t not in (BOS_TAG, EOS_TAG)]

        # dp[t][tag] = log P(best path ending at tag at position t)
        dp      = [dict() for _ in range(T)]
        back    = [dict() for _ in range(T)]

        # Init: transitions from BOS
        for tag in all_tags:
            a  = self.trans.get(BOS_TAG, {}).get(tag, 1e-10)
            w  = words[0]
            b  = self.emit.get(tag, {}).get(w, self.emit.get(tag, {}).get("<UNK>", 1e-10))
            dp[0][tag]   = math.log(a) + math.log(b)
            back[0][tag] = BOS_TAG

        # Recursion
        for t in range(1, T):
            w = words[t]
            for tag in all_tags:
                b = self.emit.get(tag, {}).get(w, self.emit.get(tag, {}).get("<UNK>", 1e-10))
                best_prev, best_score = None, -math.inf
                for prev_tag in all_tags:
                    if prev_tag not in dp[t - 1]:
                        continue
                    a = self.trans.get(prev_tag, {}).get(tag, 1e-10)
                    score = dp[t - 1][prev_tag] + math.log(a) + math.log(b)
                    if score > best_score:
                        best_score = score
                        best_prev  = prev_tag
                dp[t][tag]   = best_score
                back[t][tag] = best_prev

        # Termination
        best_last, best_final = None, -math.inf
        for tag in all_tags:
            a = self.trans.get(tag, {}).get(EOS_TAG, 1e-10)
            score = dp[T - 1].get(tag, -math.inf) + math.log(a)
            if score > best_final:
                best_final = score
                best_last  = tag

        # Backtrack
        path = [best_last]
        for t in range(T - 1, 0, -1):
            path.insert(0, back[t][path[0]])

        return path, best_final

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def sentence_log_prob(self, tokens) -> float:
        """Log probability of a sentence under the HMM-LM."""
        words = [t.lower() for t in tokens]
        _, lp  = self.viterbi(words)
        return lp

    # ------------------------------------------------------------------
    # Next-word prediction
    # ------------------------------------------------------------------

    def predict_next(self, prefix_tokens, top_k=5):
        """
        Predict next word using marginalised HMM probability.

        P(w | ctx) = Σ_{tag_next} P(w | tag_next) * P(tag_next | last_ctx_tag)

        where last_ctx_tag is obtained from Viterbi on the prefix.
        """
        if not prefix_tokens:
            last_tag = BOS_TAG
        else:
            path, _ = self.viterbi(prefix_tokens)
            last_tag = path[-1] if path else BOS_TAG

        skip = {"<s>", "</s>", "<unk>", "<pad>"}
        # Candidate words: most frequent in training corpus
        candidates = [w for w, _ in self._word_counts.most_common(2000)
                      if w not in skip]

        scored = []
        for word in candidates:
            # Marginalise over possible next tags
            prob = 0.0
            for next_tag, a in self.trans.get(last_tag, {}).items():
                if next_tag in (BOS_TAG, EOS_TAG):
                    continue
                b = self.emit.get(next_tag, {}).get(word,
                    self.emit.get(next_tag, {}).get("<UNK>", 1e-10))
                prob += a * b
            scored.append((word, prob))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path=None):
        path = Path(path or CKPT / "hmm_lm.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved HMM model -> {path}")

    @classmethod
    def load(cls, path=None):
        path = Path(path or CKPT / "hmm_lm.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(train_file: str = "train_small.txt", add_k: float = 0.01):
    t_path = PROC / train_file
    if not t_path.exists():
        raise SystemExit(f"Missing {t_path}. Run preprocess.py first.")
    print(f"Loading {t_path.name}…")
    with t_path.open(encoding="utf-8") as f:
        sents = [line.strip() for line in f if line.strip()]
    print(f"  {len(sents):,} sentences")

    model = HMMLangModel(add_k=add_k)
    model.fit(sents)
    model.save()
    return model


def load_model():
    return HMMLangModel.load()


def predict_next(model, prefix_tokens, top_k=5):
    """Wrapper matching the interface of the other models."""
    return model.predict_next(prefix_tokens, top_k)


# ---------------------------------------------------------------------------
# Evaluation (perplexity)
# ---------------------------------------------------------------------------

def evaluate(limit=200):
    import math
    model  = load_model()
    test_p = PROC / "test.txt"
    with test_p.open(encoding="utf-8") as f:
        sents  = [l.strip().split() for l in f if l.strip()][:limit]

    total_lp, total_tok = 0.0, 0
    for sent in sents:
        if len(sent) < 2:
            continue
        lp = model.sentence_log_prob(sent)
        total_lp  += lp
        total_tok += len(sent)

    ppl = math.exp(-total_lp / max(total_tok, 1))
    print(f"HMM-LM perplexity on test set ({limit} sents): {ppl:.2f}")
    return ppl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_predict(prefix: str, top_k: int = 5):
    model  = load_model()
    tokens = prefix.lower().split()
    preds  = model.predict_next(tokens, top_k)
    print(f"\nPrefix: {prefix!r}")
    print(f"Top-{top_k} next-word predictions (HMM-LM):")
    for w, p in preds:
        print(f"  {w:<20} {p:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--file",  default="train_small.txt")
    t.add_argument("--add-k", type=float, default=0.01)

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--top-k",  type=int, default=5)

    e = sub.add_parser("evaluate")
    e.add_argument("--limit", type=int, default=200)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.file, args.add_k)
    elif args.cmd == "predict":
        cli_predict(args.prefix, args.top_k)
    elif args.cmd == "evaluate":
        evaluate(args.limit)