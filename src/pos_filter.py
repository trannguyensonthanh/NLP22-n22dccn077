"""
POS-aware suggestion filter for autocomplete.

Uses spaCy to tag the current prefix, then re-ranks / filters suggestions
so that only grammatically plausible next words are surfaced.

Key idea (from Ch.2 of the lecture):
    After tagging the prefix we know the last POS. From there we can apply
    transition rules that mirror a bigram-POS Markov model:
        VBZ (is/was) → most likely DT / JJ / VBG / NN / RB
        DT  (the/a)  → most likely JJ / NN / NNS
        …etc.

Usage:
    from src.pos_filter import POSFilter
    pf = POSFilter()
    reranked = pf.rerank("the quick brown", predictions)
"""

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# POS transition weights  (last-POS → next-word-POS → boost factor)
# Derived from Penn Treebank bigram statistics (lecture Ch.2 Table).
# Factors > 1 boost, < 1 penalise, 1.0 = neutral.
# ---------------------------------------------------------------------------
POS_TRANSITION_BOOST = {
    "VBZ":  {"NN": 1.5, "NNS": 1.4, "DT": 1.3, "JJ": 1.3, "RB": 1.2, "VBG": 1.2, "IN": 1.1},
    "VBD":  {"NN": 1.4, "NNS": 1.3, "DT": 1.4, "JJ": 1.2, "IN": 1.2, "VBN": 1.3},
    "VB":   {"NN": 1.4, "NNS": 1.3, "DT": 1.5, "JJ": 1.2, "RB": 1.2, "IN": 1.2},
    "VBP":  {"NN": 1.4, "NNS": 1.3, "DT": 1.3, "JJ": 1.2, "RB": 1.2, "VBG": 1.1},
    "VBN":  {"IN": 1.5, "DT": 1.4, "JJ": 1.3, "NN": 1.2, "RB": 1.1},
    "VBG":  {"NN": 1.4, "NNS": 1.3, "DT": 1.4, "JJ": 1.2, "IN": 1.2},
    "MD":   {"VB": 1.8, "RB": 1.3, "VBN": 1.3, "VBG": 1.1},
    "DT":   {"NN": 1.6, "NNS": 1.5, "JJ": 1.8, "JJR": 1.4, "JJS": 1.4, "NNP": 1.3},
    "NN":   {"IN": 1.5, "VBZ": 1.4, "VBD": 1.3, "CC": 1.2, "NN": 1.2, "NNS": 1.1, "POS": 1.4},
    "NNS":  {"VBP": 1.5, "VBD": 1.3, "IN": 1.4, "CC": 1.2, "VBZ": 1.1},
    "NNP":  {"VBZ": 1.4, "VBD": 1.3, "IN": 1.3, "NNP": 1.3, "POS": 1.4, "CC": 1.1},
    "NNPS": {"VBP": 1.4, "VBD": 1.3, "IN": 1.3, "CC": 1.2},
    "JJ":   {"NN": 1.8, "NNS": 1.6, "CC": 1.2, "JJ": 1.1},
    "JJR":  {"NN": 1.7, "NNS": 1.5, "IN": 1.3, "CC": 1.1},
    "JJS":  {"NN": 1.7, "NNS": 1.5, "IN": 1.3},
    "RB":   {"VB": 1.3, "VBD": 1.3, "VBZ": 1.3, "JJ": 1.4, "RB": 1.2, "VBN": 1.2, "IN": 1.1},
    "RBR":  {"JJ": 1.6, "RB": 1.4, "VB": 1.2},
    "RBS":  {"JJ": 1.5, "RB": 1.3, "VB": 1.2},
    "IN":   {"DT": 1.6, "NN": 1.4, "NNS": 1.3, "NNP": 1.3, "PRP": 1.3, "JJ": 1.2},
    "CC":   {"DT": 1.5, "NN": 1.4, "NNS": 1.3, "JJ": 1.3, "VB": 1.2, "PRP": 1.2, "NNP": 1.2},
    "PRP":  {"VBZ": 1.5, "VBD": 1.4, "VBP": 1.4, "MD": 1.3, "VB": 1.2, "VBN": 1.1},
    "PRP$": {"NN": 1.7, "NNS": 1.5, "JJ": 1.3, "NNP": 1.2},
    "WP":   {"VBZ": 1.5, "VBD": 1.4, "VBP": 1.4, "MD": 1.3},
    "WDT":  {"NN": 1.4, "VBZ": 1.3, "VBP": 1.3, "JJ": 1.2},
    "TO":   {"VB": 2.0, "VBG": 1.3, "DT": 1.2, "NN": 1.1},
    "RP":   {"NN": 1.3, "DT": 1.2, "PRP": 1.2, "IN": 1.1},
    "WRB":  {"VBZ": 1.3, "DT": 1.2, "NN": 1.2, "JJ": 1.2, "PRP": 1.2},
}

# Hard-block some POS in clearly wrong contexts
POS_HARD_BLOCK = {
    "TO":  {"IN", "CC", "DT", "VBZ", "VBD"},   # after "to" → must be VB, not "the / and / is"
    "MD":  {"MD", "VBZ", "VBD", "VBN"},         # after modal → no stacked modals / finite verbs
    "DT":  {"DT", "VBZ", "VBD", "VBP", "IN"},   # after det → no det again, no verb
    "POS": {"IN", "VBZ", "VBD"},                 # after 's → likely NP follows
}


class POSFilter:
    """
    Wraps spaCy (or falls back to NLTK) to tag the prefix and re-rank
    model suggestions by grammatical plausibility.
    """

    def __init__(self, model: str = "en_core_web_sm", boost_alpha: float = 0.6):
        """
        Args:
            model:       spaCy model name (needs to be installed).
            boost_alpha: interpolation weight for POS boost.
                         final_score = (1 - alpha) * lm_prob + alpha * boosted_prob
                         Set 0 to disable POS influence, 1 for pure POS.
        """
        self.boost_alpha = boost_alpha
        self._nlp = None
        self._tagger = None
        self._spacy_ok = False
        self._nltk_ok = False
        self._load_tagger(model)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------
    def _load_tagger(self, model: str):
        try:
            import spacy
            self._nlp = spacy.load(model, disable=["parser", "ner", "lemmatizer"])
            self._spacy_ok = True
        except Exception:
            try:
                import nltk
                from nltk import pos_tag, word_tokenize  # noqa: F401
                nltk.download("averaged_perceptron_tagger", quiet=True)
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                self._nltk_ok = True
            except Exception:
                pass  # graceful degradation: no POS filtering

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------
    def _tag_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """Return list of (word, POS-tag) for the prefix tokens."""
        if self._spacy_ok:
            doc = self._nlp(prefix)
            return [(t.text, t.tag_) for t in doc]
        if self._nltk_ok:
            from nltk import pos_tag, word_tokenize
            tokens = word_tokenize(prefix)
            return pos_tag(tokens)
        return []

    def _get_last_pos(self, prefix: str) -> str:
        tags = self._tag_prefix(prefix)
        if not tags:
            return ""
        return tags[-1][1]   # last token's POS tag

    # ------------------------------------------------------------------
    # Word POS (single word)
    # ------------------------------------------------------------------
    def _pos_of_word(self, word: str) -> str:
        """Quick single-word tagging."""
        if self._spacy_ok:
            doc = self._nlp(word)
            if doc:
                return doc[0].tag_
        if self._nltk_ok:
            from nltk import pos_tag
            result = pos_tag([word])
            if result:
                return result[0][1]
        return ""

    # ------------------------------------------------------------------
    # Core: re-rank
    # ------------------------------------------------------------------
    def rerank(
        self,
        prefix: str,
        predictions: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Re-rank a list of (word, probability) predictions using POS context.

        Returns a re-sorted list of (word, adjusted_score) — same length.
        If POS tagging is unavailable, returns predictions unchanged.
        """
        if not (self._spacy_ok or self._nltk_ok):
            return predictions
        if not predictions:
            return predictions

        last_pos = self._get_last_pos(prefix)
        if not last_pos:
            return predictions

        transition_boosts = POS_TRANSITION_BOOST.get(last_pos, {})
        hard_blocks       = POS_HARD_BLOCK.get(last_pos, set())

        scored = []
        for word, prob in predictions:
            word_pos = self._pos_of_word(word)

            # Hard block
            if word_pos in hard_blocks:
                adjusted = prob * 0.05
            else:
                # Soft boost
                boost = transition_boosts.get(word_pos, 1.0)
                adjusted = prob * boost

            scored.append((word, adjusted))

        # Normalise so probabilities still sum to ~1
        total = sum(s for _, s in scored) or 1.0
        normalised = [(w, s / total) for w, s in scored]
        normalised.sort(key=lambda x: -x[1])
        return normalised

    # ------------------------------------------------------------------
    # Convenience: filter to top-k after reranking
    # ------------------------------------------------------------------
    def topk(
        self,
        prefix: str,
        predictions: List[Tuple[str, float]],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        reranked = self.rerank(prefix, predictions)
        return reranked[:k]

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------
    def explain(self, prefix: str) -> dict:
        last_pos = self._get_last_pos(prefix)
        return {
            "prefix": prefix,
            "last_pos": last_pos,
            "expected_next_pos": list(POS_TRANSITION_BOOST.get(last_pos, {}).keys()),
            "hard_blocked_pos": list(POS_HARD_BLOCK.get(last_pos, set())),
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pf = POSFilter()
    sample_preds = [
        ("the", 0.20), ("run", 0.18), ("quickly", 0.15),
        ("beautiful", 0.14), ("is", 0.12), ("cat", 0.11),
        ("and", 0.10),
    ]
    for prefix in [
        "the quick brown",
        "she will",
        "i want to",
        "according to the",
    ]:
        print(f"\nPrefix: '{prefix}'")
        print("  explain:", pf.explain(prefix))
        reranked = pf.rerank(prefix, sample_preds)
        print("  top-5 after POS filter:")
        for w, p in reranked[:5]:
            print(f"    {w:<15} {p:.4f}")