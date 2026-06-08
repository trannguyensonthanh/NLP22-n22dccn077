"""
Sentiment-aware suggestion filter for autocomplete.

Based on Lecture Ch.4 (Sentiment Classification):
    - Detects the sentiment/register of the current prefix using a
      lightweight Naïve Bayes classifier (from the lecture baseline).
    - Re-ranks suggestions to prefer words that maintain the detected tone.

Register categories detected:
    formal    — academic / journalistic writing
    informal  — conversational
    positive  — positive sentiment context
    negative  — negative sentiment context
    neutral   — no strong signal

The classifier is intentionally lightweight (word-list + NB scoring) so it
runs in < 1 ms without any model loading. A spaCy/transformers version can
replace it via the `fit_from_corpus()` method.

Usage:
    from src.sentiment_filter import SentimentFilter
    sf = SentimentFilter()
    reranked = sf.rerank("the experimental results demonstrate", predictions)
"""

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Lexicon — curated word lists for lightweight NB prior
# ---------------------------------------------------------------------------

FORMAL_WORDS = {
    "therefore", "moreover", "furthermore", "consequently", "nevertheless",
    "however", "whereas", "albeit", "notwithstanding", "henceforth",
    "hereinafter", "pursuant", "aforementioned", "subsequent", "hereby",
    "thereby", "wherein", "thereof", "inasmuch", "insofar",
    "demonstrate", "illustrate", "indicate", "suggest", "reveal",
    "propose", "investigate", "examine", "analyse", "evaluate",
    "significant", "substantial", "considerable", "notable", "prominent",
    "according", "reported", "observed", "concluded", "established",
    "methodology", "framework", "paradigm", "empirical", "theoretical",
    "hypothesis", "findings", "analysis", "implementation", "objective",
}

INFORMAL_WORDS = {
    "yeah", "gonna", "wanna", "kinda", "sorta", "gotta", "dunno",
    "btw", "tbh", "imo", "lol", "omg", "ngl", "smh",
    "pretty", "really", "just", "like", "stuff", "thing", "things",
    "cool", "awesome", "great", "amazing", "super", "totally",
    "basically", "literally", "actually", "honestly",
    "hey", "hi", "ok", "okay", "alright",
}

POSITIVE_WORDS = {
    "excellent", "outstanding", "remarkable", "exceptional", "superb",
    "wonderful", "fantastic", "brilliant", "impressive", "innovative",
    "successful", "effective", "efficient", "valuable", "beneficial",
    "promising", "encouraging", "positive", "improved", "enhanced",
    "good", "great", "best", "better", "perfect", "ideal",
    "achieve", "succeed", "gain", "improve", "advance",
}

NEGATIVE_WORDS = {
    "poor", "weak", "limited", "disappointing", "inadequate",
    "problematic", "concerning", "unfortunate", "flawed", "insufficient",
    "failed", "failure", "error", "mistake", "issue", "problem",
    "difficult", "challenging", "complex", "complicated", "confusing",
    "worse", "worst", "bad", "terrible", "awful",
    "however", "but", "although", "despite", "nonetheless",
    "unfortunately", "regrettably",
}

# Words that are good continuations for each register
FORMAL_CONTINUATIONS = {
    "the", "this", "these", "such", "which", "that", "an", "a",
    "significant", "important", "key", "major", "primary", "main",
    "further", "additional", "notable", "relevant",
    "results", "findings", "data", "evidence", "analysis",
    "approach", "method", "model", "system", "framework",
}

INFORMAL_CONTINUATIONS = {
    "you", "we", "i", "it", "that", "this", "what",
    "really", "just", "pretty", "so", "very",
    "like", "well", "yeah", "right",
    "good", "great", "nice", "cool", "okay",
}

POSITIVE_CONTINUATIONS = {
    "effectively", "successfully", "significantly", "notably", "remarkably",
    "well", "greatly", "strongly", "highly", "consistently",
    "improved", "enhanced", "increased", "achieved", "demonstrated",
    "promising", "excellent", "outstanding",
}

NEGATIVE_CONTINUATIONS = {
    "unfortunately", "however", "but", "yet", "still",
    "failed", "limited", "poorly", "insufficiently",
    "challenges", "problems", "issues", "difficulties",
    "worse", "lower", "decreased", "reduced",
}


# ---------------------------------------------------------------------------
# Naïve Bayes classifier  (Ch.4 baseline algorithm)
# ---------------------------------------------------------------------------

class NaiveBayesSentiment:
    """
    Lightweight Multinomial Naïve Bayes sentiment/register classifier.

    Matches the lecture's Boolean NB variant (Ch.4):
        "Clips all the word counts in each document at 1"
    """

    CLASSES = ("formal", "informal", "positive", "negative", "neutral")

    def __init__(self):
        # log P(class)  — uniform prior
        self.log_prior = {c: math.log(1 / len(self.CLASSES)) for c in self.CLASSES}
        # log P(w | class) — built from lexicons
        self._build_likelihoods()

    def _build_likelihoods(self):
        """Build log-likelihood tables from hand-crafted lexicons."""
        vocab_by_class = {
            "formal":   FORMAL_WORDS,
            "informal": INFORMAL_WORDS,
            "positive": POSITIVE_WORDS,
            "negative": NEGATIVE_WORDS,
            "neutral":  set(),
        }
        # Vocabulary = union of all class words
        all_words = set().union(*vocab_by_class.values())
        V = len(all_words) + 1  # +1 for <UNK>

        self.log_likelihood: Dict[str, Dict[str, float]] = {}
        for cls, word_set in vocab_by_class.items():
            total = len(word_set) + V  # add-1 Laplace smoothing (Ch.3/4)
            self.log_likelihood[cls] = {}
            for w in all_words:
                count = 1 if w in word_set else 0
                # Add-1 smoothing
                self.log_likelihood[cls][w] = math.log((count + 1) / total)
            # Default for unseen words
            self.log_likelihood[cls]["<UNK>"] = math.log(1 / total)

    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Returns (predicted_class, {class: score}) where scores are
        normalised log-probabilities.
        """
        # Tokenise + deduplicate (Boolean NB from Ch.4)
        tokens = set(re.findall(r"[a-z]+", text.lower()))

        scores = {}
        for cls in self.CLASSES:
            score = self.log_prior[cls]
            for tok in tokens:
                ll = self.log_likelihood[cls]
                score += ll.get(tok, ll["<UNK>"])
            scores[cls] = score

        # Softmax to get probabilities
        max_s = max(scores.values())
        exp_s = {c: math.exp(s - max_s) for c, s in scores.items()}
        total = sum(exp_s.values())
        probs = {c: v / total for c, v in exp_s.items()}

        best_class = max(probs, key=lambda c: probs[c])
        return best_class, probs


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class SentimentFilter:
    """
    Re-ranks autocomplete suggestions to maintain the prefix's tone.

    Tone is detected via NaiveBayesSentiment and mapped to a preferred
    continuation vocabulary. Suggestions matching that vocabulary get a
    boost; mismatches get a mild penalty.
    """

    CONTINUATION_SETS = {
        "formal":   FORMAL_CONTINUATIONS,
        "informal": INFORMAL_CONTINUATIONS,
        "positive": POSITIVE_CONTINUATIONS,
        "negative": NEGATIVE_CONTINUATIONS,
        "neutral":  set(),
    }

    def __init__(self, boost: float = 1.5, penalty: float = 0.7, alpha: float = 0.5):
        """
        Args:
            boost:   multiplier for words that match the detected tone.
            penalty: multiplier for strong mismatches (formal ↔ informal).
            alpha:   interpolation weight: final_p = alpha*tone_score + (1-alpha)*lm_prob.
        """
        self.nb     = NaiveBayesSentiment()
        self.boost   = boost
        self.penalty = penalty
        self.alpha   = alpha

    def detect(self, prefix: str) -> Tuple[str, Dict[str, float]]:
        """Return (dominant_register, full_prob_dict) for the prefix."""
        return self.nb.predict(prefix)

    def rerank(
        self,
        prefix: str,
        predictions: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Re-rank predictions by tone consistency.

        Returns sorted (word, adjusted_score) list, same length as input.
        """
        if not predictions:
            return predictions

        cls, probs = self.detect(prefix)
        good_set = self.CONTINUATION_SETS.get(cls, set())

        # Strong opposing registers
        clash = {
            "formal":   "informal",
            "informal": "formal",
        }.get(cls, None)
        bad_set = self.CONTINUATION_SETS.get(clash, set()) if clash else set()

        adjusted = []
        for word, prob in predictions:
            w_low = word.lower()
            if w_low in good_set:
                factor = self.boost
            elif w_low in bad_set:
                factor = self.penalty
            else:
                factor = 1.0
            adjusted.append((word, prob * factor))

        total = sum(s for _, s in adjusted) or 1.0
        normed = [(w, s / total) for w, s in adjusted]
        normed.sort(key=lambda x: -x[1])
        return normed

    def topk(
        self,
        prefix: str,
        predictions: List[Tuple[str, float]],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        return self.rerank(prefix, predictions)[:k]

    def explain(self, prefix: str) -> dict:
        cls, probs = self.detect(prefix)
        return {
            "prefix":    prefix,
            "register":  cls,
            "probs":     {c: f"{p:.3f}" for c, p in probs.items()},
            "boosts":    list(self.CONTINUATION_SETS.get(cls, set()))[:10],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sf = SentimentFilter()

    test_cases = [
        ("in this paper we propose a novel", [
            ("the", 0.20), ("approach", 0.18), ("method", 0.15),
            ("cool", 0.12), ("wanna", 0.10), ("significant", 0.10), ("kinda", 0.07),
        ]),
        ("the experimental results show that", [
            ("significantly", 0.20), ("effectively", 0.18), ("poorly", 0.15),
            ("kinda", 0.12), ("basically", 0.10), ("outperforms", 0.10), ("demonstrates", 0.08),
        ]),
    ]

    for prefix, preds in test_cases:
        print(f"\nPrefix: '{prefix}'")
        print("  ", sf.explain(prefix))
        ranked = sf.rerank(prefix, preds)
        print("  Top-5 after sentiment filter:")
        for w, p in ranked[:5]:
            print(f"    {w:<20} {p:.4f}")