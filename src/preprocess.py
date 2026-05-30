"""
Preprocess raw text for autocomplete training.

Pipeline:
  1. Load raw files from data/raw/*.txt
  2. Sentence-split
  3. Quality filter: length, non-ASCII ratio, repetition, optional grammar check
  4. Lowercase + light normalization
  5. Train/val/test split (90/5/5)
  6. Save to data/processed/{train,val,test}.txt

Run:  python -m src.preprocess  [--grammar-check]
"""
import argparse
import random
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# --- filters ----------------------------------------------------------------

WIKI_HEADING = re.compile(r"^\s*=.*=\s*$")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE = re.compile(r"\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

MIN_WORDS = 6
MAX_WORDS = 50


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub("", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text


def is_clean(sent: str) -> bool:
    if WIKI_HEADING.match(sent):
        return False
    words = sent.split()
    n = len(words)
    if n < MIN_WORDS or n > MAX_WORDS:
        return False
    # too many non-letters → likely table/list residue
    letters = sum(c.isalpha() for c in sent)
    if letters / max(len(sent), 1) < 0.7:
        return False
    # excessive repetition
    if len(set(words)) / n < 0.5:
        return False
    # must end with sentence punctuation
    if not sent.rstrip().endswith((".", "!", "?")):
        return False
    return True


def iter_sentences(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = normalize(line)
            if not line:
                continue
            for sent in SENT_SPLIT.split(line):
                yield sent.strip()


def maybe_grammar_filter(sentences, enabled: bool):
    """Optionally drop sentences with grammar errors using language_tool_python."""
    if not enabled:
        yield from sentences
        return
    try:
        import language_tool_python
    except ImportError:
        print("[warn] language_tool_python not installed — skipping grammar check")
        yield from sentences
        return
    tool = language_tool_python.LanguageTool("en-US")
    for s in sentences:
        try:
            if not tool.check(s):
                yield s
        except Exception:
            yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar-check", action="store_true",
                    help="Filter sentences with grammar errors (slow)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        raise SystemExit(f"No raw files in {RAW_DIR}. Run download_data.py first.")

    print(f"Loading from: {[p.name for p in raw_files]}")
    all_sents = []
    for path in raw_files:
        kept = 0
        for sent in tqdm(iter_sentences(path), desc=path.name):
            if is_clean(sent):
                all_sents.append(sent.lower())
                kept += 1
        print(f"  {path.name}: kept {kept} sentences")

    if args.grammar_check:
        print("Running grammar filter (this is slow)...")
        all_sents = list(tqdm(maybe_grammar_filter(all_sents, True),
                              total=len(all_sents)))

    print(f"\nTotal clean sentences: {len(all_sents):,}")

    rng = random.Random(args.seed)
    rng.shuffle(all_sents)
    n = len(all_sents)
    n_val = n_test = n // 20  # 5% each
    splits = {
        "train": all_sents[: n - 2 * n_val],
        "val":   all_sents[n - 2 * n_val : n - n_val],
        "test":  all_sents[n - n_val :],
    }
    for name, data in splits.items():
        out = PROC_DIR / f"{name}.txt"
        out.write_text("\n".join(data) + "\n", encoding="utf-8")
        print(f"  {name}: {len(data):,} sentences -> {out}")


if __name__ == "__main__":
    main()
