"""
Download high-quality English text data for autocomplete training.

Sources (all chosen for grammatical correctness & good writing style):
  1. WikiText-103  : Wikipedia "Good and Featured" articles, ~100M tokens
  2. BBC News      : edited journalism, formal British English
  3. arXiv abstracts: academic writing, dense and well-structured

Output: data/raw/{wikitext,bbc,arxiv}.txt  (one document per line)
"""
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_lines(path: Path, lines):
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            line = line.strip()
            if line:
                f.write(line.replace("\n", " ") + "\n")


def download_wikitext():
    out = RAW_DIR / "wikitext.txt"
    if out.exists():
        print(f"[skip] {out.name} already exists")
        return
    print("Downloading WikiText-103 (Featured/Good Wikipedia articles)...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    save_lines(out, (ex["text"] for ex in tqdm(ds)))
    print(f"  saved {out}")


def download_bbc():
    out = RAW_DIR / "bbc.txt"
    if out.exists():
        print(f"[skip] {out.name} already exists")
        return
    print("Downloading BBC News (edited journalism)...")
    # SetFit/bbc-news: BBC articles split into categories
    ds = load_dataset("SetFit/bbc-news", split="train")
    save_lines(out, (ex["text"] for ex in tqdm(ds)))
    print(f"  saved {out}")


def download_arxiv():
    out = RAW_DIR / "arxiv.txt"
    if out.exists():
        print(f"[skip] {out.name} already exists")
        return
    print("Downloading arXiv abstracts (academic writing)...")
    # ccdv/arxiv-classification has clean abstracts
    ds = load_dataset("ccdv/arxiv-classification", "no_ref", split="train",
                      trust_remote_code=True)
    save_lines(out, (ex["text"][:2000] for ex in tqdm(ds)))  # cap length
    print(f"  saved {out}")


if __name__ == "__main__":
    download_wikitext()
    download_bbc()
    try:
        download_arxiv()
    except Exception as e:
        print(f"[warn] arxiv download failed: {e}")
        print("       continuing without arxiv — wikitext + bbc is enough")
    print("\nDone. Raw files in:", RAW_DIR)
