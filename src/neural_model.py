"""
LSTM language model for next-word prediction.

Architecture: embedding -> 2-layer LSTM -> linear projection over vocab.
Trained on word-level tokens with cross-entropy.

Usage:
    python -m src.neural_model train  --epochs 5
    python -m src.neural_model predict --prefix "the quick brown"
"""
import argparse
import math
import pickle
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


# --- vocab ------------------------------------------------------------------

class Vocab:
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    @classmethod
    def build(cls, sentences, min_count=3, max_size=50_000):
        counter = Counter()
        for sent in sentences:
            counter.update(sent.split())
        stoi = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
        for word, c in counter.most_common(max_size):
            if c < min_count:
                break
            stoi[word] = len(stoi)
        return cls(stoi)


# --- dataset ----------------------------------------------------------------

class SentenceDataset(Dataset):
    def __init__(self, path: Path, vocab: Vocab, max_len: int = 60):
        self.examples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                tokens = [BOS] + line.strip().split() + [EOS]
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                if len(tokens) < 3:
                    continue
                self.examples.append(torch.tensor(vocab.encode(tokens),
                                                  dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch):
    lens = [len(x) for x in batch]
    maxlen = max(lens)
    out = torch.zeros(len(batch), maxlen, dtype=torch.long)  # 0 = PAD
    for i, x in enumerate(batch):
        out[i, : len(x)] = x
    return out


# --- model ------------------------------------------------------------------

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden=512, layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        e = self.drop(self.emb(x))
        out, hidden = self.lstm(e, hidden)
        logits = self.head(self.drop(out))
        return logits, hidden


# --- train / eval -----------------------------------------------------------

def epoch_loop(model, loader, opt, device, train: bool):
    model.train() if train else model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total, count = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           tgt.reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tokens = (tgt != 0).sum().item()
            total += loss.item() * tokens
            count += tokens
    return total / max(count, 1)


def train(epochs=5, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_path = PROC / "train_small.txt"
    val_path = PROC / "val_small.txt"
    print("Building vocab...")
    with train_path.open(encoding="utf-8") as f:
        vocab = Vocab.build(f, min_count=3, max_size=50_000)
    print(f"  vocab size: {len(vocab):,}")
    with (CKPT / "lstm_vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)

    train_ds = SentenceDataset(train_path, vocab)
    val_ds = SentenceDataset(val_path, vocab)
    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)

    model = LSTMLM(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        tr_loss = epoch_loop(model, train_loader, opt, device, train=True)
        vl_loss = epoch_loop(model, val_loader, opt, device, train=False)
        print(f"Epoch {ep}: train_loss={tr_loss:.3f} (ppl={math.exp(tr_loss):.1f}) "
              f"| val_loss={vl_loss:.3f} (ppl={math.exp(vl_loss):.1f})")
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), CKPT / "lstm_best.pt")
            print("  saved best checkpoint")


def load_model():
    with (CKPT / "lstm_vocab.pkl").open("rb") as f:
        vocab = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMLM(len(vocab)).to(device)
    model.load_state_dict(torch.load(CKPT / "lstm_best.pt", map_location=device))
    model.eval()
    return model, vocab, device


@torch.no_grad()
def predict_next(model, vocab, device, prefix: str, top_k=5):
    tokens = [BOS] + prefix.lower().split()
    ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)
    logits, _ = model(ids)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, top_k + 4)  # extra for filtering specials
    results = []
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        word = vocab.itos[idx]
        if word in (PAD, BOS, EOS, UNK):
            continue
        results.append((word, p))
        if len(results) >= top_k:
            break
    return results


def cli_predict(prefix: str, top_k: int = 5):
    model, vocab, device = load_model()
    preds = predict_next(model, vocab, device, prefix, top_k)
    print(f"\nPrefix: {prefix!r}")
    print(f"Top-{top_k} next-word predictions:")
    for w, p in preds:
        print(f"  {w:<20} {p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.epochs, args.batch_size, args.lr)
    else:
        cli_predict(args.prefix, args.top_k)
