"""
src/rnn_model.py  —  Basic Elman RNN Language Model
=====================================================

Mục đích: so sánh với LSTM để thấy rõ sự khác biệt.

Tại sao cần RNN basic?
  - LSTM giải quyết vanishing gradient bằng cell state + 3 gates
  - RNN không có gates — chỉ có 1 phép biến đổi đơn giản:
      h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
  - Kết quả: RNN bị vanishing gradient, PPL kém hơn LSTM (~200-300 vs ~118)
  - Nhưng RNN train nhanh hơn LSTM vì:
      * Ít tham số hơn (không có 3 gates)
      * Mỗi bước tính toán đơn giản hơn ~4x
      * ~15-20 phút trên CPU (so với ~30-60 phút của LSTM)

So sánh kiến trúc:
  RNN:  x_t → [W_xh] → + → [tanh] → h_t → [W_hh] → (lặp lại)
  LSTM: x_t → [Forget Gate, Input Gate, Output Gate, Cell Gate] → h_t, c_t

Cùng interface với LSTM để dễ dàng thêm vào evaluate.py.

Usage:
    python -m src.rnn_model train
    python -m src.rnn_model predict --prefix "the quick brown"
    python -m src.rnn_model evaluate
"""

import argparse
import math
import pickle
import time
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Dùng chung Vocab và SentenceDataset từ neural_model
from .neural_model import (
    BOS, EOS, PAD, UNK,
    Vocab,
    SentenceDataset,
    collate,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)


# ============================================================
# ELMAN RNN MODEL
# ============================================================

class RNNModel(nn.Module):
    """
    Elman RNN Language Model.

    Kiến trúc:
        Embedding(vocab_size, emb_dim)
        ↓  x_t: (batch, emb_dim)
        nn.RNN(emb_dim, hidden, num_layers, dropout)
        ↓  h_t: (batch, hidden)
        Dropout
        ↓
        Linear(hidden, vocab_size)
        ↓  logits: (batch, vocab_size)

    Công thức RNN (mỗi bước thời gian t):
        h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
        output_t = Linear(h_t)

    Khác với LSTM (4 gates):
        forget_gate = sigmoid(W_if*x + b_if + W_hf*h + b_hf)
        input_gate  = sigmoid(W_ii*x + b_ii + W_hi*h + b_hi)
        cell_gate   = tanh(   W_ig*x + b_ig + W_hg*h + b_hg)
        output_gate = sigmoid(W_io*x + b_io + W_ho*h + b_ho)
        c_t = forget*c_{t-1} + input*cell
        h_t = output * tanh(c_t)

    → LSTM có 4x phép tính so với RNN tại mỗi bước.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim:    int = 256,
        hidden:     int = 512,
        layers:     int = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.emb_dim  = emb_dim
        self.hidden   = hidden
        self.layers   = layers

        # Embedding: ID → dense vector
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # Elman RNN — đây là điểm khác biệt chính so với LSTMLM
        # nonlinearity='tanh' là mặc định (giống công thức bài giảng)
        self.rnn = nn.RNN(
            input_size  = emb_dim,
            hidden_size = hidden,
            num_layers  = layers,
            dropout     = dropout if layers > 1 else 0.0,
            batch_first = True,
            nonlinearity = "tanh",  # h_t = tanh(W*x + U*h + b)
        )

        # Dropout trước output head
        self.drop = nn.Dropout(dropout)

        # Output head: hidden → vocabulary scores (logits)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x:      (batch, seq_len) — token IDs
        hidden: (layers, batch, hidden) — initial hidden state
        returns:
          logits: (batch, seq_len, vocab_size)
          hidden: (layers, batch, hidden) — updated hidden state
        """
        # Embedding lookup
        e = self.embedding(x)           # (batch, seq, emb_dim)

        # RNN forward pass
        # Tất cả bước thời gian được tính song song (unrolled loop)
        out, hidden = self.rnn(e, hidden)   # out: (batch, seq, hidden)

        # Dropout + linear head
        logits = self.head(self.drop(out))  # (batch, seq, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        """Khởi tạo hidden state bằng zeros."""
        return torch.zeros(self.layers, batch_size, self.hidden, device=device)


# ============================================================
# TRAINING LOOP
# ============================================================

def epoch_loop(
    model:    RNNModel,
    loader:   DataLoader,
    opt:      torch.optim.Optimizer,
    device:   torch.device,
    train:    bool,
    grad_clip: float = 1.0,
) -> float:
    """Một epoch training hoặc validation. Trả về average loss."""
    model.train() if train else model.eval()
    loss_fn     = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
    total_loss  = 0.0
    n_batches   = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(
            loader,
            desc="Train" if train else "Valid",
            leave=False,
        )

        for batch in pbar:
            batch = batch.to(device)               # (B, T)
            inp   = batch[:, :-1]                  # input: tất cả trừ token cuối
            tgt   = batch[:, 1:]                   # target: shift 1 bước

            # Forward pass
            logits, _ = model(inp)                 # (B, T-1, vocab)

            # Tính cross-entropy loss
            B, T, V = logits.shape
            loss = loss_fn(
                logits.reshape(B * T, V),          # (B*T, vocab)
                tgt.reshape(B * T),                # (B*T,)
            )

            if train:
                opt.zero_grad()
                loss.backward()
                # Gradient clipping — quan trọng với RNN vì dễ exploding gradient
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}"
            })

    return total_loss / max(n_batches, 1)


def train(
    epochs:     int   = 5,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    hidden:     int   = 512,
    layers:     int   = 2,
    dropout:    float = 0.3,
    train_file: str   = "train_small.txt",
):
    """
    Train RNN Language Model.

    Tại sao nhanh hơn LSTM?
    - RNN: 1 gate (tanh) thay vì 4 gates
    - Số phép tính mỗi bước ≈ 1/4 của LSTM
    - Ước tính: ~15-20 phút trên CPU (LSTM ~30-60 phút)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RNN] Device: {device}")

    # Build vocab (dùng chung train file với LSTM)
    train_path = PROC / train_file
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}")

    print("[RNN] Building vocabulary…")
    with train_path.open(encoding="utf-8") as f:
        vocab = Vocab.build(f, min_count=3, max_size=50_000)
    print(f"  Vocab size: {len(vocab):,}")

    # Lưu vocab (dùng chung với LSTM vocab format)
    vocab_path = CKPT / "rnn_vocab.pkl"
    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)

    # DataLoaders
    train_ds = SentenceDataset(train_path, vocab)
    val_path = PROC / "val_small.txt"
    if val_path.exists():
        val_ds = SentenceDataset(val_path, vocab)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                collate_fn=collate, shuffle=False)
    else:
        val_loader = None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              collate_fn=collate, shuffle=True)

    # Model
    model = RNNModel(
        vocab_size = len(vocab),
        emb_dim    = 256,
        hidden     = hidden,
        layers     = layers,
        dropout    = dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[RNN] Parameters: {total_params:,}")
    print(f"      (LSTM would have ~{total_params * 4:,} — 4x more due to 4 gates)")

    # Optimizer — AdamW như LSTM
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Training loop
    best_val  = math.inf
    ckpt_path = CKPT / "rnn_best.pt"
    t_start   = time.perf_counter()

    for ep in range(1, epochs + 1):
        t0       = time.perf_counter()
        tr_loss  = epoch_loop(model, train_loader, opt, device, train=True)

        if val_loader:
            val_loss = epoch_loop(model, val_loader, opt, device, train=False)
        else:
            val_loss = tr_loss

        tr_ppl  = math.exp(tr_loss)
        val_ppl = math.exp(val_loss)
        elapsed = time.perf_counter() - t0

        print(
            f"  Epoch {ep}/{epochs} | "
            f"train_loss={tr_loss:.4f} (PPL={tr_ppl:.1f}) | "
            f"val_loss={val_loss:.4f} (PPL={val_ppl:.1f}) | "
            f"{elapsed:.0f}s",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best checkpoint (val_ppl={val_ppl:.2f})")

    total_time = time.perf_counter() - t_start
    print(f"[RNN] Training complete in {total_time/60:.1f} min")
    print(f"[RNN] Best val PPL: {math.exp(best_val):.2f}")
    return model, vocab


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load trained RNN model + vocab. Same interface as neural_model."""
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = CKPT / "rnn_vocab.pkl"
    ckpt_path  = CKPT / "rnn_best.pt"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}. Run train first.")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train first.")

    with vocab_path.open("rb") as f:
        vocab = pickle.load(f)

    model = RNNModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model, vocab, device


# ============================================================
# PREDICT NEXT WORD
# ============================================================

@torch.no_grad()
def predict_next(
    model:   RNNModel,
    vocab:   Vocab,
    device,
    prefix:  str,
    top_k:   int = 5,
) -> list:
    """
    Dự đoán top-k từ tiếp theo.
    Same interface as neural_model.predict_next().

    prefix: chuỗi text
    returns: [(word, prob), ...]
    """
    model.eval()
    tokens = [BOS] + prefix.lower().split()
    ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)

    logits, _ = model(ids)                       # (1, T, vocab_size)
    last_logits = logits[0, -1, :]               # (vocab_size,) — vị trí cuối
    probs       = F.softmax(last_logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, top_k + 10)
    results = []
    skip    = {vocab.stoi.get(BOS, -1),
               vocab.stoi.get(EOS, -1),
               vocab.stoi.get(PAD, -1)}

    for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
        if idx in skip:
            continue
        word = vocab.decode([idx])[0] if hasattr(vocab, "decode") else vocab.itos[idx]
        results.append((word, prob))
        if len(results) >= top_k:
            break

    return results


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(limit: int = 500) -> dict:
    """
    Tính PPL + Top-k accuracy + MRR trên test set.
    Trả về dict kết quả — cùng format với eval_lstm() trong evaluate.py.
    """
    model, vocab, device = load_model()
    model.eval()

    test_path = PROC / "test.txt"
    with test_path.open(encoding="utf-8") as f:
        sents = [l.strip().split() for l in f if l.strip()][:limit]

    ks         = (1, 3, 5)
    loss_fn    = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    hits       = {k: 0 for k in ks}
    ranks      = []
    total_loss = 0.0
    total_tok  = 0
    preds_n    = 0

    for sent in sents:
        tokens = [BOS] + sent + [EOS]
        ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)
        inp, tgt = ids[:, :-1], ids[:, 1:]

        logits, _ = model(inp)              # (1, T-1, vocab)
        total_loss += loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
        ).item()
        total_tok += (tgt != 0).sum().item()

        topk_ids = torch.topk(logits[0], max(ks), dim=-1).indices.tolist()
        for pos in range(min(len(sent), logits.size(1))):
            true_id = tgt[0, pos].item()
            if true_id == 0:
                continue
            row = topk_ids[pos]
            for k in ks:
                if true_id in row[:k]:
                    hits[k] += 1
            try:
                ranks.append(row.index(true_id) + 1)
            except ValueError:
                ranks.append(0)
            preds_n += 1

    ppl  = math.exp(total_loss / max(total_tok, 1))
    accs = {k: hits[k] / max(preds_n, 1) for k in ks}
    mrr  = sum(1.0 / r if r > 0 else 0.0 for r in ranks) / max(len(ranks), 1)

    print(f"[RNN] PPL: {ppl:.2f}")
    for k in ks:
        print(f"  Top-{k}: {accs[k]:.4f}")
    print(f"  MRR: {mrr:.4f}")

    return {
        "ppl":  ppl,
        "top1": accs[1], "top3": accs[3], "top5": accs[5],
        "mrr":  mrr,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs",  type=int,   default=5)
    t.add_argument("--batch",   type=int,   default=64)
    t.add_argument("--lr",      type=float, default=1e-3)
    t.add_argument("--hidden",  type=int,   default=512)
    t.add_argument("--layers",  type=int,   default=2)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--file",    default="train_small.txt")

    p = sub.add_parser("predict")
    p.add_argument("--prefix",  required=True)
    p.add_argument("--top-k",   type=int, default=5)

    e = sub.add_parser("evaluate")
    e.add_argument("--limit",   type=int, default=500)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.epochs, args.batch, args.lr,
              args.hidden, args.layers, args.dropout, args.file)
    elif args.cmd == "predict":
        m, v, d = load_model()
        for w, p in predict_next(m, v, d, args.prefix, args.top_k):
            print(f"  {w:<20} {p:.4f}")
    elif args.cmd == "evaluate":
        evaluate(args.limit)