"""
mamba_model.py — Mamba Language Model (Selective State Space Model)

Gu & Dao 2023 — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

Mamba replaces attention with a selective SSM:
    - O(n) inference (vs O(n²) for Transformer)
    - Context grows unboundedly (vs fixed window)
    - Input-dependent state transitions (vs fixed HMM)
    - No attention matrix → memory efficient

Core innovation: A, B, C parameters are FUNCTIONS of the input x,
not fixed matrices. This "selectivity" lets the model choose what to
remember vs forget based on content.

Simplified implementation (no CUDA kernel required):
    Uses PyTorch-native scan (sequential for small sequences,
    parallel prefix-sum approximation for long ones).

Architecture per Mamba block:
    x → Linear expand → SSM (selective) → Gate → Linear contract → residual

Usage:
    python -m src.mamba_model train --layers 4
    python -m src.mamba_model predict --prefix "the quick brown fox"
"""

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


# ===========================================================================
# Vocabulary (reuse pattern)
# ===========================================================================

class Vocab:
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

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


# ===========================================================================
# SSM Core (Selective State Space)
# ===========================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space layer — core of Mamba.

    Key equations:
        h_t = A_t * h_{t-1} + B_t * x_t      (state update)
        y_t = C_t * h_t                        (output)

    where A_t, B_t, C_t are INPUT-DEPENDENT (selective).
    In contrast, classic SSMs have fixed A, B, C (like S4).

    Implementation:
        A_t = softplus(Linear_A(x_t))  — must be negative (stable)
        B_t = Linear_B(x_t)
        C_t = Linear_C(x_t)
        Δ_t = softplus(Linear_delta(x_t))  — discretisation step size
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Selective parameters — functions of input
        self.W_B     = nn.Linear(d_model, d_state, bias=False)
        self.W_C     = nn.Linear(d_model, d_state, bias=False)
        self.W_delta = nn.Linear(d_model, d_model, bias=True)

        # Fixed but learnable A (log-parameterised for stability)
        # A must be positive; we store log(A) and exponentiate
        self.log_A   = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float)
                      .unsqueeze(0).expand(d_model, -1))
        )

        # Causal depthwise conv (short-range local dependency)
        self.conv1d  = nn.Conv1d(
            d_model, d_model, kernel_size=d_conv, padding=d_conv - 1,
            groups=d_model, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        returns: (B, T, d_model)
        """
        B, T, D = x.shape
        N = self.d_state

        # Short depthwise conv (local context)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective parameters (all input-dependent)
        B_t  = self.W_B(x_conv)           # (B, T, N)
        C_t  = self.W_C(x_conv)           # (B, T, N)
        # Δ_t: discretisation step (positive via softplus)
        delta = F.softplus(self.W_delta(x_conv))   # (B, T, D)

        # Discretise A: A_bar = exp(-exp(log_A) * delta)
        A = -torch.exp(self.log_A)        # (D, N)  — negative for stability
        # dA: (B, T, D, N)
        dA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        # dB: (B, T, D, N)
        dB = delta.unsqueeze(-1) * B_t.unsqueeze(2)

        # Sequential SSM scan (causal)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            # y_t = sum_n C_t[n] * h[n]
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, D)
        return y


# ===========================================================================
# Mamba Block
# ===========================================================================

class MambaBlock(nn.Module):
    """
    One Mamba residual block.

    Structure:
        x → LayerNorm → expand → SSM → gate → contract → + residual
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        d_inner     = d_model * expand

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)  # x + gate
        self.ssm      = SelectiveSSM(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        # Project to expanded + gate
        xz   = self.in_proj(x)                      # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)               # (B, T, d_inner) each

        # SSM on expanded features
        y    = self.ssm(x_in)                        # (B, T, d_inner)

        # Gating (SiLU gate from Mamba paper)
        y    = y * F.silu(z)

        return self.out_proj(y) + residual           # (B, T, d_model)


# ===========================================================================
# Mamba Language Model
# ===========================================================================

class MambaLM(nn.Module):
    """
    Full Mamba Language Model.

    Architecture:
        Embedding → N × MambaBlock → LayerNorm → Linear(vocab)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 256,
        n_layers:   int = 4,
        d_state:    int = 16,
        expand:     int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        self.layers    = nn.ModuleList([
            MambaBlock(d_model, d_state, expand)
            for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embedding.weight

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T)
        returns logits: (B, T, vocab_size)
        """
        h = self.dropout(self.embedding(x))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


# ===========================================================================
# Dataset (reusable pattern)
# ===========================================================================

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
                self.examples.append(
                    torch.tensor(vocab.encode(tokens), dtype=torch.long)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def _collate(batch):
    maxlen = max(len(x) for x in batch)
    out = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, :len(x)] = x
    return out


# ===========================================================================
# Training
# ===========================================================================

def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    d_model: int = 256,
    n_layers: int = 4,
    d_state: int = 16,
    train_file: str = "train_small.txt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Mamba] Device: {device}")

    train_path = PROC / train_file
    val_path   = PROC / "val_small.txt"

    print("[Mamba] Building vocab…")
    with train_path.open() as f:
        vocab = Vocab.build(f, min_count=3, max_size=50_000)
    print(f"[Mamba]   vocab: {len(vocab):,}")

    with (CKPT / "mamba_vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)

    train_ds = SentenceDataset(train_path, vocab)
    val_ds   = SentenceDataset(val_path,   vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=2)

    model   = MambaLM(len(vocab), d_model=d_model, n_layers=n_layers, d_state=d_state).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Mamba]   params: {n_params:,}")

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss, tr_tok = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Mamba Epoch {ep}", leave=False):
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits  = model(inp)
            loss    = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = (tgt != 0).sum().item()
            tr_loss += loss.item() * n
            tr_tok  += n

        scheduler.step()

        # Validate
        model.eval()
        vl_loss, vl_tok = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inp, tgt = batch[:, :-1], batch[:, 1:]
                logits = model(inp)
                loss   = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                n = (tgt != 0).sum().item()
                vl_loss += loss.item() * n
                vl_tok  += n

        tr_ppl = math.exp(tr_loss / max(tr_tok, 1))
        vl_ppl = math.exp(vl_loss / max(vl_tok, 1))
        print(f"[Mamba] Epoch {ep}: train_ppl={tr_ppl:.1f} | val_ppl={vl_ppl:.1f}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), CKPT / "mamba_best.pt")
            print("[Mamba]   ✓ saved")

    print("[Mamba] Training complete.")


# ===========================================================================
# Load
# ===========================================================================

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (CKPT / "mamba_vocab.pkl").open("rb") as f:
        vocab = pickle.load(f)
    model = MambaLM(len(vocab))
    model.load_state_dict(torch.load(CKPT / "mamba_best.pt", map_location=device))
    model.to(device).eval()
    return model, vocab, device


# ===========================================================================
# Predict
# ===========================================================================

@torch.no_grad()
def predict_next(
    model: MambaLM,
    vocab: Vocab,
    device: torch.device,
    prefix: str,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    tokens = [BOS] + prefix.lower().split()
    ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)
    logits = model(ids)                           # (1, T, V)
    probs  = torch.softmax(logits[0, -1], dim=-1)
    top    = torch.topk(probs, top_k + 4)

    results = []
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
        word = vocab.itos[idx]
        if word in (PAD, BOS, EOS, UNK):
            continue
        results.append((word, p))
        if len(results) >= top_k:
            break
    return results


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs",     type=int, default=5)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--d-model",    type=int, default=256)
    t.add_argument("--layers",     type=int, default=4)
    t.add_argument("--d-state",    type=int, default=16)
    t.add_argument("--lr",         type=float, default=1e-3)
    t.add_argument("--file",       default="train_small.txt")

    p = sub.add_parser("predict")
    p.add_argument("--prefix", required=True)
    p.add_argument("--top-k",  type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.epochs, args.batch_size, args.lr,
              args.d_model, args.layers, args.d_state, args.file)
    else:
        m, v, d = load_model()
        preds   = predict_next(m, v, d, args.prefix, args.top_k)
        print(f"\nPrefix: {args.prefix!r}")
        print("[Mamba] Top predictions:")
        for w, p in preds:
            print(f"  {w:<20} {p:.4f}")