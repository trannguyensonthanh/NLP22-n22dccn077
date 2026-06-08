"""
ELMo-style contextual embeddings (simplified, train-from-scratch).

ELMo (Embeddings from Language Models) — Peters et al. 2018.
Key idea: unlike static Word2Vec/GloVe, ELMo creates a different
embedding for each word depending on its CONTEXT.

"bank" in "river bank"  → different vector than
"bank" in "bank account"

Architecture:
    Forward  LSTM: predict next word  (left → right)
    Backward LSTM: predict prev word  (right → left)
    ELMo embedding = weighted sum of:
        - character CNN (static, context-independent)
        - Layer 1 BiLSTM hidden states
        - Layer 2 BiLSTM hidden states

Usage:
    from src.elmo_embeddings import ELMoEmbedder, train_elmo

    # Train
    elmo = train_elmo("data/processed/train_small.txt")

    # Use as embedding layer
    ctx_emb = elmo.embed_sentence(["the", "quick", "brown", "fox"])
    # shape: (seq_len, elmo_dim)

    # Or as drop-in for LSTM input
    elmo_layer = elmo.as_torch_layer()

    python -m src.elmo_embeddings train
    python -m src.elmo_embeddings test --sentence "the quick brown fox"
"""

import argparse
import math
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

ELMO_CKPT = CKPT / "elmo"
ELMO_CKPT.mkdir(exist_ok=True)

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


# ---------------------------------------------------------------------------
# Vocabulary (shared with neural_model if available)
# ---------------------------------------------------------------------------

class Vocab:
    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    @classmethod
    def build(cls, path: Path, min_count: int = 3, max_size: int = 30_000):
        from collections import Counter
        counter = Counter()
        with path.open(encoding="utf-8") as f:
            for line in f:
                counter.update(line.strip().lower().split())
        stoi = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
        for w, c in counter.most_common(max_size):
            if c < min_count:
                break
            stoi[w] = len(stoi)
        return cls(stoi)


# ---------------------------------------------------------------------------
# BiLSTM Language Model (core of ELMo)
# ---------------------------------------------------------------------------

class BiLSTMLM(nn.Module):
    """
    Two-layer BiLSTM trained as a language model.
    Forward direction: predict next token.
    Backward direction: predict previous token.

    The hidden states from each layer become ELMo's contextual representations.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.33,
    ):
        super().__init__()
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        # Forward LSTM layers
        self.fwd_lstms = nn.ModuleList([
            nn.LSTM(
                emb_dim if i == 0 else hidden_dim,
                hidden_dim,
                batch_first=True,
            )
            for i in range(n_layers)
        ])

        # Backward LSTM layers
        self.bwd_lstms = nn.ModuleList([
            nn.LSTM(
                emb_dim if i == 0 else hidden_dim,
                hidden_dim,
                batch_first=True,
            )
            for i in range(n_layers)
        ])

        # Output projections (fwd + bwd → vocab)
        self.fwd_head = nn.Linear(hidden_dim, vocab_size)
        self.bwd_head = nn.Linear(hidden_dim, vocab_size)

        # ELMo scalar weights (γ and s_k per layer — learned)
        # weights: [layer0_weight, layer1_weight, layer2_weight]  (layer0 = embedding)
        self.elmo_weights = nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))
        self.elmo_gamma   = nn.Parameter(torch.ones(1))

    def forward(self, ids: torch.Tensor):
        """
        ids: (B, T)
        Returns:
            fwd_logits: (B, T, V)
            bwd_logits: (B, T, V)
            layer_reps:  list of (B, T, 2*hidden)  — one per BiLSTM layer
        """
        emb = self.dropout(self.embedding(ids))           # (B, T, E)

        # Reversed input for backward LSTM
        ids_rev = torch.flip(ids, dims=[1])
        emb_rev = self.dropout(self.embedding(ids_rev))

        fwd_h = emb
        bwd_h = emb_rev
        layer_reps = [torch.cat([emb, emb_rev], dim=-1)]  # layer-0 representation

        for fwd_lstm, bwd_lstm in zip(self.fwd_lstms, self.bwd_lstms):
            fwd_h, _ = fwd_lstm(fwd_h)
            fwd_h    = self.dropout(fwd_h)

            bwd_h, _ = bwd_lstm(bwd_h)
            bwd_h    = self.dropout(bwd_h)

            # Align backward (flip back to original order)
            bwd_aligned = torch.flip(bwd_h, dims=[1])
            layer_reps.append(torch.cat([fwd_h, bwd_aligned], dim=-1))

        fwd_logits = self.fwd_head(fwd_h)
        bwd_logits = self.bwd_head(bwd_h)
        return fwd_logits, bwd_logits, layer_reps

    def elmo_embed(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Compute ELMo contextual embedding for input ids.
        Returns: (B, T, 2*hidden_dim)  — weighted sum of layer representations.
        """
        with torch.no_grad():
            _, _, layer_reps = self.forward(ids)

        # Softmax-normalize scalar weights
        weights = torch.softmax(self.elmo_weights, dim=0)  # (n_layers+1,)

        # Weighted sum
        stacked = torch.stack(layer_reps, dim=0)          # (n_layers+1, B, T, 2H)
        elmo    = self.elmo_gamma * (weights.view(-1, 1, 1, 1) * stacked).sum(0)
        return elmo   # (B, T, 2*hidden_dim)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LMDataset(Dataset):
    def __init__(self, path: Path, vocab: Vocab, max_len: int = 50):
        self.examples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                toks = [BOS] + line.strip().lower().split() + [EOS]
                if len(toks) > max_len:
                    toks = toks[:max_len]
                if len(toks) < 4:
                    continue
                self.examples.append(torch.tensor(vocab.encode(toks), dtype=torch.long))

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    n_layers: int = 2,
    train_file: str = "train_small.txt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ELMo] Device: {device}")

    train_path = PROC / train_file
    val_path   = PROC / "val_small.txt"

    print("[ELMo] Building vocabulary…")
    vocab = Vocab.build(train_path, min_count=3, max_size=30_000)
    print(f"[ELMo]   vocab size: {len(vocab):,}")

    with (ELMO_CKPT / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)

    train_ds = LMDataset(train_path, vocab)
    val_ds   = LMDataset(val_path,   vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=2)

    model = BiLSTMLM(len(vocab), hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        # --- Train ---
        model.train()
        tr_loss, tr_tok = 0.0, 0
        for batch in tqdm(train_loader, desc=f"ELMo Epoch {ep}", leave=False):
            batch = batch.to(device)
            inp = batch[:, :-1]
            fwd_tgt = batch[:, 1:]                      # next token
            bwd_tgt = torch.flip(batch[:, :-1], [1])    # prev token (reversed)

            fwd_logits, bwd_logits, _ = model(inp)
            loss_fwd = loss_fn(fwd_logits.reshape(-1, fwd_logits.size(-1)),
                               fwd_tgt.reshape(-1))
            loss_bwd = loss_fn(bwd_logits.reshape(-1, bwd_logits.size(-1)),
                               bwd_tgt.reshape(-1))
            loss = (loss_fwd + loss_bwd) / 2

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n = (fwd_tgt != 0).sum().item()
            tr_loss += loss.item() * n
            tr_tok  += n

        # --- Validation ---
        model.eval()
        vl_loss, vl_tok = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inp = batch[:, :-1]
                fwd_tgt = batch[:, 1:]
                fwd_logits, _, _ = model(inp)
                loss = loss_fn(fwd_logits.reshape(-1, fwd_logits.size(-1)),
                               fwd_tgt.reshape(-1))
                n = (fwd_tgt != 0).sum().item()
                vl_loss += loss.item() * n
                vl_tok  += n

        tr_ppl = math.exp(tr_loss / max(tr_tok, 1))
        vl_ppl = math.exp(vl_loss / max(vl_tok, 1))
        print(f"[ELMo] Epoch {ep}: train_ppl={tr_ppl:.1f} | val_ppl={vl_ppl:.1f}")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), ELMO_CKPT / "elmo_best.pt")
            print("[ELMo]   ✓ saved best")

    print("[ELMo] Training complete.")
    return model, vocab


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model() -> Tuple["BiLSTMLM", "Vocab", torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (ELMO_CKPT / "vocab.pkl").open("rb") as f:
        vocab = pickle.load(f)
    model = BiLSTMLM(len(vocab))
    model.load_state_dict(torch.load(ELMO_CKPT / "elmo_best.pt", map_location=device))
    model.to(device).eval()
    return model, vocab, device


# ---------------------------------------------------------------------------
# ELMo Embedder (inference API)
# ---------------------------------------------------------------------------

class ELMoEmbedder:
    """
    High-level API for getting ELMo contextual embeddings.

    Usage:
        embedder = ELMoEmbedder()
        emb = embedder.embed_sentence(["the", "bank", "is", "open"])
        # emb: (4, elmo_dim) tensor
    """

    def __init__(self, model=None, vocab=None, device=None):
        if model is None:
            model, vocab, device = load_model()
        self.model  = model
        self.vocab  = vocab
        self.device = device
        self.dim    = model.hidden_dim * 2

    @torch.no_grad()
    def embed_sentence(self, tokens: List[str]) -> torch.Tensor:
        """
        tokens: list of strings (no BOS/EOS needed)
        Returns: (len(tokens), elmo_dim) tensor
        """
        toks_with_bos = [BOS] + [t.lower() for t in tokens] + [EOS]
        ids = torch.tensor(
            [self.vocab.encode(toks_with_bos)], dtype=torch.long, device=self.device
        )
        elmo_emb = self.model.elmo_embed(ids)  # (1, T, elmo_dim)
        # Trim BOS/EOS
        return elmo_emb[0, 1:-1, :]            # (len(tokens), elmo_dim)

    @torch.no_grad()
    def embed_batch(self, sentences: List[List[str]]) -> List[torch.Tensor]:
        return [self.embed_sentence(s) for s in sentences]

    def as_torch_layer(self) -> "ELMoLayer":
        """Return a nn.Module that wraps this embedder for use in other models."""
        return ELMoLayer(self)


class ELMoLayer(nn.Module):
    """
    Wraps ELMoEmbedder as a PyTorch layer.
    Takes word-index tensors, returns ELMo contextual embeddings.
    Input:  (B, T) LongTensor of word indices (from ELMo's own vocab)
    Output: (B, T, elmo_dim)
    """

    def __init__(self, embedder: ELMoEmbedder):
        super().__init__()
        self.embedder = embedder
        self.out_dim  = embedder.dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedder.model.elmo_embed(ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_test(sentence: str):
    embedder = ELMoEmbedder()
    tokens   = sentence.lower().split()
    emb      = embedder.embed_sentence(tokens)
    print(f"\nSentence: {sentence}")
    print(f"ELMo embedding shape: {tuple(emb.shape)}  (words × elmo_dim)")
    for i, tok in enumerate(tokens):
        norm = emb[i].norm().item()
        print(f"  [{i:2d}] {tok:<20} ||emb|| = {norm:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs",     type=int,   default=5)
    t.add_argument("--batch-size", type=int,   default=64)
    t.add_argument("--hidden",     type=int,   default=256)
    t.add_argument("--layers",     type=int,   default=2)
    t.add_argument("--lr",         type=float, default=1e-3)
    t.add_argument("--file",       default="train_small.txt")

    ts = sub.add_parser("test")
    ts.add_argument("--sentence", default="the quick brown fox")

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.epochs, args.batch_size, args.lr, args.hidden, args.layers, args.file)
    else:
        cli_test(args.sentence)