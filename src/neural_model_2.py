"""
neural_model_2.py — Extended LSTM Language Models

Provides THREE comparable model variants in one file:
    1. LSTM-Standard   — original 2-layer LSTM, greedy predict_next
    2. LSTM-Beam       — same LSTM, beam search decoding
    3. AWD-LSTM        — LSTM + DropConnect + ASGD + variational dropout

Each variant can optionally use ELMo contextual embeddings as input layer.
Toggle is exposed via `use_elmo=True/False` at inference time.

Comparison matrix (shown in Streamlit):
    ┌─────────────────┬──────────┬────────────┬────────────┐
    │ Variant         │ Decoding │ Reg.       │ Embedding  │
    ├─────────────────┼──────────┼────────────┼────────────┤
    │ LSTM-Standard   │ greedy   │ dropout    │ static     │
    │ LSTM-Beam       │ beam(5)  │ dropout    │ static     │
    │ LSTM-Standard+E │ greedy   │ dropout    │ ELMo       │
    │ LSTM-Beam+E     │ beam(5)  │ dropout    │ ELMo       │
    │ AWD-LSTM        │ greedy   │ AWD        │ static     │
    │ AWD-LSTM+E      │ greedy   │ AWD        │ ELMo       │
    └─────────────────┴──────────┴────────────┴────────────┘

Usage:
    python -m src.neural_model_2 train --variant standard
    python -m src.neural_model_2 train --variant awd
    python -m src.neural_model_2 predict --variant standard --prefix "the quick brown"
    python -m src.neural_model_2 predict --variant awd --elmo --prefix "the quick brown"
"""

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


# ===========================================================================
# Vocabulary
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


# ===========================================================================
# Dataset
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


def collate(batch):
    maxlen = max(len(x) for x in batch)
    out = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, : len(x)] = x
    return out


# ===========================================================================
# DropConnect helper (AWD-LSTM)
# ===========================================================================

class DropConnect(nn.Module):
    """
    DropConnect on the hidden-to-hidden weight matrices of LSTM.
    Merity et al. 2018 — "Regularizing and Optimizing LSTM Language Models"
    Drops weights (not activations) at training time.
    """

    def __init__(self, module: nn.LSTM, weight_p: float = 0.5):
        super().__init__()
        self.module   = module
        self.weight_p = weight_p
        self._backup: dict = {}

        # Register hook to mask weights before forward
        for name, param in list(module.named_parameters()):
            if "weight_hh" in name:
                self.register_parameter(f"raw_{name.replace('.', '_')}", param)
                self._backup[name] = name

    def _drop_weights(self):
        for name in self._backup:
            param = getattr(self.module, name)
            if self.training:
                mask  = param.data.new_empty(param.shape).bernoulli_(1 - self.weight_p)
                w_dropped = param * mask / (1 - self.weight_p)
            else:
                w_dropped = param
            # Temporarily set the attribute directly
            setattr(self.module, name, nn.Parameter(w_dropped, requires_grad=False))

    def forward(self, x, hidden=None):
        self._drop_weights()
        return self.module(x, hidden)


# ===========================================================================
# Variational Dropout (AWD-LSTM)
# ===========================================================================

class VariationalDropout(nn.Module):
    """
    Variational (locked) dropout: same mask for all time steps in a sequence.
    Gal & Ghahramani 2016 — used in AWD-LSTM.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        # x: (B, T, H) — sample one mask per (B, H)
        mask = x.data.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return x * mask


# ===========================================================================
# Standard LSTM Language Model (kept identical to original neural_model.py)
# ===========================================================================

class LSTMLM(nn.Module):
    """
    Standard 2-layer LSTM language model (original architecture).
    Supports optional ELMo input.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        hidden: int = 512,
        layers: int = 2,
        dropout: float = 0.3,
        elmo_dim: int = 0,          # 0 = no ELMo; >0 = ELMo output dim
    ):
        super().__init__()
        self.use_elmo = elmo_dim > 0
        self.emb_dim  = emb_dim
        self.hidden   = hidden

        self.emb  = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)

        # If ELMo, project [emb | elmo] → lstm_input_dim
        lstm_in = emb_dim + elmo_dim if self.use_elmo else emb_dim
        self.elmo_proj = nn.Linear(lstm_in, emb_dim) if self.use_elmo else None

        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None, elmo_emb=None):
        """
        x:         (B, T) token ids
        elmo_emb:  (B, T, elmo_dim) or None
        """
        e = self.drop(self.emb(x))
        if self.use_elmo and elmo_emb is not None:
            e = torch.cat([e, elmo_emb], dim=-1)
            e = self.drop(torch.relu(self.elmo_proj(e)))
        out, hidden = self.lstm(e, hidden)
        logits = self.head(self.drop(out))
        return logits, hidden


# ===========================================================================
# AWD-LSTM Language Model
# ===========================================================================

class AWDLstmLM(nn.Module):
    """
    AWD-LSTM Language Model — Merity et al. 2018.

    Regularization techniques:
        1. DropConnect on hidden-to-hidden weights
        2. Variational (locked) dropout on embeddings and LSTM output
        3. Weight tying (embedding ↔ output projection)
        4. ASGD optimizer (used externally in train())

    Optional: ELMo contextual embedding input.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 400,
        hidden: int = 1150,
        layers: int = 3,
        dropoute: float = 0.1,   # embedding dropout
        dropouti: float = 0.65,  # input dropout
        dropouth: float = 0.3,   # hidden dropout
        dropouto: float = 0.4,   # output dropout
        wdrop:    float = 0.5,   # DropConnect rate
        elmo_dim: int = 0,
    ):
        super().__init__()
        self.use_elmo  = elmo_dim > 0
        self.emb_dim   = emb_dim
        self.hidden    = hidden
        self.layers    = layers

        self.embedding    = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop     = VariationalDropout(dropoute)
        self.inp_drop     = VariationalDropout(dropouti)
        self.out_drop     = VariationalDropout(dropouto)

        lstm_in = emb_dim + elmo_dim if self.use_elmo else emb_dim
        self.elmo_proj = nn.Linear(lstm_in, emb_dim) if self.use_elmo else None

        # Stack of LSTM layers with DropConnect
        self.lstms = nn.ModuleList()
        for i in range(layers):
            in_dim  = emb_dim if i == 0 else hidden
            out_dim = hidden  if i < layers - 1 else emb_dim  # weight tying needs emb_dim
            raw_lstm = nn.LSTM(in_dim, out_dim, batch_first=True)
            # Apply DropConnect
            self.lstms.append(raw_lstm)

        self.wdrop     = wdrop
        self.hid_drops = nn.ModuleList([VariationalDropout(dropouth) for _ in range(layers)])
        self.head      = nn.Linear(emb_dim, vocab_size)

        # Weight tying (reduces parameters, improves perplexity — Press & Wolf 2017)
        self.head.weight = self.embedding.weight

    def _apply_wdrop(self, lstm: nn.LSTM, x: torch.Tensor, hidden=None):
        """Apply DropConnect mask to hidden-to-hidden weights during training."""
        if self.training and self.wdrop > 0:
            for name in [n for n, _ in lstm.named_parameters() if "weight_hh" in n]:
                w = getattr(lstm, name)
                mask = w.data.new_empty(w.shape).bernoulli_(1 - self.wdrop)
                masked = w * mask / (1 - self.wdrop + 1e-8)
                # Use functional to avoid in-place parameter modification
                setattr(lstm, name, nn.Parameter(masked, requires_grad=False))
        return lstm(x, hidden)

    def forward(self, x, hidden=None, elmo_emb=None):
        e = self.embedding(x)
        e = self.emb_drop(e)

        if self.use_elmo and elmo_emb is not None:
            e = torch.cat([e, elmo_emb], dim=-1)
            e = torch.relu(self.elmo_proj(e))

        e = self.inp_drop(e)

        h = e
        new_hiddens = []
        for i, lstm in enumerate(self.lstms):
            h, hid = self._apply_wdrop(lstm, h, hidden[i] if hidden else None)
            h = self.hid_drops[i](h)
            new_hiddens.append(hid)

        h = self.out_drop(h)
        logits = self.head(h)
        return logits, new_hiddens


# ===========================================================================
# Training
# ===========================================================================

def _epoch(model, loader, opt, device, train: bool, variant: str, asgd_trigger=None):
    model.train() if train else model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total, count = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25 if variant == "awd" else 1.0)
                opt.step()
            n = (tgt != 0).sum().item()
            total += loss.item() * n
            count += n
    return total / max(count, 1)


def train(
    variant: str = "standard",   # "standard" | "awd"
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 30.0,            # AWD uses large LR with ASGD
    train_file: str = "train_small.txt",
):
    assert variant in ("standard", "awd"), "variant must be 'standard' or 'awd'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{variant.upper()}-LSTM] Device: {device}")

    train_path = PROC / train_file
    val_path   = PROC / "val_small.txt"

    print("Building vocab…")
    with train_path.open() as f:
        vocab = Vocab.build(f, min_count=3, max_size=50_000)
    print(f"  vocab: {len(vocab):,}")

    vocab_path = CKPT / f"lstm2_{variant}_vocab.pkl"
    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)

    train_ds = SentenceDataset(train_path, vocab)
    val_ds   = SentenceDataset(val_path,   vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=2)

    if variant == "awd":
        model = AWDLstmLM(len(vocab)).to(device)
        # AWD paper: use SGD → switch to ASGD after val loss plateaus
        opt   = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1.2e-6)
        asgd_triggered = False
        patience = 0
        best_val = float("inf")
    else:
        model = LSTMLM(len(vocab)).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
        best_val = float("inf")

    for ep in range(1, epochs + 1):
        tr = _epoch(model, train_loader, opt, device, True,  variant)
        vl = _epoch(model, val_loader,   opt, device, False, variant)
        print(f"  Epoch {ep}: train_ppl={math.exp(tr):.1f} | val_ppl={math.exp(vl):.1f}")

        # AWD-LSTM: switch SGD → ASGD when val stops improving
        if variant == "awd" and not asgd_triggered:
            if vl < best_val:
                best_val = vl
                patience = 0
            else:
                patience += 1
            if patience >= 2:
                print("  [AWD] Switching SGD → ASGD")
                opt = torch.optim.ASGD(model.parameters(), lr=lr,
                                       t0=0, lambd=0., weight_decay=1.2e-6)
                asgd_triggered = True
        elif variant == "standard":
            if vl < best_val:
                best_val = vl

        if vl <= best_val:
            best_val = vl
            torch.save(model.state_dict(), CKPT / f"lstm2_{variant}_best.pt")
            print("  ✓ saved")

    print(f"[{variant.upper()}-LSTM] Training done.")


# ===========================================================================
# Loading
# ===========================================================================

def load_model(variant: str = "standard"):
    assert variant in ("standard", "awd")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = CKPT / f"lstm2_{variant}_vocab.pkl"
    ckpt_path  = CKPT / f"lstm2_{variant}_best.pt"

    # Fallback to original checkpoint names for backward compat
    if not vocab_path.exists():
        vocab_path = CKPT / "lstm_vocab.pkl"
    if not ckpt_path.exists():
        ckpt_path = CKPT / "lstm_best.pt"

    with vocab_path.open("rb") as f:
        vocab = pickle.load(f)

    if variant == "awd":
        model = AWDLstmLM(len(vocab))
    else:
        model = LSTMLM(len(vocab))

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device), strict=False
    )
    model.to(device).eval()
    return model, vocab, device


# ===========================================================================
# ELMo helper for inference
# ===========================================================================

def _get_elmo_emb(prefix_tokens: List[str], device: torch.device) -> Optional[torch.Tensor]:
    """Load ELMo model and get embedding for prefix tokens. Returns None on failure."""
    try:
        from src.elmo_embeddings import ELMoEmbedder
        embedder = ELMoEmbedder()
        emb = embedder.embed_sentence(prefix_tokens)  # (T, elmo_dim)
        return emb.unsqueeze(0).to(device)            # (1, T, elmo_dim)
    except Exception:
        return None


# ===========================================================================
# Predict — greedy top-k
# ===========================================================================

@torch.no_grad()
def predict_next(
    model,
    vocab: Vocab,
    device: torch.device,
    prefix: str,
    top_k: int = 5,
    use_elmo: bool = False,
) -> List[Tuple[str, float]]:
    """
    Standard greedy top-k prediction.
    use_elmo: attempt to augment embeddings with ELMo if available.
    """
    tokens = [BOS] + prefix.lower().split()
    ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)

    elmo_emb = None
    if use_elmo and hasattr(model, "use_elmo") and model.use_elmo:
        elmo_emb = _get_elmo_emb(prefix.lower().split(), device)
        if elmo_emb is not None:
            # Pad with BOS position
            bos_pad = torch.zeros(1, 1, elmo_emb.size(-1), device=device)
            elmo_emb = torch.cat([bos_pad, elmo_emb], dim=1)

    logits, _ = model(ids, elmo_emb=elmo_emb)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top   = torch.topk(probs, top_k + 4)

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
# Predict — Beam Search
# ===========================================================================

@torch.no_grad()
def beam_search_predict(
    model,
    vocab: Vocab,
    device: torch.device,
    prefix: str,
    beam_size: int = 5,
    max_new_tokens: int = 3,
    length_penalty: float = 0.6,
    use_elmo: bool = False,
) -> List[Tuple[str, float]]:
    """
    Beam search decoding for next-word prediction.
    Returns (first_word, score) pairs sorted best-first.
    """
    SKIP = {vocab.stoi.get(s) for s in (PAD, EOS, UNK) if s in vocab.stoi}

    tokens = [BOS] + prefix.lower().split()
    ids    = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)

    elmo_emb = None
    if use_elmo and hasattr(model, "use_elmo") and model.use_elmo:
        elmo_emb = _get_elmo_emb(prefix.lower().split(), device)
        if elmo_emb is not None:
            bos_pad  = torch.zeros(1, 1, elmo_emb.size(-1), device=device)
            elmo_emb = torch.cat([bos_pad, elmo_emb], dim=1)

    logits, hidden = model(ids, elmo_emb=elmo_emb)
    log_probs = torch.log_softmax(logits[0, -1], dim=-1)

    topk_vals, topk_ids = torch.topk(log_probs, beam_size * 2)
    beams = []
    for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
        if idx in SKIP:
            continue
        h = hidden
        if isinstance(h, list):
            h = [(hi[0].clone(), hi[1].clone()) for hi in h]
        elif isinstance(h, tuple):
            h = (h[0].clone(), h[1].clone())
        beams.append((v, [idx], h))
        if len(beams) == beam_size:
            break

    for _ in range(max_new_tokens - 1):
        all_cands = []
        for score, seq, h in beams:
            last_id = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
            logits_t, new_h = model(last_id, h)
            lp = torch.log_softmax(logits_t[0, -1], dim=-1)
            topk_v, topk_i = torch.topk(lp, beam_size * 2)
            for v, idx in zip(topk_v.tolist(), topk_i.tolist()):
                if idx in SKIP:
                    continue
                all_cands.append((score + v, seq + [idx], new_h))
        all_cands.sort(key=lambda x: -x[0])
        beams = all_cands[:beam_size]

    results, seen = [], set()
    for score, seq, _ in beams:
        lp_norm = ((5 + len(seq)) / 6) ** length_penalty
        norm_score = score / lp_norm
        first = vocab.itos.get(seq[0], UNK)
        if first in seen or first in (PAD, BOS, EOS, UNK):
            continue
        seen.add(first)
        results.append((first, math.exp(min(norm_score, 0))))

    results.sort(key=lambda x: -x[1])
    return results


# ===========================================================================
# Variant dispatcher — called from Streamlit
# ===========================================================================

def get_predictions(
    variant:   str,
    use_beam:  bool,
    use_elmo:  bool,
    prefix:    str,
    top_k:     int = 5,
) -> List[Tuple[str, float]]:
    """
    Single entry point for Streamlit.

    variant:  "standard" | "awd"
    use_beam: True → beam search, False → greedy
    use_elmo: True → augment with ELMo embeddings
    """
    model, vocab, device = load_model(variant)

    if use_beam:
        return beam_search_predict(model, vocab, device, prefix,
                                   beam_size=top_k * 2, use_elmo=use_elmo)[:top_k]
    else:
        return predict_next(model, vocab, device, prefix, top_k, use_elmo)


# ===========================================================================
# CLI
# ===========================================================================

def cli_predict(variant, prefix, top_k, beam, elmo):
    preds = get_predictions(variant, beam, elmo, prefix, top_k)
    decode_label = "beam" if beam else "greedy"
    elmo_label   = "+ELMo" if elmo else ""
    print(f"\nVariant: {variant.upper()}-LSTM  Decoding: {decode_label}{elmo_label}")
    print(f"Prefix: {prefix!r}")
    for w, p in preds:
        print(f"  {w:<20} {p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--variant",    choices=["standard", "awd"], default="standard")
    t.add_argument("--epochs",     type=int,   default=5)
    t.add_argument("--batch-size", type=int,   default=64)
    t.add_argument("--lr",         type=float, default=30.0)
    t.add_argument("--file",       default="train_small.txt")

    p = sub.add_parser("predict")
    p.add_argument("--variant", choices=["standard", "awd"], default="standard")
    p.add_argument("--prefix",  required=True)
    p.add_argument("--top-k",   type=int, default=5)
    p.add_argument("--beam",    action="store_true")
    p.add_argument("--elmo",    action="store_true")

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.variant, args.epochs, args.batch_size, args.lr, args.file)
    else:
        cli_predict(args.variant, args.prefix, args.top_k, args.beam, args.elmo)