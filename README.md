# English Writing Autocomplete — NLP22 Project

> **Next-word prediction system** trained on high-quality, well-edited English  
> (WikiText-103 · BBC News · arXiv abstracts)  
> Compares **11 model architectures** across 6 metric dimensions with an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architectures](#2-model-architectures)
3. [Smart Filters & Enhancements](#3-smart-filters--enhancements)
4. [Project Structure](#4-project-structure)
5. [Quick Start](#5-quick-start)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Guide](#7-training-guide)
8. [Colab / Kaggle Setup](#8-colab--kaggle-setup)
9. [Evaluation](#9-evaluation)
10. [Streamlit Demo](#10-streamlit-demo)
11. [Results](#11-results)
12. [Technical Reference](#12-technical-reference)
13. [Lecture Connections](#13-lecture-connections)

---

## 1. Project Overview

This project implements and compares **11 different language model architectures** for English next-word autocomplete, ranging from classical statistical methods (n-gram, HMM) to modern deep learning (Transformer, Mamba SSM) and retrieval-augmented generation.

### Goals

- Demonstrate the evolution of language modeling: n-gram → neural → Transformer → SSM
- Provide fair quantitative comparison across **Perplexity, Top-k Accuracy, F1, MRR, and Latency**
- Build an interactive demo where each model can be run with or without enhancements
- Show practical trade-offs: accuracy vs. speed, model size vs. domain fit

### Data Sources

| Dataset | Size | Domain | Why chosen |
|---------|------|--------|------------|
| WikiText-103 | ~536 MB | Wikipedia Featured/Good articles | High-quality encyclopedic prose |
| BBC News | ~2.8 MB | Journalism | Edited formal English, news register |
| arXiv abstracts | ~57 MB | Academic papers | Dense, grammatical academic writing |

All three sources are **professionally edited** — deliberately avoiding Common Crawl / Reddit noise, which matters for a writing-assist tool.

---

## 2. Model Architectures

### Tier 1 — Classical Statistical

#### 2.1 4-gram Kneser-Ney (KN-4)
**File:** `src/ngram_model.py`

The classical baseline. Uses **Kneser-Ney smoothing** — the gold standard for n-gram smoothing — which estimates probabilities of unseen n-grams using continuation counts rather than raw frequency.

- Context window: 3 previous words
- Smoothing: Kneser-Ney interpolated (NLTK `KneserNeyInterpolated`)
- Fast-path: only scores words seen in matching context; backs off to shorter context before unigram fallback
- **No GPU needed**; trains in under 5 minutes

```bash
python -m src.ngram_model train --order 4
python -m src.ngram_model predict --prefix "the quick brown"
```

#### 2.2 N-gram Interpolated (1–4 gram)
**File:** `src/ngram_model_2.py`

Trains **four separate KN models** (orders 1, 2, 3, 4) and combines them with **learned linear interpolation weights** λ:

```
P_interp(w | ctx) = λ₁·P₁(w) + λ₂·P₂(w|w₋₁) + λ₃·P₃(w|w₋₂,w₋₁) + λ₄·P₄(w|w₋₃,w₋₂,w₋₁)
```

Weights are tuned on the validation set using coordinate ascent to maximise held-out likelihood. Usually reduces PPL by 15–20% vs. KN-4 alone.

```bash
python -m src.ngram_model_2 train_interpolated --order 4
python -m src.ngram_model_2 predict_interp --prefix "the quick brown"
```

### Tier 2 — Generative / Discriminative Statistical

#### 2.3 HMM Language Model
**File:** `src/hmm_model.py`

A **Hidden Markov Model** where:
- **Hidden states** = POS tags (Penn Treebank tagset, 37 tags + BOS/EOS)
- **Observations** = words
- **Transition matrix A**: P(tag_t | tag_{t-1}) — bigram POS transitions
- **Emission matrix B**: P(word | tag) — word likelihood given POS

Training counts transitions and emissions from tagged corpus (using spaCy or NLTK tagger), then applies **add-k smoothing** (k=0.01). Prediction via Viterbi decoding followed by marginalisation over next POS tags.

```bash
python -m src.hmm_model train
python -m src.hmm_model predict --prefix "the quick brown"
python -m src.hmm_model evaluate --limit 200
```

#### 2.4 Maximum Entropy Language Model
**File:** `src/maxent_model.py`

A **log-linear (MaxEnt) language model** trained with online SGD + L2 regularisation. Features include:
- Bigram, trigram, and skip-bigram indicators
- POS class of previous word (heuristic: verb-ing, det, prep, etc.)
- Word class of candidate (digit, capitalized, short, etc.)
- Punctuation context and sentence length class

Also serves as a **plug-in enhancer** for N-gram and HMM outputs via `MaxEntEnhancer.enhance()`.

```bash
python -m src.maxent_model train --epochs 3
python -m src.maxent_model predict --prefix "the quick brown"
```

### Tier 3 — Neural RNN

#### 2.5 LSTM Language Model (Standard)
**File:** `src/neural_model.py` (original) + `src/neural_model_2.py` (extended)

Two-layer word-level LSTM trained from scratch with cross-entropy loss:

```
Embedding → Dropout → 2-layer LSTM → Dropout → Linear(vocab)
```

- Vocab: 50,000 most frequent words (min count 3), `<unk>` for OOV
- Hidden dim: 512, Embedding dim: 256, Dropout: 0.3
- Trained with AdamW, gradient clipping at 1.0
- Saves best checkpoint by validation loss

```bash
python -m src.neural_model train --epochs 5
python -m src.neural_model predict --prefix "the quick brown"
```

#### 2.6 LSTM + Beam Search
**File:** `src/neural_model_2.py` (`beam_search_predict`)

Same LSTM checkpoint, but decodes with **beam search** (beam size 5) instead of greedy top-k. Applies **length normalisation** (GNMT formula: `score / ((5+len)^0.6 / 6^0.6)`) to avoid preferring shorter continuations.

Accessible in Streamlit via "Beam Search decoding" toggle on any LSTM variant. No retraining needed.

#### 2.7 AWD-LSTM
**File:** `src/neural_model_2.py` (`AWDLstmLM`)

**ASGD Weight-Dropped LSTM** (Merity et al. 2018). Three additional regularisation techniques over standard LSTM:

1. **DropConnect**: drops hidden-to-hidden weight matrices (not activations) at rate 0.5
2. **Variational (locked) dropout**: same dropout mask applied at every timestep — prevents mask averaging
3. **Weight tying**: output projection shares weights with embedding matrix — reduces parameters significantly
4. **Optimizer**: SGD with large LR (30.0) → switches to **ASGD** after validation loss plateaus (typically epoch 3–4)

Expected PPL improvement: ~15% over standard LSTM.

```bash
python -m src.neural_model_2 train --variant awd --epochs 5
python -m src.neural_model_2 predict --variant awd --prefix "the quick brown"
```

#### 2.8 ELMo Contextual Embeddings
**File:** `src/elmo_embeddings.py`

**Embeddings from Language Models** (Peters et al. 2018). A BiLSTM trained as a language model in both directions:

```
Forward  LSTM: predict next token (left → right)
Backward LSTM: predict prev token (right → left)
```

ELMo embedding = γ · Σ_k (s_k · h_k), where s_k are learned scalar weights per layer, γ is a learned scalar. This gives **context-sensitive embeddings**: the word "bank" has different vectors in "river bank" vs "bank account".

Used as **optional input layer** for LSTM Standard and AWD-LSTM — enable via "ELMo contextual embeddings" toggle in Streamlit.

```bash
python -m src.elmo_embeddings train --epochs 5
python -m src.elmo_embeddings test --sentence "the bank is open"
```

### Tier 4 — Transformer (Pretrained)

#### 2.9 GPT-2 Fine-tuned
**File:** `src/finetune_gpt2.py`

GPT-2 small (117M params) pretrained on 40GB web text, fine-tuned on our filtered corpus (WikiText + BBC + arXiv). Fine-tuning adapts the distribution toward well-edited formal English.

- Architecture: 12 Transformer layers, 12 attention heads, d_model=768
- Training: 2 epochs, AdamW lr=5e-5, fp16 if GPU available, batch=8, block_size=128
- BPE tokenizer: no OOV by design
- Best model on every metric; used as backbone for RAG

**This checkpoint is already trained. Do not retrain unless you have 4+ hours of GPU time.**

```bash
python -m src.finetune_gpt2 predict --prefix "the quick brown"
```

### Tier 5 — State Space Model (2023)

#### 2.10 Mamba SSM
**File:** `src/mamba_model.py`

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao 2023).

Unlike Transformers (O(n²) attention) or LSTMs (sequential), Mamba uses a **Selective State Space Model**:

```
h_t = A_t · h_{t-1} + B_t · x_t     (state update)
y_t = C_t · h_t                       (output)
```

The key innovation: **A_t, B_t, C_t are functions of the input** (not fixed matrices), allowing the model to selectively remember or forget based on content. Training uses a convolutional view; inference uses the recurrent view (O(1) per token).

Our implementation uses pure PyTorch (no custom CUDA kernel) with a sequential scan — sufficient for the corpus size.

Architecture: Embedding → 4× MambaBlock (conv → SSM → SiLU gate) → LayerNorm → Linear. Weight tying between embedding and output.

```bash
python -m src.mamba_model train --epochs 5 --layers 4 --d-model 256
python -m src.mamba_model predict --prefix "the quick brown"
```

### Tier 6 — Retrieval-Augmented Generation

#### 2.11 RAG (BM25 / Dense / Hybrid)
**File:** `src/rag_dense.py`

RAG does **not** add a new neural architecture — it augments GPT-2 with retrieved context at inference time:

```
prefix → retrieve top-2 passages → prepend as context → GPT-2 predicts next word
```

Three retrieval strategies:

| Mode | Method | Speed | Semantic quality |
|------|--------|-------|-----------------|
| BM25 | Keyword TF-IDF matching (Okapi BM25) | Fastest | Good for exact terms |
| Dense | Average word embedding cosine similarity | Medium | Better for synonyms |
| Hybrid | RRF fusion of BM25 + Dense ranks | Medium | Best overall |

**No neural training required** — only index building (5–10 min, CPU).

```bash
python -m src.rag_dense build --mode hybrid --corpus data/processed/train_small.txt
python -m src.rag_dense predict --prefix "the experimental results show" --mode hybrid
```

### Tier 7 — Ensemble Methods

#### 2.12 Reciprocal Rank Fusion (RRF)
**File:** `src/moe_ensemble.py` (`reciprocal_rank_fusion`)

Combines ranked lists from multiple models without any training:

```
RRF_score(w) = Σ_i  1 / (60 + rank_i(w))
```

k=60 is the standard smoothing constant. Merges top-k lists from any subset of models. Typically outperforms any individual model in the ensemble.

#### 2.13 Mixture of Experts (MoE)
**File:** `src/moe_ensemble.py` (`MoEEnsemble`)

A small **gating network** (2-layer MLP, 12 features → 32 → n_experts) learns to weight models based on prefix features:

- Prefix length bucket
- Academic word ratio
- News word ratio
- Last word POS class
- Unique word diversity
- … (12 total features)

Two modes: `heuristic` (rule-based weights, no training) and `learned` (trained on validation set via reward signal: 1.0 if expert got it right, 0 otherwise).

```bash
python -m src.moe_ensemble train_gate --val val_small.txt --epochs 5
python -m src.moe_ensemble predict --prefix "the quick brown" --mode heuristic
```

---

## 3. Smart Filters & Enhancements

These are **optional post-processing layers** applied after a model generates its top-k candidates. Each is only shown in Streamlit for models where it makes sense.

### 3.1 MaxEnt Enhancement
**File:** `src/maxent_model.py` (`MaxEntEnhancer`)
**Compatible with:** N-gram KN-4, N-gram Interpolated, HMM-LM

Re-ranks the base model's output using MaxEnt scores. Interpolates:
```
final_score = α · maxent_prob + (1-α) · base_prob,   α = 0.4
```

### 3.2 Beam Search
**Compatible with:** LSTM Standard, AWD-LSTM

Replaces greedy top-k decoding with beam search (beam size 5, max 3 new tokens). Uses length normalisation to prevent short-sequence bias.

### 3.3 ELMo Input Layer
**Compatible with:** LSTM Standard, AWD-LSTM

Replaces static word embeddings with ELMo contextual embeddings. Requires `elmo_embeddings train` to run first.

### 3.4 BM25 Corpus Re-ranking
**Compatible with:** GPT-2 fine-tuned

Re-ranks GPT-2's output by how often each candidate word appears in contexts similar to the current prefix (BM25 score against a sliding-window corpus index).

### 3.5 Domain / Register Filter
**Compatible with:** All models except Ensemble

Re-ranks suggestions to match a target writing style using Naïve Bayes register detection:

| Setting | Boosts | Penalises |
|---------|--------|-----------|
| Formal / Academic | methodology, furthermore, empirical… | gonna, kinda, yeah… |
| Informal / Casual | really, just, like, cool… | aforementioned, pursuant… |
| News / Journalistic | announced, according, reported… | casual contractions |
| None (default) | — auto-detect from prefix — | — |

### 3.6 POS Grammar Filter
**File:** `src/pos_filter.py`
**Compatible with:** All models

Uses spaCy to tag the prefix, then boosts/penalises candidates based on POS transition probabilities (e.g., after `TO` → strongly prefer `VB`; after `DT` → prefer `NN/JJ`, penalise `DT`). Reduces grammatically implausible suggestions.

---

## 4. Project Structure

```
NLP22-n22dccn077/
│
├── data/
│   ├── raw/                        # Downloaded raw corpora
│   │   ├── wikitext.txt            # 536 MB — WikiText-103
│   │   ├── bbc.txt                 # 2.8 MB — BBC News
│   │   └── arxiv.txt               # 57 MB  — arXiv abstracts
│   └── processed/                  # Cleaned, split data
│       ├── train.txt               # 402 MB — full training set
│       ├── train_small.txt         # 27 MB  — subset for fast iteration
│       ├── val.txt                 # 22 MB  — validation set
│       ├── val_small.txt           # 2.75 MB
│       ├── test.txt                # 22 MB  — held-out test set
│       └── test_small.txt          # 2.76 MB
│
├── checkpoints/                    # Saved model weights
│   ├── ngram_4.pkl                 # ✓ KN-4 model (already trained)
│   ├── ngram_interp_4.pkl          # N-gram Interpolated
│   ├── lstm_best.pt                # ✓ LSTM Standard (already trained)
│   ├── lstm_vocab.pkl              # ✓ LSTM vocab (already trained)
│   ├── lstm2_awd_best.pt           # AWD-LSTM
│   ├── lstm2_awd_vocab.pkl         # AWD-LSTM vocab
│   ├── elmo/
│   │   ├── elmo_best.pt            # ELMo BiLSTM
│   │   └── vocab.pkl               # ELMo vocab
│   ├── gpt2-finetuned/             # ✓ GPT-2 fine-tuned (already trained)
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer files...
│   ├── mamba_best.pt               # Mamba SSM
│   ├── mamba_vocab.pkl             # Mamba vocab
│   ├── hmm_lm.pkl                  # HMM Language Model
│   ├── maxent_model.pkl            # MaxEnt LM
│   ├── bm25_reranker.pkl           # BM25 corpus re-ranker index
│   ├── rag_hybrid_index.pkl        # RAG dense+BM25 index
│   └── moe_gate.pkl                # MoE gating network
│
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── download_data.py            # Download WikiText, BBC, arXiv
│   ├── preprocess.py               # Clean, filter, split corpus
│   │
│   ├── ngram_model.py              # KN-4 n-gram (original)
│   ├── ngram_model_2.py            # Interpolated n-gram (new)
│   ├── hmm_model.py                # HMM Language Model (new)
│   ├── maxent_model.py             # MaxEnt LM + Enhancer (new)
│   ├── neural_model.py             # LSTM Standard (original)
│   ├── neural_model_2.py           # LSTM + AWD-LSTM + Beam + ELMo (new)
│   ├── elmo_embeddings.py          # ELMo BiLSTM embeddings (new)
│   ├── finetune_gpt2.py            # GPT-2 fine-tuning (original)
│   ├── mamba_model.py              # Mamba SSM (new)
│   ├── rag_model.py                # RAG BM25 basic (original)
│   ├── rag_dense.py                # RAG BM25+Dense+Hybrid (new)
│   ├── moe_ensemble.py             # MoE + RRF ensemble (new)
│   │
│   ├── pos_filter.py               # POS grammar re-ranker (new)
│   ├── sentiment_filter.py         # Domain/register filter (new)
│   ├── bm25_reranker.py            # BM25 corpus re-ranker (new)
│   │
│   ├── evaluate.py                 # Original evaluation (ngram/lstm/gpt2)
│   ├── evaluate_all.py             # Full evaluation — all 11 models (new)
│   └── eval_lambada.py             # LAMBADA benchmark
│
├── demo/
│   └── app.py                      # Streamlit interactive demo (updated)
│
├── notebooks/
│   └── analysis.ipynb              # Quantitative + qualitative analysis
│
├── logs/
│   ├── evaluation.log              # Raw evaluate output
│   ├── eval_results.json           # Structured results → Streamlit charts
│   ├── lstm_training.log
│   ├── gpt2_training.log
│   └── lambada_eval.log
│
├── requirements.txt
└── README.md
```

---

## 5. Quick Start

### Prerequisites

```bash
# Python 3.9+
pip install -r requirements.txt

# Optional but recommended for POS filter
pip install spacy
python -m spacy download en_core_web_sm

# Optional for interactive charts in Streamlit
pip install plotly
```

### requirements.txt

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
nltk>=3.8
tqdm>=4.65.0
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
plotly>=5.18.0
```

### Run the demo immediately (with existing checkpoints)

```bash
# Uses the 3 pre-trained models: KN-4, LSTM, GPT-2
streamlit run demo/app.py
```

---

## 6. Data Pipeline

### Step 1 — Download

```bash
python -m src.download_data
```

Downloads WikiText-103, BBC News, and arXiv abstracts to `data/raw/`. Skips files that already exist.

### Step 2 — Preprocess

```bash
# Standard (recommended)
python -m src.preprocess

# With slow grammar filter (removes ungrammatical sentences)
python -m src.preprocess --grammar-check
```

**Pipeline:**
1. Load raw text → sentence split
2. Quality filters: length (6–50 words), non-ASCII ratio (<30%), repetition check
3. Lowercase + URL removal + whitespace normalisation
4. 90/5/5 train/val/test split (shuffled, seed=42)
5. Save to `data/processed/`

**Why these sources?** Most large corpora (Common Crawl, Reddit) contain noisy, ungrammatical text. For a writing-assist tool, we deliberately use professionally edited sources so that suggestions reflect good written English.

---

## 7. Training Guide

### What's already trained (do not retrain)

| Checkpoint | Model | Notes |
|------------|-------|-------|
| `checkpoints/ngram_4.pkl` | KN-4 n-gram | PPL 1088 |
| `checkpoints/lstm_best.pt` | LSTM Standard | PPL 118 |
| `checkpoints/lstm_vocab.pkl` | LSTM vocab | — |
| `checkpoints/gpt2-finetuned/` | GPT-2 fine-tuned | PPL 42.7 |

### What needs to be trained (new models)

#### Group A — CPU, run locally (fast)

```bash
# 1. N-gram Interpolated (~15 min)
python -m src.ngram_model_2 train_interpolated --order 4

# 2. HMM Language Model (~30 min)
python -m src.hmm_model train

# 3. MaxEnt LM (~20 min)
python -m src.maxent_model train --epochs 3

# 4. RAG index — build only, no neural training (~10 min)
python -m src.rag_dense build --mode hybrid

# 5. BM25 re-ranker index (~5 min)
python -m src.bm25_reranker build

# 6. MoE gating network (~15 min)
python -m src.moe_ensemble train_gate
```

#### Group B — GPU recommended (Colab T4 free tier)

```bash
# AWD-LSTM  (~30 min on T4 | ~4h on CPU)
python -m src.neural_model_2 train --variant awd --epochs 5

# ELMo BiLSTM  (~40 min on T4 | ~5h on CPU)
python -m src.elmo_embeddings train --epochs 5

# Mamba SSM  (~50 min on T4 | ~8h on CPU)
python -m src.mamba_model train --epochs 5 --layers 4 --d-model 256
```

### Full training sequence (local server with GPU)

```bash
# 1. Data (if not done)
python -m src.download_data
python -m src.preprocess

# 2. Statistical models (CPU, run in background)
python -m src.ngram_model train --order 4
python -m src.ngram_model_2 train_interpolated
python -m src.hmm_model train
python -m src.maxent_model train

# 3. Neural models (GPU)
python -m src.neural_model train --epochs 5
python -m src.neural_model_2 train --variant awd --epochs 5
python -m src.elmo_embeddings train --epochs 5
python -m src.mamba_model train --epochs 5

# 4. GPT-2 fine-tune (already done — skip unless retraining)
# python -m src.finetune_gpt2 train --epochs 2

# 5. Indexes and ensemble (CPU, fast)
python -m src.rag_dense build --mode hybrid
python -m src.bm25_reranker build
python -m src.moe_ensemble train_gate

# 6. Evaluate everything
python -m src.evaluate_all
```

---

## 8. Colab / Kaggle Setup

### Using Google Colab (free T4 GPU)

**Step 1 — Upload project to Google Drive**

Upload the folder `NLP22-n22dccn077/` to your Google Drive. You only need:
```
NLP22-n22dccn077/
├── src/                        ← all source files
├── checkpoints/                ← existing checkpoints
└── data/processed/
    ├── train_small.txt         ← 27 MB (required)
    └── val_small.txt           ← 2.75 MB (required)
```

You do NOT need `train.txt` (402 MB) — all new models use `train_small.txt`.

**Step 2 — Colab notebook template**

```python
# Cell 1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Navigate and install
import os
os.chdir('/content/drive/MyDrive/NLP22-n22dccn077')

!pip install -q torch transformers datasets nltk tqdm sentence-transformers

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

```python
# Cell 3 — Train AWD-LSTM (~30 min on T4)
!python -m src.neural_model_2 train --variant awd --epochs 5
```

```python
# Cell 4 — Train ELMo (~40 min on T4)
!python -m src.elmo_embeddings train --epochs 5
```

```python
# Cell 5 — Train Mamba (~50 min on T4)
!python -m src.mamba_model train --epochs 5 --layers 4 --d-model 256
```

```python
# Cell 6 — Build indexes (CPU, fast)
!python -m src.rag_dense build --mode hybrid
!python -m src.bm25_reranker build
!python -m src.moe_ensemble train_gate
```

```python
# Cell 7 — Evaluate all trained models
!python -m src.evaluate_all --limit 300
```

**Tips for Colab free tier:**
- Sessions reset after ~12h or when the tab is closed
- Checkpoints are saved to Google Drive automatically (they're in `checkpoints/` which is inside your Drive folder)
- You can run Cell 3, 4, 5 in **separate notebooks simultaneously** to save time
- Use `Runtime → Change runtime type → T4 GPU` before running

### Using Kaggle (P100 GPU, 30h/week free)

```python
# In a Kaggle notebook cell:
import os
os.makedirs('/kaggle/working/NLP22-n22dccn077', exist_ok=True)

# Upload your src/ and data/processed/ as a Kaggle dataset
# Then reference them:
os.chdir('/kaggle/input/nlp22-project')

!python -m src.neural_model_2 train --variant awd --epochs 5
```

---

## 9. Evaluation

### Running evaluation

```bash
# Evaluate all available models (uses test.txt, limit 300 sentences)
python -m src.evaluate_all

# Evaluate specific models only
python -m src.evaluate_all --models ngram lstm gpt2 mamba

# Quick smoke-test (50 sentences)
python -m src.evaluate_all --limit 50

# Show current results without re-running
python -m src.evaluate_all --show-only

# Update existing results (skip already-evaluated models)
python -m src.evaluate_all --update --models lstm_awd
```

Results are saved to `logs/eval_results.json` and automatically loaded by Streamlit Page 2.

### Metrics explained

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Perplexity (PPL)** | exp(−(1/N)·Σ log p(wᵢ\|ctx)) | How "surprised" the model is. Lower = better. Range: [1, ∞) |
| **Top-1 Accuracy** | hits@1 / total | Did the model's #1 guess match the true word? |
| **Top-3 Accuracy** | hits@3 / total | Was the true word in the model's top 3? |
| **Top-5 Accuracy** | hits@5 / total | Was the true word in the model's top 5? Closest to real autocomplete UX |
| **F1@k** | 2·P·R/(P+R) | Equals Top-k accuracy when \|true labels\|=1 per position |
| **MRR** | (1/N)·Σ 1/rank | Mean Reciprocal Rank — rewards getting closer to rank 1 |
| **Latency (ms)** | avg inference time | Critical for real-time autocomplete (<50ms is good) |

### LAMBADA benchmark

```bash
# Evaluate on LAMBADA (long-range discourse prediction)
python -m src.eval_lambada
```

LAMBADA tests whether a model can predict the final word of a passage using **whole-passage context**, not just local n-gram context. The target is almost always a proper noun or named entity.

| Model | Top-1 | Top-5 | Notes |
|-------|-------|-------|-------|
| LSTM | 0.000 | 0.000 | Expected: no discourse tracking |
| GPT-2 fine-tuned | 0.144 | 0.246 | Only model with long-range context |

---

## 10. Streamlit Demo

### Start the demo

```bash
streamlit run demo/app.py
```

Opens at `http://localhost:8501`

### Page 1 — Autocomplete Lab

**Panel configuration** — up to 3 model panels side-by-side. Each panel independently configures:

**Model selection:**
- N-gram KN-4 / Interpolated / HMM-LM / LSTM Standard / AWD-LSTM / GPT-2 / Mamba / RAG / Ensemble

**Options shown per model (only compatible options appear):**

| Model | MaxEnt | Beam | ELMo | BM25 rerank | Domain filter |
|-------|:------:|:----:|:----:|:-----------:|:-------------:|
| N-gram KN-4 | ✓ | ✗ | ✗ | ✗ | ✓ |
| N-gram Interp | ✓ | ✗ | ✗ | ✗ | ✓ |
| HMM-LM | ✓ | ✗ | ✗ | ✗ | ✓ |
| LSTM Standard | ✗ | ✓ | ✓ | ✗ | ✓ |
| AWD-LSTM | ✗ | ✓ | ✓ | ✗ | ✓ |
| GPT-2 | ✗ | ✗ | ✗ | ✓ | ✓ |
| Mamba | ✗ | ✗ | ✗ | ✗ | ✓ |
| RAG | ✗ | ✗ | ✗ | ✗ | ✓ + retrieval mode |
| Ensemble | ✗ | ✗ | ✗ | ✗ | ✗ + strategy + experts |

**Domain filter options:**
- `None` — model's natural distribution
- `Formal / Academic` — boosts: methodology, furthermore, demonstrate…
- `Informal / Casual` — boosts: gonna, really, just, like…
- `News / Journalistic` — boosts: announced, according, reported…

**Clicking a suggestion** appends the word to the input automatically.

### Page 2 — Score Comparison Dashboard

- **Perplexity chart** (log scale bar chart)
- **Top-k Accuracy chart** (grouped bar: Top-1, Top-3, Top-5)
- **MRR chart**
- **Latency chart** (ms per query)
- **Full table** with all metrics
- **Radar chart** (multi-metric comparison for 2–5 selected models)
- **Manual update form** — enter results after training a new model

All charts powered by Plotly. Data loaded from `logs/eval_results.json`.

### Page 3 — Architecture Guide

- Architecture table with PPL, Top-5, latency, and training status
- Colab setup notebook template (copy-paste ready)
- Full training command reference

---

## 11. Results

Results from the original three trained models (ground truth):

| Model | PPL | Top-1 | Top-3 | Top-5 | MRR |
|-------|----:|------:|------:|------:|----:|
| 4-gram KN-4 | 1088.10 | n/a | n/a | n/a | n/a |
| LSTM Standard | 118.35 | 0.242 | 0.370 | 0.433 | 0.185 |
| GPT-2 fine-tuned | **42.74** | **0.352** | **0.501** | **0.562** | **0.284** |

Expected results after training new models:

| Model | PPL (est.) | Top-5 (est.) | Train time (T4) |
|-------|--------:|----------:|--------------|
| N-gram Interpolated | ~900 | n/a | ~15 min CPU |
| HMM-LM | ~600 | ~0.15 | ~30 min CPU |
| MaxEnt LM | — | ~0.20 | ~20 min CPU |
| AWD-LSTM | ~100 | ~0.45 | ~30 min T4 |
| LSTM + ELMo | ~110 | ~0.44 | (reuses ckpts) |
| Mamba SSM | ~55 | ~0.48 | ~50 min T4 |
| RAG (Hybrid) | — | ~0.57 | ~10 min CPU |
| RRF Ensemble | — | best | no training |

### LAMBADA results

| Model | Top-1 | Top-5 |
|-------|------:|------:|
| LSTM | 0.000 | 0.000 |
| GPT-2 fine-tuned | 0.144 | 0.246 |

### Top-k coverage curve

Diminishing returns after k=5: displaying 5 suggestions captures almost all practical benefit; k=10 adds <3% accuracy while making the UI cluttered.

---

## 12. Technical Reference

### CLI reference — all commands

```bash
# ── Data ───────────────────────────────────────────────────────────────────
python -m src.download_data
python -m src.preprocess [--grammar-check] [--seed 42]

# ── N-gram ─────────────────────────────────────────────────────────────────
python -m src.ngram_model train [--order 4] [--file train_small.txt]
python -m src.ngram_model predict --prefix "TEXT" [--order 4] [--top-k 5]

python -m src.ngram_model_2 train_interpolated [--order 4] [--no-tune]
python -m src.ngram_model_2 predict_interp --prefix "TEXT"

# ── HMM ────────────────────────────────────────────────────────────────────
python -m src.hmm_model train [--add-k 0.01]
python -m src.hmm_model predict --prefix "TEXT"
python -m src.hmm_model evaluate [--limit 200]

# ── MaxEnt ─────────────────────────────────────────────────────────────────
python -m src.maxent_model train [--epochs 3] [--top-words 5000] [--lr 0.01]
python -m src.maxent_model predict --prefix "TEXT"

# ── LSTM ───────────────────────────────────────────────────────────────────
python -m src.neural_model train [--epochs 5] [--batch-size 64] [--lr 1e-3]
python -m src.neural_model predict --prefix "TEXT"

python -m src.neural_model_2 train --variant standard [--epochs 5]
python -m src.neural_model_2 train --variant awd [--epochs 5]
python -m src.neural_model_2 predict --variant standard --prefix "TEXT"
python -m src.neural_model_2 predict --variant awd --prefix "TEXT" [--beam] [--elmo]
python -m src.neural_model_2 beam --prefix "TEXT" [--beam-size 5]

# ── ELMo ───────────────────────────────────────────────────────────────────
python -m src.elmo_embeddings train [--epochs 5] [--hidden 256] [--layers 2]
python -m src.elmo_embeddings test --sentence "TEXT"

# ── GPT-2 ──────────────────────────────────────────────────────────────────
python -m src.finetune_gpt2 train [--epochs 2] [--batch-size 8]
python -m src.finetune_gpt2 predict --prefix "TEXT"

# ── Mamba ──────────────────────────────────────────────────────────────────
python -m src.mamba_model train [--epochs 5] [--layers 4] [--d-model 256]
python -m src.mamba_model predict --prefix "TEXT"

# ── RAG ────────────────────────────────────────────────────────────────────
python -m src.rag_dense build [--mode hybrid] [--corpus data/processed/train_small.txt]
python -m src.rag_dense predict --prefix "TEXT" [--mode hybrid]

python -m src.bm25_reranker build [--max-windows 50000]
python -m src.bm25_reranker test --prefix "TEXT"

# ── Ensemble ───────────────────────────────────────────────────────────────
python -m src.moe_ensemble train_gate [--val val_small.txt] [--epochs 5]
python -m src.moe_ensemble predict --prefix "TEXT" [--mode heuristic]

# ── Evaluation ─────────────────────────────────────────────────────────────
python -m src.evaluate_all [--models ngram lstm gpt2] [--limit 300]
python -m src.evaluate_all --show-only
python -m src.evaluate_all --update --models lstm_awd mamba
python -m src.eval_lambada
```

### Checkpoint compatibility

The new `neural_model_2.py` falls back automatically to `lstm_best.pt` and `lstm_vocab.pkl` if the new checkpoint files don't exist yet — backward compatible with the original LSTM training.

### Adding a new model

1. Create `src/your_model.py` with `train()`, `load_model()`, `predict_next(model, ..., prefix, top_k)` functions
2. Add a loader `_load_yourmodel()` in `demo/app.py`
3. Add `"your_model"` to `TIER_OPTIONS` and `TIER_CAPABILITIES` dicts
4. Add the `elif tier == "your_model"` branch in `get_predictions()`
5. Add an evaluator in `src/evaluate_all.py` and register it in `EVALUATORS`

---

## 13. Lecture Connections

This project directly implements concepts from the NLP course lectures:

| Lecture Chapter | Concept | Implementation |
|----------------|---------|---------------|
| Ch.2 — HMM POS Tagging | HMM architecture, Viterbi decoding | `src/hmm_model.py` |
| Ch.2 — Beam Search | Beam search decoding | `src/neural_model_2.py` (`beam_search_predict`) |
| Ch.3 — N-gram LMs | MLE, smoothing, perplexity | `src/ngram_model.py` |
| Ch.3 — KN Smoothing | Kneser-Ney interpolated | `src/ngram_model.py` (NLTK `KneserNeyInterpolated`) |
| Ch.3 — Interpolation | Linear λ-weighted mixture | `src/ngram_model_2.py` (`InterpolatedNgramModel`) |
| Ch.3 — Perplexity | Evaluation metric | `src/evaluate_all.py` |
| Ch.4 — Naïve Bayes | Sentiment/register classifier | `src/sentiment_filter.py` (`NaiveBayesSentiment`) |
| Ch.4 — MaxEnt / Log-linear | MaxEnt LM + enhancer | `src/maxent_model.py` |
| Ch.4 — Precision/Recall/F1 | Evaluation metrics | `src/evaluate_all.py` |
| Ch.7 — Information Extraction | Feature extraction for MaxEnt | `src/maxent_model.py` (`FeatureExtractor`) |
| Ch.8 — IR / BM25 | BM25 retrieval | `src/rag_dense.py`, `src/bm25_reranker.py` |
| Ch.8 — RAG | Retrieval-augmented generation | `src/rag_dense.py` |
| Ch.8 — MRR | Evaluation metric | `src/evaluate_all.py` |

### Beyond the lecture (additional architectures)

| Model | Paper | Year | Key idea |
|-------|-------|------|---------|
| AWD-LSTM | Merity et al. | 2018 | DropConnect + ASGD + weight tying |
| ELMo | Peters et al. | 2018 | Contextual BiLSTM embeddings |
| GPT-2 | Radford et al. | 2019 | Large-scale Transformer pretrain |
| Mamba | Gu & Dao | 2023 | Selective State Space Model |
| RRF | Cormack et al. | 2009 | Reciprocal Rank Fusion ensemble |
| MoE | Shazeer et al. | 2017 | Learned gating over expert models |

---

*Project: NLP22 — n22dccn077*  
*Data: WikiText-103 + BBC News + arXiv abstracts*  
*Framework: PyTorch + HuggingFace Transformers + Streamlit*