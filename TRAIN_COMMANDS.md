# Training Commands — NLP22 Autocomplete Project

## Checkpoint Map (tất cả file cần có để chạy đầy đủ)

```
checkpoints/
├── ngram_4.pkl                   ← ✓ đã có sẵn (~38 MB, custom KN)
├── hmm_lm.pkl                    ← ✓ đã có sẵn
├── maxent_model.pkl              ← ✓ đã có sẵn
├── rnn_best.pt                   ← cần train (GPU recommended)
├── rnn_vocab.pkl                 ← cần train
├── lstm_best.pt                  ← ✓ đã có sẵn
├── lstm_vocab.pkl                ← ✓ đã có sẵn
├── lstm2_awd_best.pt             ← cần train (GPU)
├── lstm2_awd_vocab.pkl           ← cần train (GPU)
├── elmo/
│   ├── elmo_best.pt              ← optional (GPU)
│   └── vocab.pkl                 ← optional (GPU)
├── gpt2-finetuned/               ← ✓ đã có sẵn (nguyên folder)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
├── mamba_best.pt                 ← cần train (GPU)
├── mamba_vocab.pkl               ← cần train (GPU)
├── bm25_reranker.pkl             ← cần build (CPU, nhanh)
├── rag_hybrid_index.pkl          ← cần build (CPU, nhanh)
└── moe_gate.pkl                  ← cần train (CPU, nhanh)
```

---

## Nhóm 1 — CPU (chạy trên máy local, không cần GPU)

Chạy từng lệnh theo thứ tự. Có thể mở nhiều terminal chạy song song.

### 1.1 N-gram KN-4 (đã có, skip nếu checkpoint tồn tại)
```bash
python -m src.ngram_model train
```
- Output: `checkpoints/ngram_4.pkl` (~38 MB)
- Thời gian: ~23 giây
- Data dùng: `data/processed/train_small.txt`
- Ghi chú: Custom KN (không dùng NLTK), min_count=2 mặc định

---

### 1.2 HMM Language Model
```bash
python -m src.hmm_model train
```
- Output: `checkpoints/hmm_lm.pkl`
- Thời gian: ~30–40 phút (do POS tagging từng câu)
- Data dùng: `data/processed/train_small.txt`
- Yêu cầu: spaCy hoặc NLTK tagger (tự động fallback)

---

### 1.3 MaxEnt Language Model
```bash
python -m src.maxent_model train --epochs 3 --top-words 5000 --lr 0.01
```
- Output: `checkpoints/maxent_model.pkl`
- Thời gian: ~20–30 phút
- Data dùng: `data/processed/train_small.txt`

---

### 1.4 BM25 Corpus Re-ranker (index only, không train neural)
```bash
python -m src.bm25_reranker build --max-windows 50000
```
- Output: `checkpoints/bm25_reranker.pkl`
- Thời gian: ~5 phút
- Data dùng: `data/processed/train_small.txt`

---

### 1.5 RAG Dense Index (BM25 + Dense hybrid)
```bash
python -m src.rag_dense build --mode hybrid --corpus data/processed/train_small.txt
```
- Output: `checkpoints/rag_hybrid_index.pkl`
- Thời gian: ~10 phút (không có sentence-transformers) / ~20 phút (có sentence-transformers)
- Data dùng: `data/processed/train_small.txt`
- Ghi chú: Nếu muốn dense retrieval tốt hơn, cài `sentence-transformers` trước

---

### 1.6 MoE Gating Network (train sau khi có đủ các model khác)
```bash
python -m src.moe_ensemble train_gate --val val_small.txt --epochs 5
```
- Output: `checkpoints/moe_gate.pkl`
- Thời gian: ~15 phút (phụ thuộc số model đã load được)
- Yêu cầu: ít nhất ngram + lstm + gpt2 checkpoints phải tồn tại
- Ghi chú: Nếu chưa đủ model, dùng mode `heuristic` (không cần file này)

---

## Nhóm 2 — GPU (chạy trên Colab T4 / Kaggle P100 / server GPU)

**Lưu ý Colab:** Thêm `!` trước mỗi lệnh. Ví dụ:
```python
!python -m src.rnn_model train --epochs 5
```

---

### 2.1 RNN Basic
```bash
python -m src.rnn_model train --epochs 5 --batch-size 64
```
- Output: `checkpoints/rnn_best.pt` + `checkpoints/rnn_vocab.pkl`
- Thời gian: ~20 phút (T4) / ~2–3 giờ (CPU)
- Ghi chú: Model đơn giản nhất, dùng để so sánh với LSTM

---

### 2.2 LSTM Standard (nếu muốn train lại với neural_model_2 format)
```bash
python -m src.neural_model_2 train --variant standard --epochs 5 --batch-size 64 --lr 0.001
```
- Output: `checkpoints/lstm2_standard_best.pt` + `checkpoints/lstm2_standard_vocab.pkl`
- Thời gian: ~25 phút (T4) / ~3–4 giờ (CPU)
- Ghi chú: neural_model_2.py tự fallback về `lstm_best.pt` nếu không có file này → **có thể skip nếu đã có checkpoint cũ**

---

### 2.3 AWD-LSTM
```bash
python -m src.neural_model_2 train --variant awd --epochs 5 --batch-size 64 --lr 30.0
```
- Output: `checkpoints/lstm2_awd_best.pt` + `checkpoints/lstm2_awd_vocab.pkl`
- Thời gian: ~30 phút (T4) / ~4–5 giờ (CPU)
- Ghi chú: Optimizer tự switch từ SGD → ASGD sau epoch 3 (khi val loss ngừng cải thiện)

---

### 2.4 ELMo Contextual Embeddings (optional)
```bash
python -m src.elmo_embeddings train --epochs 5 --batch-size 64 --hidden 256 --layers 2 --lr 0.001
```
- Output: `checkpoints/elmo/elmo_best.pt` + `checkpoints/elmo/vocab.pkl`
- Thời gian: ~40 phút (T4) / ~5–6 giờ (CPU)
- Ghi chú: Optional — nếu không train, ELMo toggle sẽ tự ẩn trong Streamlit

---

### 2.5 Mamba SSM
```bash
python -m src.mamba_model train --epochs 5 --batch-size 32 --d-model 256 --layers 4 --d-state 16 --lr 0.001
```
- Output: `checkpoints/mamba_best.pt` + `checkpoints/mamba_vocab.pkl`
- Thời gian: ~50 phút (T4) / ~8–10 giờ (CPU)
- Ghi chú: batch-size 32 để tránh OOM trên T4 (16GB); tăng lên 64 nếu có nhiều VRAM hơn

---

## Thứ tự chạy tối ưu

### Nếu có máy local CPU + Colab cùng lúc

**Local (chạy background):**
```bash
# Terminal 1
python -m src.hmm_model train &
python -m src.maxent_model train &

# Terminal 2 (sau khi terminal 1 xong ~30 phút)
python -m src.bm25_reranker build &
python -m src.rag_dense build --mode hybrid &
```

**Colab (3 notebook riêng biệt chạy song song):**
```python
# Notebook 1 — RNN + AWD-LSTM
!python -m src.rnn_model train --epochs 5
!python -m src.neural_model_2 train --variant awd --epochs 5

# Notebook 2 — ELMo (optional)
!python -m src.elmo_embeddings train --epochs 5

# Notebook 3 — Mamba
!python -m src.mamba_model train --epochs 5 --layers 4
```

**Sau khi tất cả xong (local):**
```bash
python -m src.moe_ensemble train_gate
python -m src.evaluate_all
```

---

### Nếu chỉ có CPU local (không có GPU, không có Colab)

```bash
# Bước 1: Model thống kê (chạy được, không quá chậm)
python -m src.ngram_model train
python -m src.hmm_model train
python -m src.maxent_model train

# Bước 2: Indexes (nhanh)
python -m src.bm25_reranker build
python -m src.rag_dense build --mode bm25   # chỉ dùng bm25, bỏ qua dense

# Bước 3: Neural (chỉ chạy nếu có đủ thời gian — qua đêm)
python -m src.rnn_model train --epochs 3
python -m src.neural_model_2 train --variant awd --epochs 3   # giảm epoch xuống 3
# ELMo và Mamba: bỏ qua nếu không có GPU — quá chậm trên CPU

# Bước 4: Ensemble gate
python -m src.moe_ensemble train_gate

# Bước 5: Evaluate
python -m src.evaluate_all
```

---

## Evaluate — chạy sau khi train xong

### Evaluate tất cả model có checkpoint
```bash
python -m src.evaluate_all
```
→ Tự detect model nào có checkpoint, skip model chưa train.
→ Lưu kết quả vào `logs/eval_results.json` → Streamlit tự load.

### Evaluate chỉ một số model
```bash
python -m src.evaluate_all --models ngram rnn hmm lstm maxent lstm_awd gpt2
```

### Cập nhật kết quả cho model vừa train xong
```bash
python -m src.evaluate_all --models rnn lstm_awd
```

---

## Kiểm tra nhanh từng model sau khi train

```bash
# N-gram KN-4
python -m src.ngram_model predict --prefix "the quick brown fox"

# HMM
python -m src.hmm_model predict --prefix "the quick brown fox"

# MaxEnt
python -m src.maxent_model predict --prefix "the quick brown fox"

# RNN Basic
python -m src.rnn_model predict --prefix "the quick brown fox"

# LSTM Standard (checkpoint cũ)
python -m src.neural_model predict --prefix "the quick brown fox"

# AWD-LSTM
python -m src.neural_model_2 predict --variant awd --prefix "the quick brown fox"

# ELMo (test embedding output)
python -m src.elmo_embeddings test --sentence "the bank is open near the river bank"

# GPT-2 fine-tuned
python -m src.finetune_gpt2 predict --prefix "the quick brown fox"

# Mamba
python -m src.mamba_model predict --prefix "the quick brown fox"

# RAG Hybrid
python -m src.rag_dense predict --prefix "the experimental results show" --mode hybrid

# MoE Ensemble
python -m src.moe_ensemble predict --prefix "the quick brown fox" --mode heuristic

# LAMBADA benchmark
python -m src.eval_lambada
```

---

## Chạy Streamlit (sau khi có checkpoint)

```bash
streamlit run demo/app_2.py
```

Mở trình duyệt tại: `http://localhost:8501`

- **Page 1** — Autocomplete Lab: chọn model, options, gõ câu → xem gợi ý
- **Page 2** — Score Comparison: charts PPL / Top-k / MRR / Latency (load từ `logs/eval_results.json`)
- **Page 3** — Architecture Guide: giải thích model + Colab notebook template