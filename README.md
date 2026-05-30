# English Writing Autocomplete

A next-word prediction system trained on **high-quality, well-edited English**
(WikiText-103, BBC News, arXiv abstracts) so that suggestions reflect good
written style — useful as a writing-practice aid.

Three models are trained and compared:

| Model            | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| 4-gram (Kneser-Ney) | Classical statistical baseline                       |
| LSTM             | Word-level neural LM trained from scratch               |
| GPT-2 fine-tuned | Strongest model: pretrained Transformer adapted to corpus |

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. data
python -m src.download_data
python -m src.preprocess              # add --grammar-check for slow filter

# 3. train
python -m src.ngram_model train --order 4
python -m src.neural_model train --epochs 5
python -m src.finetune_gpt2 train --epochs 2

# 4. evaluate
python -m src.evaluate

# 5. demo
streamlit run demo/app.py
```

## Quick predictions from the CLI

```bash
python -m src.ngram_model    predict --prefix "the quick brown"
python -m src.neural_model   predict --prefix "the quick brown"
python -m src.finetune_gpt2  predict --prefix "the quick brown"
```

## Project layout

```
autocomplete/
├── data/
│   ├── raw/         # downloaded corpora
│   └── processed/   # cleaned train/val/test
├── src/
│   ├── download_data.py    # WikiText + BBC + arXiv
│   ├── preprocess.py       # clean, filter, split
│   ├── ngram_model.py      # KN 4-gram baseline
│   ├── neural_model.py     # LSTM LM
│   ├── finetune_gpt2.py    # GPT-2 fine-tuning
│   ├── evaluate.py         # perplexity + top-k accuracy
│   └── eval_lambada.py     # LAMBADA next-word benchmark
├── notebooks/analysis.ipynb  # comparison + figures (run with outputs)
├── demo/app.py             # Streamlit interactive demo
├── checkpoints/            # saved model weights
└── requirements.txt
```

## Evaluation metrics

- **Perplexity** on the held-out test set — lower is better.
- **Top-k accuracy** (k = 1, 3, 5) — fraction of next-word predictions where
  the true word appears in the model's top-k. This metric is directly aligned
  with the autocomplete user experience.

## Results

Evaluated on `data/processed/test.txt` (see `logs/evaluation.log`):

| Model               |     PPL | Top-1 | Top-3 | Top-5 |
| ------------------- | ------: | ----: | ----: | ----: |
| 4-gram (Kneser-Ney) | 1088.10 |   n/a |   n/a |   n/a |
| LSTM                |  118.35 | 0.242 | 0.370 | 0.433 |
| GPT-2 fine-tuned    |   42.74 | 0.352 | 0.501 | 0.562 |

GPT-2 wins on every metric, as expected — pretrained Transformer + curated
fine-tuning corpus is far stronger than from-scratch LSTM or a smoothed
n-gram. Top-k accuracy is not reported for the n-gram because computing it
with NLTK's pure-Python `KneserNeyInterpolated.score()` over a 227k-word
vocabulary is prohibitively slow; the perplexity number alone already shows
the baseline gap.

### LAMBADA benchmark

[LAMBADA](https://arxiv.org/abs/1606.06031) is the standard next-word
benchmark: predict the **last word** of a passage where the target cannot be
guessed from local context — you must track the whole discourse. We evaluate
on the first 500 examples of `lambada_openai` (`python -m src.eval_lambada`,
see `logs/lambada_eval.log`):

| Model            | Top-1 | Top-5 |
| ---------------- | ----: | ----: |
| LSTM             | 0.000 | 0.000 |
| GPT-2 fine-tuned | 0.144 | 0.246 |

GPT-2 is the only model that captures any long-range discourse. The LSTM
scores ~0 — expected: it is trained on formal Wikipedia/news/academic prose,
while LAMBADA passages are narrative fiction with mostly proper-noun targets
(character names) that are out of its domain and vocabulary. The n-gram is
omitted here for the same reason as above — pure-Python KN scoring over the
227k-word vocabulary is far too slow for a 500-example sweep.

## Data quality rationale

Most large LM corpora (Common Crawl, Reddit, etc.) contain a lot of noisy or
ungrammatical text. For a writing-assistance tool we deliberately pick
sources that are professionally edited:

- **WikiText-103** — Wikipedia Featured / Good articles only.
- **BBC News**     — edited British journalism, formal register.
- **arXiv abstracts** — academic prose, dense and grammatical.

`preprocess.py` further filters for sentence length, character composition,
and (optionally, via `--grammar-check`) grammatical correctness using
`language_tool_python`.
