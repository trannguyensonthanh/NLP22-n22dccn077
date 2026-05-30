"""
Streamlit demo: type a sentence, see next-word suggestions from each model.

Run:   streamlit run demo/app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="English Autocomplete",
    page_icon="✍️",
    layout="wide",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .main .block-container {padding-top: 2rem; max-width: 1200px;}
      .suggestion-btn button {
          width: 100%; text-align: left; font-family: monospace;
      }
      .model-card {
          background: #f7f7fa; border-radius: 12px; padding: 16px 18px;
          border: 1px solid #e6e6ee; margin-bottom: 8px;
      }
      .prob-bar {
          background: #eef; border-radius: 6px; height: 8px; overflow: hidden;
          margin: 2px 0 8px 0;
      }
      .prob-bar > div {
          background: linear-gradient(90deg,#6366f1,#a855f7);
          height: 100%;
      }
      .word-chip {
          display: inline-block; padding: 2px 10px; margin: 2px;
          background: #eef2ff; color: #3730a3; border-radius: 999px;
          font-family: monospace; font-size: 0.9em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✍️ English Writing Autocomplete")
st.caption(
    "Trained on WikiText-103 + BBC News + arXiv abstracts — high-quality, "
    "well-edited English. Compare three language models side by side."
)

# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading n-gram model…")
def load_ngram(order=4):
    from src import ngram_model
    ngram_model._ensure_nltk()
    return ngram_model, ngram_model.load_model(order)


@st.cache_resource(show_spinner="Loading LSTM model…")
def load_lstm():
    from src import neural_model
    return (neural_model,) + neural_model.load_model()


@st.cache_resource(show_spinner="Loading GPT-2 model…")
def load_gpt2():
    from src import finetune_gpt2
    return (finetune_gpt2,) + finetune_gpt2.load_model()


# ---------------------------------------------------------------- state
if "prefix" not in st.session_state:
    st.session_state.prefix = "The quick brown fox"

def append_word(word: str):
    text = st.session_state.prefix.rstrip()
    st.session_state.prefix = f"{text} {word}".lstrip()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-k suggestions", 1, 10, 5)
    st.markdown("**Models**")
    use_ngram = st.checkbox("N-gram (Kneser-Ney 4)", value=True)
    use_lstm = st.checkbox("LSTM (from scratch)", value=True)
    use_gpt2 = st.checkbox("GPT-2 fine-tuned", value=True)

    st.divider()
    st.markdown("**Quick examples**")
    examples = [
        "The quick brown fox",
        "Artificial intelligence is",
        "According to the latest research",
        "The president announced that",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.prefix = ex
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.prefix = ""
        st.rerun()


# ---------------------------------------------------------------- input
st.text_area(
    "Start typing a sentence:",
    key="prefix",
    height=120,
    placeholder="Type something in English…",
)

prefix = st.session_state.prefix.strip()

# token preview
if prefix:
    chips = "".join(
        f"<span class='word-chip'>{w}</span>" for w in prefix.split()
    )
    st.markdown(f"**Tokens ({len(prefix.split())}):** {chips}",
                unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------- predict
def render_predictions(title: str, preds, key_prefix: str):
    """Render a model's predictions: list of (word, prob)."""
    st.markdown(f"#### {title}")
    if not preds:
        st.info("No predictions.")
        return
    max_p = max(p for _, p in preds) or 1.0
    for i, (w, p) in enumerate(preds):
        pct = int((p / max_p) * 100)
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button(f"➕ {w}", key=f"{key_prefix}_{i}_{w}",
                         use_container_width=True):
                append_word(w)
                st.rerun()
        with c2:
            st.markdown(
                f"<div class='prob-bar'><div style='width:{pct}%'></div></div>"
                f"<small>p = {p:.4f}</small>",
                unsafe_allow_html=True,
            )

active = [m for m, on in
          [("ngram", use_ngram), ("lstm", use_lstm), ("gpt2", use_gpt2)] if on]

if not prefix:
    st.info("👈 Type a sentence above or pick an example from the sidebar.")
elif not active:
    st.warning("Select at least one model in the sidebar.")
else:
    cols = st.columns(len(active))
    results = {}  # name -> list[(w, p)]

    for col, name in zip(cols, active):
        with col:
            with st.container(border=True):
                try:
                    if name == "ngram":
                        mod, model = load_ngram()
                        from nltk.tokenize import word_tokenize
                        preds = mod.predict_next(
                            model, word_tokenize(prefix.lower()), top_k)
                        render_predictions("📊 N-gram (KN-4)", preds, "ng")
                    elif name == "lstm":
                        mod, model, vocab, device = load_lstm()
                        preds = mod.predict_next(model, vocab, device,
                                                 prefix, top_k)
                        render_predictions("🧠 LSTM", preds, "ls")
                    elif name == "gpt2":
                        mod, model, tokenizer, device = load_gpt2()
                        preds = mod.predict_next(model, tokenizer, device,
                                                 prefix + " ", top_k)
                        render_predictions("🤖 GPT-2 fine-tuned", preds, "gp")
                    results[name] = preds
                except FileNotFoundError as e:
                    st.error(f"Model not trained yet: {e}")
                except Exception as e:
                    st.error(str(e))

    # comparison summary
    if len(results) > 1:
        st.divider()
        st.markdown("### 🆚 Top-1 comparison")
        cmp_cols = st.columns(len(results))
        for col, (name, preds) in zip(cmp_cols, results.items()):
            with col:
                label = {"ngram": "N-gram", "lstm": "LSTM",
                         "gpt2": "GPT-2"}[name]
                if preds:
                    w, p = preds[0]
                    st.metric(label, w, f"p = {p:.3f}")
                else:
                    st.metric(label, "—")

st.divider()
st.caption(
    "💡 Tip: click ➕ next to any suggestion to append it and continue writing. "
    "Models must be trained first (see README.md)."
)
