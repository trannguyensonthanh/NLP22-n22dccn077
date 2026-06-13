"""
demo/app.py — NLP Autocomplete Lab  (Final Version)

Quy tắc hiển thị options:
    Mỗi model chỉ hiện đúng những tuỳ chọn phù hợp với nó.

    N-gram KN-4:
        ✓ MaxEnt enhancement (re-rank)
        ✓ Domain filter (formal/informal)
        ✗ Beam search (không áp dụng)
        ✗ ELMo (không áp dụng)
        ✗ BM25 reranker (không đủ mạnh để rerank)

    HMM-LM:
        ✓ MaxEnt enhancement
        ✓ Domain filter
        ✗ các thứ khác

    RNN Basic:
        ✓ Domain filter
        ✗ các thứ khác

    MaxEnt (standalone):
        ✓ Domain filter
        ✗ các thứ khác

    LSTM Standard:
        ✓ Greedy / Beam Search (chọn 1)
        ✓ ELMo toggle (nếu đã train ELMo)
        ✓ Domain filter
        ✗ MaxEnt (không compatible)
        ✗ BM25 reranker

    AWD-LSTM:
        ✓ Greedy / Beam Search (chọn 1)
        ✓ ELMo toggle (nếu đã train ELMo)
        ✓ Domain filter
        ✗ MaxEnt

    GPT-2 fine-tuned:
        ✓ BM25 reranker (re-rank GPT-2 output)
        ✓ Domain filter
        ✗ MaxEnt (không compatible)
        ✗ Beam / ELMo

    Mamba SSM:
        ✓ Domain filter
        ✗ tất cả enhancement khác (model đứng độc lập)

    RAG (BM25 / Dense / Hybrid):
        ✓ Chọn retrieval mode (bm25/dense/hybrid)
        ✓ Domain filter
        ✗ MaxEnt, Beam, ELMo

    Ensemble (RRF / MoE):
        ✓ Chọn strategy + experts
        ✗ Tất cả enhancement khác

Run:  streamlit run demo/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="NLP Autocomplete Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
  .prob-bar {
    background: rgba(128,128,128,.12); border-radius: 4px;
    height: 6px; overflow: hidden; margin: 3px 0 6px;
  }
  .prob-bar > div {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #5C6BC0, #26C6DA);
  }
  .word-chip {
    display: inline-block; padding: 2px 8px; margin: 2px;
    background: rgba(92,107,192,.1); border-radius: 999px;
    font-family: 'JetBrains Mono', monospace; font-size: .82em;
  }
  .opt-section {
    background: rgba(128,128,128,.05);
    border-radius: 8px; padding: 10px 12px; margin: 8px 0;
    border: 0.5px solid rgba(128,128,128,.15);
  }
  .opt-label {
    font-size: 11px; font-weight: 500; color: rgba(128,128,128,.8);
    text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px;
  }
  .filter-tag {
    display: inline-block; font-size: 10px; padding: 1px 7px;
    border-radius: 999px; margin: 2px; font-weight: 500;
  }
  .stExpander { border: 0.5px solid rgba(128,128,128,.2) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================ Constants

# Tier display name → internal key
TIER_OPTIONS = {
    "N-gram KN-4":          "ngram",
    "HMM-LM":               "hmm",
    "RNN Basic":            "rnn",
    "MaxEnt (standalone)":  "maxent_standalone",
    "LSTM Standard":        "lstm_std",
    "AWD-LSTM":             "lstm_awd",
    "GPT-2 fine-tuned":     "gpt2",
    "Mamba SSM":            "mamba",
    "RAG":                  "rag",
    "Ensemble":             "ensemble",
}

# Which options are valid for each tier
# Keys: maxent, beam, elmo, bm25_rerank, domain_filter, rag_mode, ensemble_opts
TIER_CAPABILITIES = {
    "ngram":              {"maxent": True,  "beam": False, "elmo": False, "bm25_rerank": False, "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "hmm":                {"maxent": True,  "beam": False, "elmo": False, "bm25_rerank": False, "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "rnn":                {"maxent": False, "beam": False, "elmo": False, "bm25_rerank": False, "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "maxent_standalone":  {"maxent": False, "beam": False, "elmo": False, "bm25_rerank": False, "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "lstm_std":           {"maxent": False, "beam": True,  "elmo": True,  "bm25_rerank": False, "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "lstm_awd":           {"maxent": False, "beam": True,  "elmo": True,  "bm25_rerank": False, "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "gpt2":               {"maxent": False, "beam": False, "elmo": False, "bm25_rerank": True,  "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "mamba":              {"maxent": False, "beam": False, "elmo": False, "bm25_rerank": False, "domain_filter": True,  "rag_mode": False, "ensemble_opts": False},
    "rag":                {"maxent": False, "beam": False, "elmo": False, "bm25_rerank": False, "domain_filter": True,  "rag_mode": True,  "ensemble_opts": False},
    "ensemble":           {"maxent": False, "beam": False, "elmo": False, "bm25_rerank": False, "domain_filter": False, "rag_mode": False, "ensemble_opts": True},
}

DOMAIN_LABELS = {
    "none":     "No domain filter",
    "formal":   "Formal / Academic",
    "informal": "Informal / Casual",
    "news":     "News / Journalistic",
}

TIER_COLORS = {
    "ngram":              "#5C6BC0",
    "hmm":                "#F57C00",
    "rnn":                "#EF5350",
    "maxent_standalone":  "#26A69A",
    "lstm_std":           "#8E24AA",
    "lstm_awd":           "#AB47BC",
    "gpt2":               "#00838F",
    "mamba":              "#2E7D32",
    "rag":                "#1565C0",
    "ensemble":           "#37474F",
}


# ============================================================ State

def _init():
    for k, v in [
        ("prefix", "The quick brown fox"),
        ("page", "autocomplete"),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

_init()


def _append(w):
    st.session_state.prefix = st.session_state.prefix.rstrip() + " " + w


# ============================================================ Cached loaders

@st.cache_resource(show_spinner="Loading KN-4 n-gram…")
def _load_ngram():
    from src import ngram_model
    ngram_model._ensure_nltk()
    return ngram_model, ngram_model.load_model(4)

@st.cache_resource(show_spinner="Loading RNN Basic…")
def _load_rnn():
    from src import rnn_model
    mod, vocab, device = rnn_model.load_model()
    return rnn_model, mod, vocab, device

@st.cache_resource(show_spinner="Loading MaxEnt standalone…")
def _load_maxent_standalone():
    from src import maxent_model
    return maxent_model, maxent_model.load_model()

@st.cache_resource(show_spinner="Loading HMM…")
def _load_hmm():
    from src import hmm_model
    return hmm_model, hmm_model.load_model()

@st.cache_resource(show_spinner="Loading MaxEnt enhancer…")
def _load_maxent():
    from src.maxent_model import MaxEntEnhancer
    enh = MaxEntEnhancer()
    enh.load()
    return enh

@st.cache_resource(show_spinner="Loading LSTM standard…")
def _load_lstm_std():
    try:
        from src import neural_model_2
        mod, vocab, device = neural_model_2.load_model("standard")
        return neural_model_2, mod, vocab, device
    except Exception:
        from src import neural_model
        mod, vocab, device = neural_model.load_model()
        return neural_model, mod, vocab, device

@st.cache_resource(show_spinner="Loading AWD-LSTM…")
def _load_lstm_awd():
    from src import neural_model_2
    mod, vocab, device = neural_model_2.load_model("awd")
    return neural_model_2, mod, vocab, device

@st.cache_resource(show_spinner="Loading GPT-2…")
def _load_gpt2():
    from src import finetune_gpt2
    return finetune_gpt2, *finetune_gpt2.load_model()

@st.cache_resource(show_spinner="Loading Mamba…")
def _load_mamba():
    from src import mamba_model
    return mamba_model, *mamba_model.load_model()

@st.cache_resource(show_spinner="Loading RAG index…")
def _load_rag():
    from src.rag_dense import RAGDenseModel
    return RAGDenseModel.load()

@st.cache_resource(show_spinner="Loading BM25 reranker…")
def _load_bm25_rerank():
    try:
        from src.bm25_reranker import BM25Reranker
        return BM25Reranker.load()
    except Exception:
        return None

@st.cache_resource(show_spinner="Loading domain filter…")
def _load_domain_filter():
    from src.sentiment_filter import SentimentFilter
    return SentimentFilter()


# ============================================================ Domain filter helper

def _apply_domain_filter(prefix: str, preds, domain: str):
    """
    Apply domain/register filter to predictions.
    domain: "none" | "formal" | "informal" | "news"
    """
    if domain == "none" or not preds:
        return preds
    try:
        sf = _load_domain_filter()
        # Force the register regardless of auto-detection
        if domain == "formal":
            sf.nb.log_prior = {c: -999.0 for c in sf.nb.CLASSES}
            sf.nb.log_prior["formal"] = 0.0
        elif domain == "informal":
            sf.nb.log_prior = {c: -999.0 for c in sf.nb.CLASSES}
            sf.nb.log_prior["informal"] = 0.0
        elif domain == "news":
            sf.nb.log_prior = {c: -999.0 for c in sf.nb.CLASSES}
            sf.nb.log_prior["formal"] = 0.0  # news ≈ formal
        # Auto-detect
        return sf.rerank(prefix, preds)
    except Exception:
        return preds


# ============================================================ Prediction dispatcher

def get_predictions(cfg: dict, prefix: str, top_k: int):
    """
    cfg fields:
        model_tier   : internal tier key
        use_maxent   : bool (ngram/hmm only)
        use_beam     : bool (lstm only)
        use_elmo     : bool (lstm only)
        use_bm25r    : bool (gpt2 only)
        domain       : "none"|"formal"|"informal"|"news" (most models)
        rag_mode     : "bm25"|"dense"|"hybrid"
        ens_strategy : "rrf"|"moe_heuristic"|"moe_learned"
        ens_experts  : list[str]
    Returns (preds, error_str | None)
    """
    tier  = cfg.get("model_tier", "gpt2")
    raw   = []

    try:
        # ---- N-gram KN-4 ----
        if tier == "ngram":
            from nltk.tokenize import word_tokenize
            mod, model = _load_ngram()
            raw = mod.predict_next(model, word_tokenize(prefix.lower()), top_k * 4)

        # ---- HMM ----
        elif tier == "hmm":
            mod, model = _load_hmm()
            raw = mod.predict_next(model, prefix.lower().split(), top_k * 4)

        # ---- RNN Basic ----
        elif tier == "rnn":
            mod, model, vocab, device = _load_rnn()
            raw = mod.predict_next(model, vocab, device, prefix, top_k * 4)

        # ---- MaxEnt standalone ----
        elif tier == "maxent_standalone":
            mod, model = _load_maxent_standalone()
            raw = model.predict_next(prefix.lower().split(), top_k * 4)

        # ---- LSTM Standard ----
        elif tier == "lstm_std":
            nm, model, vocab, device = _load_lstm_std()
            if cfg.get("use_beam"):
                if hasattr(nm, "beam_search_predict"):
                    raw = nm.beam_search_predict(model, vocab, device, prefix,
                                                  beam_size=top_k * 2,
                                                  use_elmo=cfg.get("use_elmo", False))
                else:
                    raw = nm.predict_next(model, vocab, device, prefix, top_k * 4)
            else:
                if hasattr(nm, "predict_next") and "use_elmo" in nm.predict_next.__code__.co_varnames:
                    raw = nm.predict_next(model, vocab, device, prefix, top_k * 4,
                                          use_elmo=cfg.get("use_elmo", False))
                else:
                    raw = nm.predict_next(model, vocab, device, prefix, top_k * 4)

        # ---- AWD-LSTM ----
        elif tier == "lstm_awd":
            nm, model, vocab, device = _load_lstm_awd()
            if cfg.get("use_beam"):
                raw = nm.beam_search_predict(model, vocab, device, prefix,
                                              beam_size=top_k * 2,
                                              use_elmo=cfg.get("use_elmo", False))
            else:
                raw = nm.predict_next(model, vocab, device, prefix, top_k * 4,
                                      use_elmo=cfg.get("use_elmo", False))

        # ---- GPT-2 ----
        elif tier == "gpt2":
            mod, model, tok, device = _load_gpt2()
            raw = mod.predict_next(model, tok, device, prefix + " ", top_k * 4)

        # ---- Mamba ----
        elif tier == "mamba":
            mod, model, vocab, device = _load_mamba()
            raw = mod.predict_next(model, vocab, device, prefix, top_k * 4)

        # ---- RAG ----
        elif tier == "rag":
            rag = _load_rag()
            raw = rag.predict_next(prefix, top_k * 4)

        # ---- Ensemble ----
        elif tier == "ensemble":
            strategy = cfg.get("ens_strategy", "rrf")
            experts  = cfg.get("ens_experts", ["ngram", "lstm_std", "gpt2"])

            # Map UI expert names to internal predict functions
            def _expert_preds(exp_name, pfx, k):
                sub = dict(cfg)
                sub["model_tier"] = exp_name
                sub["use_beam"]   = False
                sub["use_elmo"]   = False
                p, _ = get_predictions(sub, pfx, k)
                return p or []

            if strategy == "rrf":
                from src.moe_ensemble import reciprocal_rank_fusion
                lists = []
                for exp in experts:
                    p = _expert_preds(exp, prefix, top_k)
                    if p:
                        lists.append(p)
                raw = reciprocal_rank_fusion(lists)[:top_k * 2] if lists else []
            else:
                mode_str = "learned" if strategy == "moe_learned" else "heuristic"
                from src.moe_ensemble import MoEEnsemble
                # Map UI names to internal keys
                internal_exp = [
                    e.replace("lstm_std", "lstm").replace("lstm_awd", "lstm_awd")
                    for e in experts
                ]
                moe = MoEEnsemble(active_experts=internal_exp, mode=mode_str)
                moe.load_gate()
                raw = moe.predict(prefix, top_k * 4)

    except FileNotFoundError as e:
        return None, f"Model not trained yet: {e}"
    except Exception as e:
        return None, f"Error: {e}"

    # ---- MaxEnt enhancement (ngram / hmm only) ----
    caps = TIER_CAPABILITIES.get(tier, {})
    if caps.get("maxent") and cfg.get("use_maxent") and raw:
        try:
            enh = _load_maxent()
            if enh.is_loaded():
                raw = enh.enhance(prefix, raw)
        except Exception:
            pass

    # ---- BM25 reranker (gpt2 only) ----
    if caps.get("bm25_rerank") and cfg.get("use_bm25r") and raw:
        bm25r = _load_bm25_rerank()
        if bm25r:
            try:
                raw = bm25r.rerank(prefix, raw)
            except Exception:
                pass

    # ---- Domain / register filter ----
    domain = cfg.get("domain", "none")
    if caps.get("domain_filter") and domain != "none" and raw:
        raw = _apply_domain_filter(prefix, raw, domain)

    return raw[:top_k], None


# ============================================================ Sidebar

with st.sidebar:
    st.markdown("### 🧪 NLP Autocomplete Lab")
    page = st.radio(
        "Page", ["✍️ Autocomplete", "📊 Score Comparison", "📖 Architecture Guide"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Quick examples**")
    examples = {
        "General":   ["The quick brown fox", "She opened the door and"],
        "Academic":  ["In this paper we propose", "The experimental results show"],
        "News":      ["The government announced that", "According to the latest"],
        "Informal":  ["I wanna go to", "It's really gonna be"],
    }
    for grp, items in examples.items():
        st.markdown(f"<span style='font-size:11px;opacity:.6'>{grp}</span>", unsafe_allow_html=True)
        for ex in items:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.prefix = ex
                st.rerun()
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.prefix = ""
        st.rerun()


# ============================================================ PAGE 1: AUTOCOMPLETE

if "Autocomplete" in page:
    st.markdown("## ✍️ Autocomplete Lab")
    st.text_area("Type your sentence:", key="prefix", height=90,
                 placeholder="Start typing…")
    prefix = st.session_state.prefix.strip()

    if prefix:
        chips = "".join(f"<span class='word-chip'>{w}</span>" for w in prefix.split())
        st.markdown(f"**Tokens:** {chips}", unsafe_allow_html=True)

    st.divider()

    # ---- Panel setup ----
    top_k   = st.slider("Top-k suggestions", 1, 10, 5)
    n_panels = st.number_input("Số model so sánh cùng lúc", 1, 3, 2)
    st.divider()

    panel_cols = st.columns(n_panels)
    configs    = []

    for idx, col in enumerate(panel_cols):
        with col:
            color = "#5C6BC0"
            st.markdown(f"**Panel {idx+1}**")

            # ---- Tier select ----
            tier_label = st.selectbox(
                "Model", list(TIER_OPTIONS.keys()), key=f"tier_{idx}",
                help="Chọn kiến trúc model"
            )
            tier = TIER_OPTIONS[tier_label]
            caps = TIER_CAPABILITIES[tier]
            cfg  = {"model_tier": tier}

            # ---- Options — only show what's valid for this tier ----

            # 1. MaxEnt (ngram / hmm only)
            if caps["maxent"]:
                st.markdown('<div class="opt-section"><div class="opt-label">Enhancement</div>', unsafe_allow_html=True)
                cfg["use_maxent"] = st.checkbox(
                    "➕ MaxEnt re-ranking",
                    key=f"me_{idx}",
                    help="Re-rank output bằng Maximum Entropy model (log-linear features). Chỉ dùng cho N-gram và HMM."
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # 2. LSTM options (beam + elmo)
            if caps["beam"] or caps["elmo"]:
                st.markdown('<div class="opt-section"><div class="opt-label">LSTM Options</div>', unsafe_allow_html=True)
                if caps["beam"]:
                    cfg["use_beam"] = st.checkbox(
                        "🔭 Beam Search decoding",
                        key=f"beam_{idx}",
                        help="Beam size=5. Thay greedy top-k bằng beam search → đa dạng hơn. Chỉ dùng cho LSTM."
                    )
                if caps["elmo"]:
                    # Only show ELMo toggle if checkpoint exists
                    elmo_ckpt = ROOT / "checkpoints" / "elmo" / "elmo_best.pt"
                    if elmo_ckpt.exists():
                        cfg["use_elmo"] = st.checkbox(
                            "🧠 ELMo contextual embeddings",
                            key=f"elmo_{idx}",
                            help="Dùng ELMo BiLSTM làm input embedding thay vì static embedding."
                        )
                    else:
                        st.caption("⚠️ ELMo chưa train — bỏ qua. Chạy: `python -m src.elmo_embeddings train`")
                        cfg["use_elmo"] = False
                st.markdown('</div>', unsafe_allow_html=True)

            # 3. BM25 reranker (gpt2 only)
            if caps["bm25_rerank"]:
                st.markdown('<div class="opt-section"><div class="opt-label">Enhancement</div>', unsafe_allow_html=True)
                cfg["use_bm25r"] = st.checkbox(
                    "🔍 BM25 corpus re-ranking",
                    key=f"bm25r_{idx}",
                    help="Re-rank GPT-2 output dựa trên BM25 score với corpus training. Cần build bm25_reranker trước."
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # 4. RAG mode selector
            if caps["rag_mode"]:
                st.markdown('<div class="opt-section"><div class="opt-label">Retrieval Mode</div>', unsafe_allow_html=True)
                cfg["rag_mode"] = st.radio(
                    "Retrieval strategy",
                    ["bm25", "dense", "hybrid"],
                    format_func=lambda x: {
                        "bm25":   "BM25 (keyword)",
                        "dense":  "Dense (semantic)",
                        "hybrid": "Hybrid BM25+Dense",
                    }[x],
                    key=f"rag_{idx}",
                    help="BM25=nhanh, Dense=ngữ nghĩa, Hybrid=tốt nhất (RRF kết hợp)"
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # 5. Ensemble options
            if caps["ensemble_opts"]:
                st.markdown('<div class="opt-section"><div class="opt-label">Ensemble Strategy</div>', unsafe_allow_html=True)
                cfg["ens_strategy"] = st.radio(
                    "Strategy",
                    ["rrf", "moe_heuristic", "moe_learned"],
                    format_func=lambda x: {
                        "rrf":          "RRF (no training needed)",
                        "moe_heuristic":"MoE heuristic gate",
                        "moe_learned":  "MoE learned gate",
                    }[x],
                    key=f"ens_strat_{idx}",
                )
                all_experts = ["ngram", "hmm", "rnn", "maxent_standalone", "lstm_std", "lstm_awd", "gpt2", "mamba"]
                cfg["ens_experts"] = st.multiselect(
                    "Expert models",
                    all_experts,
                    default=["ngram", "lstm_std", "gpt2"],
                    key=f"ens_exp_{idx}",
                    format_func=lambda x: {
                        "ngram": "N-gram KN-4",
                        "hmm": "HMM-LM",
                        "rnn": "RNN Basic",
                        "maxent_standalone": "MaxEnt",
                        "lstm_std": "LSTM Standard",
                        "lstm_awd": "AWD-LSTM",
                        "gpt2": "GPT-2",
                        "mamba": "Mamba",
                    }.get(x, x)
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # 6. Domain filter (most models — NOT ensemble since it combines)
            if caps["domain_filter"]:
                st.markdown('<div class="opt-section"><div class="opt-label">Domain / Register Filter</div>', unsafe_allow_html=True)
                cfg["domain"] = st.selectbox(
                    "Writing style target",
                    list(DOMAIN_LABELS.keys()),
                    format_func=lambda x: DOMAIN_LABELS[x],
                    key=f"dom_{idx}",
                    help=(
                        "None = tự nhiên theo model.\n"
                        "Formal/Academic = ưu tiên từ học thuật (methodology, furthermore…)\n"
                        "Informal = ưu tiên từ bình dân (gonna, really, just…)\n"
                        "News = ưu tiên từ báo chí (announced, according, reported…)"
                    )
                )
                st.markdown('</div>', unsafe_allow_html=True)

            configs.append(cfg)

    st.divider()

    # ---- Run predictions ----
    if not prefix:
        st.info("👈 Type something above or pick an example from the sidebar.")
    else:
        st.markdown("### Predictions")
        result_cols = st.columns(n_panels)
        all_results = {}

        for idx, (col, cfg) in enumerate(zip(result_cols, configs)):
            tier = cfg["model_tier"]
            color = TIER_COLORS.get(tier, "#5C6BC0")

            # Build display title
            tier_label = [k for k, v in TIER_OPTIONS.items() if v == tier][0]
            tags = []
            if cfg.get("use_maxent"):       tags.append("MaxEnt")
            if cfg.get("use_beam"):         tags.append("Beam")
            if cfg.get("use_elmo"):         tags.append("ELMo")
            if cfg.get("use_bm25r"):        tags.append("BM25")
            if cfg.get("domain", "none") != "none":
                tags.append(DOMAIN_LABELS[cfg["domain"]].split("/")[0].strip())
            if cfg.get("rag_mode"):         tags.append(cfg["rag_mode"].upper())
            if cfg.get("ens_strategy"):
                s = cfg["ens_strategy"]
                tags.append("RRF" if s == "rrf" else "MoE")

            with col:
                tag_html = "".join(
                    f'<span class="filter-tag" style="background:{color}18;color:{color}">{t}</span>'
                    for t in tags
                )
                st.markdown(
                    f'<div style="font-weight:500;font-size:14px;margin-bottom:4px">'
                    f'<span style="color:{color}">■</span> {tier_label}</div>'
                    + (f'<div style="margin-bottom:8px">{tag_html}</div>' if tags else ""),
                    unsafe_allow_html=True,
                )

                with st.container(border=True):
                    with st.spinner(""):
                        preds, err = get_predictions(cfg, prefix, top_k)

                    if err:
                        st.error(err)
                        all_results[idx] = []
                    elif not preds:
                        st.info("No predictions.")
                        all_results[idx] = []
                    else:
                        max_p = max(p for _, p in preds) or 1.0
                        for i, (w, prob) in enumerate(preds):
                            pct = int((prob / max_p) * 100)
                            c1, c2 = st.columns([1, 3])
                            with c1:
                                if st.button(f"+ {w}", key=f"add_{idx}_{i}_{w}",
                                             use_container_width=True):
                                    _append(w)
                                    st.rerun()
                            with c2:
                                st.markdown(
                                    f'<div class="prob-bar"><div style="width:{pct}%"></div></div>'
                                    f'<small style="opacity:.55">p = {prob:.4f}</small>',
                                    unsafe_allow_html=True,
                                )
                        all_results[idx] = preds

        # ---- Top-1 comparison ----
        valid = {i: r for i, r in all_results.items() if r}
        if len(valid) > 1:
            st.divider()
            st.markdown("### 🆚 Top-1 Comparison")
            cmp_cols = st.columns(n_panels)
            for idx, col in enumerate(cmp_cols):
                preds = all_results.get(idx, [])
                cfg   = configs[idx]
                label = [k for k, v in TIER_OPTIONS.items() if v == cfg["model_tier"]][0]
                with col:
                    if preds:
                        w, p = preds[0]
                        st.metric(label, w, f"p={p:.3f}")
                    else:
                        st.metric(label, "—")

    st.divider()
    st.caption("💡 Click + next to append a word. Options shown depend on the selected model.")


# ============================================================ PAGE 2: SCORE COMPARISON

elif "Score" in page:
    import json, os

    st.markdown("## 📊 Score Comparison Dashboard")

    EVAL_JSON = ROOT / "logs" / "eval_results.json"

    # Default skeleton (filled after evaluate_all.py runs)
    DEFAULT = {
        "4-gram KN":            {"ppl": 1088.10, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": True},
        "HMM-LM":               {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
        "RNN Basic":            {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
        "MaxEnt LM":            {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
        "LSTM Standard":        {"ppl": 118.35, "top1": 0.242, "top5": 0.433, "mrr": 0.185, "latency_ms": None, "trained": True},
        "LSTM + Beam Search":   {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": True},
        "AWD-LSTM":             {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
        "GPT-2 fine-tuned":     {"ppl": 42.74, "top1": 0.352, "top5": 0.562, "mrr": 0.284, "latency_ms": None, "trained": True},
        "Mamba SSM":            {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
        "RAG (Hybrid)":         {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
        "RRF Ensemble":         {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
        "MoE Ensemble":         {"ppl": None, "top1": None, "top5": None, "mrr": None, "latency_ms": None, "trained": False},
    }

    results = dict(DEFAULT)
    if EVAL_JSON.exists():
        try:
            with EVAL_JSON.open() as f:
                saved = json.load(f)
            results.update(saved)
        except Exception:
            pass

    # ---- Manual update ----
    with st.expander("📥 Update results manually (sau khi train xong)"):
        c1, c2 = st.columns(2)
        with c1:
            model_to_edit = st.selectbox("Model", list(results.keys()))
        with c2:
            st.markdown("")
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        with cc1: new_ppl  = st.number_input("PPL",     0.0, step=0.01, key="upd_ppl")
        with cc2: new_top1 = st.number_input("Top-1",   0.0, 1.0, step=0.001, key="upd_t1")
        with cc3: new_top5 = st.number_input("Top-5",   0.0, 1.0, step=0.001, key="upd_t5")
        with cc4: new_mrr  = st.number_input("MRR",     0.0, 1.0, step=0.001, key="upd_mrr")
        with cc5: new_lat  = st.number_input("ms",      0.0, step=0.1,   key="upd_lat")
        if st.button("💾 Save"):
            results[model_to_edit].update({
                "ppl":        new_ppl  if new_ppl  > 0 else results[model_to_edit].get("ppl"),
                "top1":       new_top1 if new_top1 > 0 else results[model_to_edit].get("top1"),
                "top5":       new_top5 if new_top5 > 0 else results[model_to_edit].get("top5"),
                "mrr":        new_mrr  if new_mrr  > 0 else results[model_to_edit].get("mrr"),
                "latency_ms": new_lat  if new_lat  > 0 else results[model_to_edit].get("latency_ms"),
                "trained": True,
            })
            os.makedirs(ROOT / "logs", exist_ok=True)
            with EVAL_JSON.open("w") as f:
                json.dump(results, f, indent=2)
            st.success("Saved! Refresh to see updated charts.")

    # ---- Filter which models ----
    trained_models = [k for k, v in results.items() if v.get("trained")]
    show = st.multiselect("Models to compare", list(results.keys()),
                          default=trained_models or list(results.keys())[:4])

    import pandas as pd
    df = pd.DataFrame([{"Model": k, **results[k]} for k in show if k in results])

    try:
        import plotly.graph_objects as go

        COLORS = ["#5C6BC0","#00838F","#2E7D32","#F57C00","#8E24AA",
                  "#1565C0","#37474F","#AB47BC","#00695C","#6D4C41","#37474F","#455A64"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📉 Perplexity", "🎯 Top-k Accuracy", "🔢 MRR", "⚡ Latency", "📋 Full Table"]
        )

        with tab1:
            sub = df.dropna(subset=["ppl"])
            if not sub.empty:
                fig = go.Figure(go.Bar(
                    x=sub["Model"], y=sub["ppl"],
                    marker_color=[COLORS[i % len(COLORS)] for i in range(len(sub))],
                    text=[f"{v:.1f}" for v in sub["ppl"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Perplexity — lower is better",
                    yaxis_type="log", yaxis_title="PPL (log scale)",
                    xaxis_tickangle=-25, height=400,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No PPL data yet. Run: `python -m src.evaluate_all`")

        with tab2:
            sub = df.dropna(subset=["top1", "top5"], how="all")
            if not sub.empty:
                fig = go.Figure()
                for k_col, color, label in [
                    ("top1", "#26C6DA", "Top-1"),
                    ("top5", "#66BB6A", "Top-5"),
                ]:
                    s2 = sub.dropna(subset=[k_col])
                    if not s2.empty:
                        fig.add_trace(go.Bar(
                            name=label, x=s2["Model"], y=s2[k_col],
                            marker_color=color,
                            text=[f"{v:.3f}" for v in s2[k_col]],
                            textposition="outside",
                        ))
                fig.update_layout(
                    barmode="group",
                    title="Top-k Accuracy — higher is better",
                    yaxis=dict(range=[0, 0.75]),
                    xaxis_tickangle=-25, height=400,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Top-k data yet.")

        with tab3:
            sub = df.dropna(subset=["mrr"])
            if not sub.empty:
                fig = go.Figure(go.Bar(
                    x=sub["Model"], y=sub["mrr"],
                    marker_color="#AB47BC",
                    text=[f"{v:.3f}" for v in sub["mrr"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Mean Reciprocal Rank (MRR) — higher is better",
                    yaxis=dict(range=[0, 0.5]),
                    xaxis_tickangle=-25, height=380,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No MRR data yet.")

        with tab4:
            sub = df.dropna(subset=["latency_ms"])
            if not sub.empty:
                fig = go.Figure(go.Bar(
                    x=sub["Model"], y=sub["latency_ms"],
                    marker_color="#F57C00",
                    text=[f"{v:.1f}ms" for v in sub["latency_ms"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Inference latency per query (ms) — lower is better",
                    yaxis_title="ms",
                    xaxis_tickangle=-25, height=380,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No latency data yet. Run evaluate_all.py to measure.")

        with tab5:
            disp = df.copy()
            for col in ["ppl", "top1", "top5", "mrr", "latency_ms"]:
                if col in disp.columns:
                    disp[col] = disp[col].apply(
                        lambda x: f"{x:.4f}" if isinstance(x, float) else ("✗ not trained" if x is None else str(x))
                    )
            rename = {"ppl": "PPL", "top1": "Top-1", "top5": "Top-5",
                      "mrr": "MRR", "latency_ms": "Latency (ms)", "trained": "Trained"}
            disp = disp.rename(columns=rename)
            cols_show = ["Model", "PPL", "Top-1", "Top-5", "MRR", "Latency (ms)"]
            cols_show = [c for c in cols_show if c in disp.columns]
            st.dataframe(disp[cols_show], use_container_width=True, hide_index=True)

        # Radar
        if len(show) >= 2:
            st.divider()
            st.markdown("#### Radar — multi-metric comparison")
            radar_sel = st.multiselect("Select 2–5 models",
                                       [m for m in show if results.get(m, {}).get("top1")],
                                       default=[m for m in show if results.get(m, {}).get("top1")][:3])
            if 2 <= len(radar_sel) <= 5:
                cols_radar = ["top1", "top5", "mrr"]
                labels_r   = ["Top-1", "Top-5", "MRR"]
                fig = go.Figure()
                for i, name in enumerate(radar_sel):
                    r = results.get(name, {})
                    vals = [r.get(c) or 0.0 for c in cols_radar]
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=labels_r + [labels_r[0]],
                        name=name, line_color=COLORS[i % len(COLORS)],
                        fill="toself", fillcolor=COLORS[i % len(COLORS)], opacity=0.15,
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(range=[0, 0.65])),
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.dataframe(df, use_container_width=True)
        st.info("Install plotly for charts: `pip install plotly`")


# ============================================================ PAGE 3: ARCHITECTURE

elif "Architecture" in page:
    st.markdown("## 📖 Architecture Guide")

    # Colab setup guide
    with st.expander("🚀 Colab Setup — copy-paste guide", expanded=False):
        st.markdown("""
### Upload structure to Google Drive

```
MyDrive/
└── NLP22-n22dccn077/          ← upload this whole folder
    ├── src/                   ← all .py files
    ├── checkpoints/
    │   ├── ngram_4.pkl        ← already have ✓
    │   ├── lstm_best.pt       ← already have ✓
    │   ├── lstm_vocab.pkl     ← already have ✓
    │   └── gpt2-finetuned/   ← already have ✓
    └── data/
        └── processed/
            ├── train_small.txt  ← 27MB — upload this
            └── val_small.txt    ← upload this
```

### Colab notebook template

```python
# Cell 1 — mount drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — cd to project
import os
os.chdir('/content/drive/MyDrive/NLP22-n22dccn077')
!pip install -q nltk torch transformers datasets tqdm

# Cell 3 — train AWD-LSTM (~30 min on T4)
!python -m src.neural_model_2 train --variant awd --epochs 5

# Cell 4 — train ELMo (~40 min on T4)
!python -m src.elmo_embeddings train --epochs 5

# Cell 5 — train Mamba (~50 min on T4)
!python -m src.mamba_model train --epochs 5 --layers 4

# Cell 6 — build indexes (CPU, fast)
!python -m src.rag_dense build --mode hybrid
!python -m src.bm25_reranker build
!python -m src.moe_ensemble train_gate

# Cell 7 — evaluate everything
!python -m src.evaluate_all --limit 300
```
        """)

    # Train commands reference
    with st.expander("⚙️ All training commands"):
        st.markdown("""
```bash
# --- CPU, train locally (fast) ---
python -m src.hmm_model train
python -m src.rnn_model train
python -m src.maxent_model train --epochs 3
python -m src.rag_dense build --mode hybrid
python -m src.bm25_reranker build
python -m src.moe_ensemble train_gate

# --- GPU preferred (Colab/Kaggle) ---
python -m src.neural_model_2 train --variant awd --epochs 5
python -m src.elmo_embeddings train --epochs 5
python -m src.mamba_model train --epochs 5 --layers 4

# --- Evaluate all ---
python -m src.evaluate_2
python -m src.evaluate_2 --models rnn maxent lstm_awd
```
        """)

    # Architecture table
    st.markdown("### Model comparison")
    arch_data = [
        ("4-gram KN",         "Statistical",  "ngram_model.py",       "~1088",  "n/a",    "< 1ms",  "✓ done"),
        ("HMM-LM",            "Generative",   "hmm_model.py",         "~600",   "~0.15",  "~5ms",   "need train"),
        ("RNN Basic",         "Neural RNN",   "rnn_model.py",         "~250",   "~0.30",  "~8ms",   "need train"),
        ("MaxEnt LM",         "Log-linear",   "maxent_model.py",      "~800?",  "~0.20",  "< 5ms",  "need train"),
        ("LSTM Standard",     "Neural RNN",   "neural_model.py",      "~118",   "0.433",  "~10ms",  "✓ done"),
        ("LSTM + Beam",       "Neural RNN",   "neural_model_2.py",    "~118",   "~0.45?", "~30ms",  "reuse LSTM ckpt"),
        ("AWD-LSTM",          "Reg. RNN",     "neural_model_2.py",    "~100?",  "~0.46?", "~12ms",  "need GPU train"),
        ("ELMo",              "BiLSTM emb.",  "elmo_embeddings.py",   "n/a",    "—",      "~20ms",  "optional GPU"),
        ("GPT-2 fine-tuned",  "Transformer",  "finetune_gpt2.py",     "~42.7",  "0.562",  "~80ms",  "✓ done"),
        ("Mamba SSM",         "SSM (2023)",   "mamba_model.py",       "~55?",   "~0.48?", "~25ms",  "need GPU train"),
        ("RAG Hybrid",        "Retrieval+GPT","rag_dense.py",         "n/a",    "~0.57?", "~120ms", "build index"),
        ("RRF Ensemble",      "Rank fusion",  "moe_ensemble.py",      "n/a",    "best?",  "varies", "no training"),
    ]
    hdr = ["Model", "Type", "File", "PPL", "Top-5", "Latency", "Status"]
    st.dataframe(
        pd.DataFrame(arch_data, columns=hdr),
        use_container_width=True, hide_index=True,
    )
    st.caption("? = estimated, not yet measured. Run evaluate_2.py after training to fill in real numbers.")