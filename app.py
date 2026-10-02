#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Review Sentiment Analyzer - Streamlit app

Run:
    pip install streamlit joblib scikit-learn
    streamlit run app.py

Put model.pkl (and vectorizer.pkl if the model is not a Pipeline) next to app.py.
Without them, a small keyword demo model is used.
"""

import html
import os
import re

import streamlit as st

try:
    import joblib
except ImportError:
    joblib = None

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find(*names):
    for name in names:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    return os.path.join(BASE_DIR, names[0])


MODEL_PATH = _find("model.pkl", "model.joblib", "sentiment_model.pkl", "classifier.pkl")
VECTORIZER_PATH = _find("vectorizer.pkl", "vectorizer.joblib", "tfidf.pkl", "tfidf_vectorizer.pkl")

EXAMPLES = {
    "😍 Loved it": "This product is excellent and I love it! Great quality and fast delivery.",
    "😐 Mixed": "It works, but the battery life is average and the packaging was damaged.",
    "😡 Disappointed": "Terrible experience. It stopped working after two days and support never replied.",
}

POS_WORDS = {"good", "great", "excellent", "love", "loved", "amazing", "perfect", "fast",
             "happy", "best", "awesome", "fantastic", "recommend", "nice", "quality"}
NEG_WORDS = {"bad", "terrible", "awful", "worst", "hate", "broken", "poor", "slow",
             "disappointed", "disappointing", "waste", "refund", "stopped", "never", "damaged"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&display=swap');
:root{--ink:#1B1840;--muted:#6B6A8C;--line:#E4E2F5;--pos:#0E9F6E;--neg:#E5484D;--accent:#6D3FE0;}
html,body,[class*="css"],.stApp{font-family:'Sora',system-ui,sans-serif;color:var(--ink);}
.stApp{background:#F4F2FC;}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0;}
.block-container{max-width:780px;padding-top:1.6rem;padding-bottom:4rem;}

/* hero banner */
.hero{position:relative;overflow:hidden;border-radius:28px;padding:2.4rem 2.2rem 2.2rem;
  background:linear-gradient(125deg,#3A2BD9 0%,#7B3FE4 52%,#E0489B 100%);
  box-shadow:0 24px 50px rgba(88,52,210,.32);color:#fff;margin-bottom:1.6rem;}
.hero:before,.hero:after{content:"";position:absolute;border-radius:50%;background:rgba(255,255,255,.12);}
.hero:before{width:240px;height:240px;right:-70px;top:-90px;}
.hero:after{width:150px;height:150px;right:90px;bottom:-80px;}
.hero-badge{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.35);
  padding:.3rem .8rem;border-radius:999px;font-size:.8rem;font-weight:600;margin-bottom:1rem;}
.hero h1{color:#fff;font-size:2.5rem;font-weight:800;letter-spacing:-.03em;line-height:1.1;margin:0 0 .6rem;padding:0;}
.hero p{color:rgba(255,255,255,.88);font-size:1rem;max-width:46ch;margin:0;}

/* section card */
.card-title{font-weight:700;font-size:1rem;margin:0 0 .2rem;}
.hint{color:var(--muted);font-size:.84rem;margin:0 0 .6rem;}

/* input */
.stTextArea label p{font-weight:600;font-size:.9rem;}
.stTextArea textarea{font-family:'Sora',sans-serif;background:#fff;border:2px solid var(--line);
  border-radius:18px;padding:1rem 1.1rem;font-size:1rem;line-height:1.6;color:var(--ink);
  box-shadow:0 10px 28px rgba(60,40,140,.08);transition:border-color .15s,box-shadow .15s;}
.stTextArea textarea:focus{border-color:var(--accent);box-shadow:0 0 0 5px rgba(109,63,224,.16);}
.stTextArea div[data-baseweb="textarea"]{background:transparent;border:none;}

/* buttons */
.stButton>button{font-family:'Sora',sans-serif;border-radius:999px;border:1.5px solid var(--line);
  background:#fff;color:var(--ink);font-weight:600;font-size:.86rem;padding:.5rem 1rem;
  transition:border-color .15s,transform .1s,color .15s;}
.stButton>button:hover{border-color:var(--accent);color:var(--accent);}
.stButton>button:focus-visible{outline:3px solid rgba(109,63,224,.4);}
.stButton>button:active{transform:scale(.98);}
.stButton>button[kind="primary"]{width:100%;background:linear-gradient(120deg,#5B35DD,#B03FC9);border:none;
  color:#fff;font-size:1.02rem;font-weight:700;padding:.85rem 1.5rem;box-shadow:0 14px 28px rgba(109,63,224,.35);}
.stButton>button[kind="primary"]:hover{color:#fff;filter:brightness(1.07);}

/* result */
.result{margin-top:1.6rem;border-radius:26px;padding:1.8rem;background:#fff;border:2px solid;
  box-shadow:0 20px 44px rgba(60,40,140,.12);animation:reveal .4s ease-out both;}
.result.pos{border-color:#9FE3C8;}
.result.neg{border-color:#F6B4B6;}
.result-grid{display:flex;align-items:center;gap:1.8rem;flex-wrap:wrap;}
.gauge{width:200px;flex:0 0 auto;text-align:center;}
.gauge-num{font-size:1.9rem;font-weight:800;margin-top:-2.6rem;letter-spacing:-.02em;}
.gauge-cap{font-size:.78rem;color:var(--muted);font-weight:500;}
.verdict{flex:1 1 220px;}
.pill{display:inline-block;padding:.3rem .85rem;border-radius:999px;font-weight:700;font-size:.82rem;}
.pos .pill{background:#DDF6EC;color:var(--pos);}
.neg .pill{background:#FDE7E8;color:var(--neg);}
.verdict h2{font-size:1.9rem;font-weight:800;letter-spacing:-.02em;margin:.7rem 0 .3rem;padding:0;}
.pos h2{color:var(--pos);} .neg h2{color:var(--neg);}
.verdict p{color:var(--muted);margin:0;font-size:.95rem;}
.divider{height:1px;background:var(--line);margin:1.4rem 0 1.1rem;}
.read{font-size:1rem;line-height:1.85;}
.read-label{font-size:.84rem;color:var(--muted);margin-bottom:.4rem;font-weight:600;}
mark.p{background:#CDF3E3;color:#06684A;padding:.1rem .35rem;border-radius:7px;font-weight:600;}
mark.n{background:#FBD5D7;color:#A8262B;padding:.1rem .35rem;border-radius:7px;font-weight:600;}
.stats{display:flex;gap:.6rem;margin-top:1.1rem;flex-wrap:wrap;}
.stat{background:#F4F2FC;border-radius:12px;padding:.45rem .85rem;font-size:.82rem;color:var(--muted);}
.stat b{color:var(--ink);}
.footer-note{margin-top:2.4rem;text-align:center;color:var(--muted);font-size:.8rem;}

@keyframes reveal{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
@media (prefers-reduced-motion:reduce){.result{animation:none;}}
@media (max-width:600px){.hero h1{font-size:1.9rem;}.hero{padding:1.8rem 1.4rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if joblib is None or not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None
    return model, vec


def normalize_label(label) -> str:
    s = str(label).strip().lower()
    return "pos" if s.startswith("pos") or s in {"1", "true"} else "neg"


def demo_predict(text):
    words = re.findall(r"[a-z']+", text.lower())
    p = sum(w in POS_WORDS for w in words)
    n = sum(w in NEG_WORDS for w in words)
    if p == n:
        return "pos", 0.5
    score = (p + 1) / (p + n + 2)
    return ("pos" if p > n else "neg"), score


def predict(text):
    """Return (label, positive_score or None, demo_mode)."""
    model, vec = load_artifacts()
    if model is None:
        label, score = demo_predict(text)
        return label, score, True
    X = vec.transform([text]) if vec is not None else [text]
    label = normalize_label(model.predict(X)[0])
    score = None
    if hasattr(model, "predict_proba"):
        for cls, p in zip(getattr(model, "classes_", []), model.predict_proba(X)[0]):
            if normalize_label(cls) == "pos":
                score = float(p)
                break
    return label, score, False


def highlight(text):
    safe = html.escape(text, quote=False)

    def repl(m):
        w = m.group(0)
        if w.lower() in POS_WORDS:
            return f'<mark class="p">{w}</mark>'
        if w.lower() in NEG_WORDS:
            return f'<mark class="n">{w}</mark>'
        return w

    return re.sub(r"[A-Za-z']+", repl, safe).replace("\n", "<br>")


def gauge_svg(pct, color):
    length = 251.3
    return f"""
    <svg viewBox="0 0 200 112" width="200" role="img" aria-label="Positivity score {pct:.0f} percent">
      <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#ECE9FA" stroke-width="16" stroke-linecap="round"/>
      <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16"
            stroke-linecap="round" stroke-dasharray="{length * pct / 100:.1f} {length}"/>
    </svg>"""


# ---------------------------------------------------------------- UI
st.markdown(
    """
    <div class="hero">
        <span class="hero-badge">💬 Sentiment check</span>
        <h1>Review Sentiment Analyzer</h1>
        <p>Paste a customer review and see in seconds whether it is positive or negative.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "review" not in st.session_state:
    st.session_state.review = ""


def set_example(text):
    st.session_state.review = text


st.markdown('<p class="card-title">Start with an example</p><p class="hint">Or write your own below.</p>',
            unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=set_example, args=(text,), use_container_width=True, key=f"ex_{name}")

review = st.text_area("Your review", key="review", height=150,
                      placeholder="Example: This product is excellent and I love it!")

if st.button("Analyze review", type="primary"):
    text = review.strip()
    if not text:
        st.warning("Write or paste a review first, then select Analyze review.")
    else:
        with st.spinner("Reading the review..."):
            label, pos_score, demo = predict(text)

        is_pos = label == "pos"
        cls = "pos" if is_pos else "neg"
        color = "#0E9F6E" if is_pos else "#E5484D"
        score = pos_score if pos_score is not None else (1.0 if is_pos else 0.0)
        shown = (score if is_pos else 1 - score) * 100 if pos_score is not None else 100
        gauge_cap = "Model confidence" if pos_score is not None else "Prediction"
        gauge_val = f"{shown:.0f}%" if pos_score is not None else ("👍" if is_pos else "👎")

        words = len(text.split())
        n_pos = sum(w.lower() in POS_WORDS for w in re.findall(r"[A-Za-z']+", text))
        n_neg = sum(w.lower() in NEG_WORDS for w in re.findall(r"[A-Za-z']+", text))

        st.markdown(
            f"""
            <div class="result {cls}">
              <div class="result-grid">
                <div class="gauge">
                  {gauge_svg(shown if pos_score is not None else 100, color)}
                  <div class="gauge-num" style="color:{color}">{gauge_val}</div>
                  <div class="gauge-cap">{gauge_cap}</div>
                </div>
                <div class="verdict">
                  <span class="pill">{"😊 Positive" if is_pos else "😞 Negative"}</span>
                  <h2>{"Happy customer" if is_pos else "Unhappy customer"}</h2>
                  <p>{"This review speaks well of the product." if is_pos else "This review raises problems with the product."}</p>
                </div>
              </div>
              <div class="divider"></div>
              <div class="read-label">Your review, with key words marked</div>
              <div class="read">{highlight(text)}</div>
              <div class="stats">
                <div class="stat"><b>{words}</b> words</div>
                <div class="stat"><b>{n_pos}</b> positive cues</div>
                <div class="stat"><b>{n_neg}</b> negative cues</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if demo:
            st.caption("Demo mode: model.pkl was not found, so a simple keyword check was used. "
                       "Add your trained model files next to app.py for real predictions.")

st.markdown('<p class="footer-note">Predictions come from a model and can miss sarcasm or mixed feelings.</p>',
            unsafe_allow_html=True)

