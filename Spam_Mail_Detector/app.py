"""
Spam Mail Detector - Web App (Streamlit)  |  Dark neon theme

Needs these files in a 'models' folder next to this app.py:
    vectorizer.pkl, naive_bayes.pkl, logistic_regression.pkl

Run:
    python -m streamlit run app.py
"""
import html
import os
import re

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="centered")

# ---------------------------------------------------------------
# Styling (dark background + bright neon colours)
# ---------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(circle at 10% 15%, rgba(56, 189, 248, 0.30), transparent 38%),
        radial-gradient(circle at 90% 10%, rgba(236, 72, 153, 0.30), transparent 40%),
        radial-gradient(circle at 75% 92%, rgba(139, 92, 246, 0.32), transparent 42%),
        linear-gradient(160deg, #070b16 0%, #0d1224 50%, #150b2e 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 820px; padding-top: 2rem; }

/* ----- Hero ----- */
.hero { text-align: center; padding: 10px 0 22px 0; }
.hero .badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    font-weight: 600;
    color: #67e8f9;
    background: rgba(34, 211, 238, 0.10);
    border: 1px solid rgba(34, 211, 238, 0.45);
    box-shadow: 0 0 18px rgba(34, 211, 238, 0.25);
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    margin: 14px 0 6px 0;
    line-height: 1.15;
}
.hero .grad {
    background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #94a3b8; font-size: 1.05rem; margin: 0; }

/* ----- Labels & text ----- */
[data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; font-weight: 600; }
[data-testid="stCaptionContainer"] { color: #64748b !important; }

/* ----- Card around the form ----- */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 22px;
    padding: 26px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 45px rgba(139, 92, 246, 0.18);
}

/* ----- Inputs ----- */
.stTextInput input, .stTextArea textarea {
    background: rgba(8, 12, 28, 0.75) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148, 163, 184, 0.28) !important;
    border-radius: 12px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.35) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #64748b !important; }

div[data-baseweb="select"] > div {
    background: rgba(8, 12, 28, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.28) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] div { color: #e2e8f0 !important; }
div[data-baseweb="select"] svg { fill: #a5b4fc !important; }
div[data-baseweb="popover"] ul { background: #121833 !important; }
div[data-baseweb="popover"] li { background: #121833 !important; color: #e2e8f0 !important; }
div[data-baseweb="popover"] li:hover { background: rgba(139, 92, 246, 0.35) !important; }

/* ----- Buttons ----- */
[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1rem;
    font-size: 1.15rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899);
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.5);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 40px rgba(236, 72, 153, 0.6);
    color: white;
}
.stButton > button {
    width: 100%;
    background: rgba(255, 255, 255, 0.05);
    color: #c4b5fd;
    border: 1px solid rgba(167, 139, 250, 0.5);
    border-radius: 12px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: rgba(167, 139, 250, 0.18);
    border-color: #a78bfa;
    color: #ffffff;
}

/* ----- Result cards ----- */
.res {
    border-radius: 22px;
    padding: 28px 26px;
    margin-top: 26px;
    text-align: center;
    animation: pop 0.45s ease;
}
@keyframes pop { from { transform: scale(0.94); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.res .verdict { font-size: 2.5rem; font-weight: 800; margin: 0; }
.res .sub { font-size: 1.05rem; margin: 8px 0 18px 0; color: #cbd5e1; }
.res.spam {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.20), rgba(168, 85, 247, 0.16));
    border: 1px solid #fb7185;
    box-shadow: 0 0 45px rgba(244, 63, 94, 0.40);
}
.res.spam .verdict { color: #fb7185; text-shadow: 0 0 22px rgba(244, 63, 94, 0.7); }
.res.ham {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.20), rgba(6, 182, 212, 0.16));
    border: 1px solid #34d399;
    box-shadow: 0 0 45px rgba(16, 185, 129, 0.38);
}
.res.ham .verdict { color: #6ee7b7; text-shadow: 0 0 22px rgba(16, 185, 129, 0.7); }

.meter-label { text-align: left; font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px; }
.meter {
    position: relative;
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: linear-gradient(90deg, #34d399, #facc15, #f43f5e);
}
.meter .cover { position: absolute; top: 0; bottom: 0; right: 0; background: #151a33; }

.chips-title { text-align: left; margin: 20px 0 6px 0; font-size: 0.85rem; color: #94a3b8; }
.chip {
    display: inline-block;
    margin: 4px 6px 0 0;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.88rem;
    color: #fda4af;
    background: rgba(244, 63, 94, 0.14);
    border: 1px solid rgba(244, 63, 94, 0.55);
}
.footer { text-align: center; color: #475569; margin-top: 44px; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Load model + NLP resources (cached so they load only once)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_resources():
    for pkg in ["stopwords", "punkt", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
    models = {
        "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "models", "naive_bayes.pkl")),
        "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "models", "logistic_regression.pkl")),
    }
    return vectorizer, models, set(stopwords.words("english"))


def preprocess(text, stop_words):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = [t for t in word_tokenize(text) if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def spam_signals(clean_text, top_n=6):
    """Words in the email that push the Logistic Regression model towards 'spam'."""
    vec = vectorizer.transform([clean_text])
    coefs = models["Logistic Regression"].coef_[0]
    names = vectorizer.get_feature_names_out()
    scored = [(names[i], vec[0, i] * coefs[i]) for i in vec.nonzero()[1]]
    scored = [(w, s) for w, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:top_n]]


try:
    vectorizer, models, STOP_WORDS = load_resources()
except FileNotFoundError:
    st.error(
        f"Model files not found in: {os.path.join(BASE_DIR, 'models')}\n\n"
        "Run phase1 to phase5 first, and keep app.py in the same folder as the 'models' folder."
    )
    raise SystemExit

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<div class="badge">AI-POWERED SPAM FILTER</div>'
    '<h1>📧 <span class="grad">Spam Mail Detector</span></h1>'
    "<p>Enter your email's subject and message to find out if it's spam.</p>"
    "</div>",
    unsafe_allow_html=True,
)

EXAMPLES = {
    "spam": (
        "You have won a FREE prize!",
        "Congratulations! You've been selected to win a free iPhone. "
        "Click the link now and claim your prize. Urgent, offer ends today!",
    ),
    "ham": (
        "Lunch tomorrow?",
        "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    ),
}


def fill_example(kind):
    st.session_state["subject"], st.session_state["body"] = EXAMPLES[kind]


c1, c2 = st.columns(2)
c1.button("⚡ Fill a spam example", on_click=fill_example, args=("spam",))
c2.button("✉️ Fill a normal example", on_click=fill_example, args=("ham",))

with st.form("mail_form"):
    subject = st.text_input("Subject", key="subject", placeholder="e.g. Your account needs verification")
    body = st.text_area("Message", key="body", height=190, placeholder="Paste or type the email message here...")
    model_name = st.selectbox("Model", list(models.keys()))
    submitted = st.form_submit_button("🔍 Check Email")

if submitted:
    full_text = f"{subject} {body}".strip()
    clean = preprocess(full_text, STOP_WORDS) if full_text else ""

    if not full_text:
        st.warning("Please enter a subject or message first.")
    elif not clean:
        st.info("There isn't enough readable text to analyse. Try a longer message.")
    else:
        model = models[model_name]
        vec = vectorizer.transform([clean])
        pred = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]
        confidence = proba[pred] * 100
        spam_pct = proba[1] * 100

        meter = (
            f'<div class="meter-label">Spam probability: {spam_pct:.1f}%</div>'
            f'<div class="meter"><div class="cover" style="left:{spam_pct:.1f}%"></div></div>'
        )

        if pred == 1:
            chips = "".join(f'<span class="chip">{html.escape(w)}</span>' for w in spam_signals(clean))
            chips_html = f'<div class="chips-title">Words that raised the spam score</div>{chips}' if chips else ""
            st.markdown(
                '<div class="res spam">'
                '<p class="verdict">🚨 SPAM</p>'
                f'<p class="sub">This email looks like spam. Confidence: <b>{confidence:.1f}%</b></p>'
                f"{meter}{chips_html}"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="res ham">'
                '<p class="verdict">✅ NOT SPAM</p>'
                f'<p class="sub">This email looks safe. Confidence: <b>{confidence:.1f}%</b></p>'
                f"{meter}"
                "</div>",
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="footer">Built with Python · scikit-learn · Streamlit</div>',
    unsafe_allow_html=True,
)