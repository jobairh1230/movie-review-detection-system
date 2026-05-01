# app.py — Movie Review Detection System (fast, tuned, stronger per-class F1)

# --- Imports ---
import os, re, random, inspect             # OS paths, regex, RNG seeding, signature inspection
import numpy as np                         # Numerical arrays & ops
import pandas as pd                        # DataFrames & CSV IO
import streamlit as st                     # Web UI framework

from joblib import dump, load              # Model caching to disk
from sklearn.model_selection import train_test_split, GridSearchCV   # Split & hyperparameter search
from sklearn.metrics import f1_score, classification_report          # Metrics (macro/per-class F1)
from sklearn.feature_extraction.text import TfidfVectorizer          # TF-IDF vectorizers
from sklearn.pipeline import Pipeline, FeatureUnion                   # Pipelines + feature unions
from sklearn.naive_bayes import MultinomialNB                        # NB classifier
from sklearn.linear_model import LogisticRegression                  # Logistic regression
from sklearn.multiclass import OneVsRestClassifier                   # OVR wrapper for multi-class
from sklearn.svm import LinearSVC                                    # Linear SVM classifier
from sklearn.calibration import CalibratedClassifierCV               # Probability calibration

# ---------- UI & CSS ----------
st.set_page_config(page_title="Movie Review Detection System", page_icon="🎬", layout="wide")  # Page config
# Inject custom CSS
st.markdown("""                                                                
<style>
.stApp { background: linear-gradient(180deg,#f8fbff 0%,#f3f0ff 45%,#ffffff 100%); }
.card { background:#fff;border:1px solid rgba(0,0,0,.06);box-shadow:0 10px 30px rgba(0,0,0,.07);
        border-radius:18px;padding:22px 24px; }
h1,h2,h3{ letter-spacing:-.3px; color:#0f172a !important; }
.stButton>button{ width:100%;height:56px;border-radius:14px;border:none;
  background:linear-gradient(90deg,#a855f7,#ec4899);color:#fff;font-weight:700;font-size:16px;
  box-shadow:0 10px 24px rgba(236,72,153,.25) }
.stButton>button:hover{ filter:brightness(1.03) }
.badge{ background:#eef2ff;color:#1e3a8a;border:1px solid #dbeafe;padding:3px 10px;border-radius:999px;
        font-size:12px;font-weight:700;margin-left:8px }
.result-banner{ border-radius:18px;padding:20px;border:1px solid;display:flex;gap:14px;align-items:center }
.result-positive{ background:#ecfdf5;border-color:#a7f3d0;color:#065f46 }
.result-neutral{ background:#f8fafc;border-color:#e2e8f0;color:#334155 }
.result-negative{ background:#fef2f2;border-color:#fecaca;color:#7f1d1d }
.conf-row{ margin-top:10px }
.conf-label{ font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px;color:#1e293b }
.conf-bar{ width:100%;height:12px;background:#e5e7eb;border-radius:999px;overflow:hidden }
.conf-inner{ height:100%;background:#0f172a }
.small{ color:#334155;font-size:14px }
.kpis { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px }
.kpi { background:#f8fafc; border:1px solid #e5e7eb; padding:6px 10px; border-radius:999px; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Data helpers ----------
def _find_csv():                                                # Locate IMDB CSV in common paths
    for p in ["IMDB Dataset.csv", "./IMDB Dataset.csv", "/mnt/data/IMDB Dataset.csv"]:
        if os.path.exists(p): return p                          # Return first path that exists
    return None                                                 # None if not found

NEG_TRIGGERS = re.compile(r"\b(?:not|no|never|n't)\b", flags=re.I)  # Regex for negation tokens

def _negation_join(text: str) -> str:                           # Negation handling: "not good" -> "not_good"
    """Join negations with the next word: 'not good' -> 'not_good'."""
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z\s]", text)        # Split into words and punctuation tokens
    out, i = [], 0                                              # Output list and index pointer
    while i < len(tokens):                                      # Iterate tokens
        tok = tokens[i].lower()                                 # Lowercase current token
        # If token is a negation and next token is a word, join with underscore
        if NEG_TRIGGERS.fullmatch(tok) and i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z']+", tokens[i+1]):
            out.append(tok + "_" + tokens[i+1].lower()); i += 2 # Append joined token and skip next
        else:
            out.append(tok); i += 1                              # Otherwise append token as-is
    return " ".join(out)                                         # Recombine to string

def _clean(t: str) -> str:                                       # Basic HTML + whitespace cleaning
    t = re.sub(r"<.*?>", " ", str(t))                            # Remove HTML tags
    t = re.sub(r"\s+", " ", t)                                   # Collapse whitespace
    return t.strip()                                             # Trim ends

def _prep_text(t: str) -> str:                                   # Full preprocessing pipeline
    return _negation_join(_clean(t))                             # Clean then join negations

@st.cache_data(show_spinner=True)                                # Cache loaded DataFrame in Streamlit
def load_data(seed: int = 42, neutral_cap_ratio: float = 0.18):  # Load & label data; synthesize neutral
    """
    Load IMDB, apply negation preprocessing, and synthesize a capped Neutral class.
    Slightly higher neutral cap improves separation from weak positive/negative.
    """
    random.seed(seed); np.random.seed(seed)                      # Set RNG seeds for reproducibility
    path = _find_csv()                                           # Find CSV path
    if not path:                                                 # If not found, show error and stop app
        st.error("`IMDB Dataset.csv` not found beside app.py.")
        st.stop()

    df = pd.read_csv(path).dropna(subset=["review","sentiment"]).copy()  # Read CSV and drop NA rows
    df["review"] = df["review"].apply(_prep_text)                # Preprocess reviews (clean + neg join)
    df["label"] = df["sentiment"].map({"positive": 2, "negative": 0}).astype(int)  # Map to ints

    # Conservative Neutral synthesis                                        # Build weak-neutral class heuristically
    neutral_terms = r"(?:okay|fine|average|meh|so-so|not_bad|not_good|nothing special|decent|alright|ok|mediocre|fair|passable|mixed)"
    strong_pos = r"(?:amazing|excellent|masterpiece|outstanding|brilliant|fantastic|love[d]?|superb|great|phenomenal)"
    strong_neg = r"(?:terrible|awful|horrible|worst|boring|waste|hate[d]?|garbage|mess|poor|unwatchable|bad acting|pathetic|dreadful)"

    cand = df["review"].str.contains(neutral_terms, case=False, na=False, regex=True)     # Neutral cue terms present
    not_strong = ~df["review"].str.contains(strong_pos + "|" + strong_neg, case=False, na=False, regex=True)  # Exclude strong polarity
    mid_len = df["review"].str.len().between(50, 700)                                     # Keep mid-length reviews

    neutral_idx = df.index[cand & not_strong & mid_len].to_numpy()  # Candidate indices for neutral class
    cap = int(len(df) * neutral_cap_ratio)                          # Cap size for neutral class
    if len(neutral_idx) > cap:                                      # If too many candidates, subsample
        neutral_idx = np.random.choice(neutral_idx, size=cap, replace=False)
    df.loc[neutral_idx, "label"] = 1                                # Assign label 1 (Neutral)
    return df                                                       # Return labeled DataFrame

# ---------- Training (fast, tiny grids, cached) ----------
@st.cache_resource(show_spinner=True)                               # Cache trained models across reruns
def train_models(random_state: int = 42):                           # Train 3 models + compute F1s
    """
    Fast tuned profile:
      - Stratified ~4.5k rows (keeps time low)
      - Tiny GridSearchCV (cv=2) for 3 models
      - Hybrid TF-IDF: word 1–3g + char 3–5g (capped, float32)
      - Calibrated LinearSVC for probabilities
      - Stores macro & per-class F1; cached to disk
    """
    os.makedirs("models_cache", exist_ok=True)                      # Ensure cache dir exists
    cache_key = "models_cache/imdb_3class_tuned_smallcv.joblib"     # Cache file path
    if os.path.exists(cache_key):                                   # Load from disk if present
        return load(cache_key)

    df = load_data(seed=random_state)                               # Load (possibly synthesized) dataset

    # Stratified subsample (~4.5k) for speed but better class coverage than 4k
    target_n = 4500                                                 # Target training size for speed
    if len(df) > target_n:                                          # If dataset larger than target
        parts = []                                                  # Collect per-class samples
        per = target_n // 3                                         # Rough per-class quota
        for c in [0,1,2]:                                           # For each class: 0=Neg,1=Neu,2=Pos
            part = df[df["label"] == c]                             # Filter class rows
            parts.append(part.sample(min(per, len(part)), random_state=random_state))  # Sample up to quota
        df = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)  # Shuffle & reset

    X_train, X_test, y_train, y_test = train_test_split(            # Train/validation split (stratified)
        df["review"], df["label"], test_size=0.2, stratify=df["label"], random_state=random_state
    )

    # Hybrid features with caps (keeps speed + captures subtlety)
    word_vec = TfidfVectorizer(lowercase=True, stop_words=None,     # Word-level TF-IDF (1–3 grams)
                               ngram_range=(1,3), max_df=0.55, min_df=3,
                               max_features=55_000, sublinear_tf=True, dtype=np.float32)
    char_vec = TfidfVectorizer(analyzer="char", lowercase=True,     # Char-level TF-IDF (3–5 grams)
                               ngram_range=(3,5), min_df=3,
                               max_features=22_000, sublinear_tf=True, dtype=np.float32)
    hybrid = FeatureUnion([("word", word_vec), ("char", char_vec)]) # Combine word & char features

    models, f1_macro, f1_per_class = {}, {}, {}                     # Dicts to hold models & metrics

    # --- 1) Logistic Regression (OVR) ---
    lr_pipe = Pipeline([("vec", hybrid),                            # Pipeline: features -> OVR(LogReg)
                        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=350, n_jobs=-1)))])
    lr_grid = {                                                     # Tiny hyperparameter grid
        "clf__estimator__C": [1.0, 1.8],
        "clf__estimator__class_weight": [None, "balanced"],
    }
    lr_gs = GridSearchCV(lr_pipe, lr_grid, scoring="f1_macro", cv=2, n_jobs=-1, verbose=0).fit(X_train, y_train)  # Grid search
    models["Logistic Regression"] = lr_gs.best_estimator_           # Save best pipeline
    y_pred_lr = lr_gs.best_estimator_.predict(X_test)               # Predict on holdout
    f1_macro["Logistic Regression"] = f1_score(y_test, y_pred_lr, average="macro")  # Macro-F1

    # --- 2) Naive Bayes ---
    nb_pipe = Pipeline([("vec", hybrid), ("clf", MultinomialNB())]) # Pipeline: features -> NB
    nb_grid = {"clf__alpha": [0.5, 0.8]}                            # Tiny smoothing grid
    nb_gs = GridSearchCV(nb_pipe, nb_grid, scoring="f1_macro", cv=2, n_jobs=-1, verbose=0).fit(X_train, y_train)  # Grid search
    models["Naive Bayes"] = nb_gs.best_estimator_                   # Save best NB pipeline
    y_pred_nb = nb_gs.best_estimator_.predict(X_test)               # Predict on holdout
    f1_macro["Naive Bayes"] = f1_score(y_test, y_pred_nb, average="macro")  # Macro-F1

    # --- 3) LinearSVC (calibrated) ---
    svm_base = Pipeline([("vec", hybrid), ("clf", LinearSVC())])    # Base pipeline: features -> LinearSVC
    svm_grid = {"clf__C": [1.0, 1.6], "clf__class_weight": [None, "balanced"]}  # Tiny grid
    svm_gs = GridSearchCV(svm_base, svm_grid, scoring="f1_macro", cv=2, n_jobs=-1, verbose=0).fit(X_train, y_train)  # Grid search

    arg = "estimator" if "estimator" in inspect.signature(CalibratedClassifierCV).parameters else "base_estimator"  # API compat
    best_vec = svm_gs.best_estimator_.named_steps["vec"]            # Extract best vectorizer union
    best_svc = svm_gs.best_estimator_.named_steps["clf"]            # Extract best LinearSVC
    svm_cal = Pipeline([("vec", best_vec),                          # Rebuild pipeline with calibrator
                        ("clf", CalibratedClassifierCV(**{arg: best_svc}, method="sigmoid", cv=2))]).fit(X_train, y_train)
    models["Support Vector Machine"] = svm_cal                      # Save calibrated SVM
    y_pred_svm = svm_cal.predict(X_test)                            # Predict on holdout
    f1_macro["Support Vector Machine"] = f1_score(y_test, y_pred_svm, average="macro")  # Macro-F1

    # Per-class F1 (0=Neg,1=Neu,2=Pos)
    def per_class_f1(y_true, y_pred):                               # Helper to compute class-wise F1 (percent)
        rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # Dict metrics
        return {
            "Negative": round(100*rpt.get("0", {}).get("f1-score", 0.0), 1),            # F1 for class 0
            "Neutral":  round(100*rpt.get("1", {}).get("f1-score", 0.0), 1),            # F1 for class 1
            "Positive": round(100*rpt.get("2", {}).get("f1-score", 0.0), 1),            # F1 for class 2
        }

    f1_per_class["Logistic Regression"]     = per_class_f1(y_test, y_pred_lr)  # Store per-class F1s
    f1_per_class["Naive Bayes"]             = per_class_f1(y_test, y_pred_nb)
    f1_per_class["Support Vector Machine"]  = per_class_f1(y_test, y_pred_svm)

    dump((models, f1_macro, f1_per_class), cache_key)               # Persist (models + metrics) to disk
    return models, f1_macro, f1_per_class                           # Return for UI use

# ---------- UI helpers ----------
def _emoji(s): return {"Positive":"😊","Neutral":"😐","Negative":"😞"}[s]  # Map class -> emoji
def _conf_row(label, pct):                                           # Render a single confidence bar row
    st.markdown(
        f"""
        <div class="conf-row">
          <div class="conf-label">{_emoji(label)} {label} &nbsp; {pct:.1f}%</div>
          <div class="conf-bar"><div class="conf-inner" style="width:{pct:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True
    )

# ---------- Header ----------
st.markdown("<h1 style='text-align:center'>Movie Review Detection System</h1>", unsafe_allow_html=True)  # Title
st.markdown("<p style='text-align:center;color:#1e293b'>Discover the emotions hidden in movie reviews using advanced AI</p>", unsafe_allow_html=True)  # Subtitle
st.markdown("<br>", unsafe_allow_html=True)                                          # Spacer

# ---------- Layout ----------
left, right = st.columns([1,1], gap="large")                                        # Two-column layout

with left:                                                                          # Left card: input + model choice
    st.markdown('<div class="card">', unsafe_allow_html=True)                       # Card container
    st.markdown("### Enter Your Text")                                              # Section heading
    st.markdown('<p class="small">Type or paste any movie review below to analyze its sentiment</p>', unsafe_allow_html=True)  # Help text
    review = st.text_area(" ", height=150, placeholder="“It was okay, nothing special but not terrible either.”")  # Input box

    MODELS, F1S, F1S_PC = train_models()                                            # Train/load models + metrics
    names = list(MODELS.keys())                                                     # Model names
    model_name = st.selectbox("Select Model", names, index=0)                       # Choose model

    st.markdown(f'<span class="badge">Macro F1: {F1S[model_name]*100:.1f}%</span>', unsafe_allow_html=True)  # Show macro F1
    pcs = F1S_PC[model_name]                                                        # Per-class F1 dict for chosen model
    st.markdown(                                                                    # Show per-class F1 as chips
        f"""<div class="kpis">
              <div class="kpi">Neg F1: {pcs['Negative']:.1f}%</div>
              <div class="kpi">Neu F1: {pcs['Neutral']:.1f}%</div>
              <div class="kpi">Pos F1: {pcs['Positive']:.1f}%</div>
            </div>""",
        unsafe_allow_html=True
    )

    go = st.button("Analyze Sentiment")                                             # Action button to run prediction
    st.markdown('</div>', unsafe_allow_html=True)                                   # Close card

with right:                                                                         # Right card: results
    st.markdown('<div class="card">', unsafe_allow_html=True)                       # Card container
    st.markdown("### Analysis Results")                                             # Section heading

    if go and review.strip():                                                       # If button clicked and text provided
        model = MODELS[model_name]                                                  # Get selected model pipeline
        proba = model.predict_proba([_prep_text(review)])[0]                        # Predict class probabilities
        labels = ["Negative","Neutral","Positive"]                                  # Class labels
        idx = int(np.argmax(proba))                                                 # Index of highest prob class
        label = labels[idx]; conf = float(proba[idx])                               # Selected label + confidence

        css_cls = "result-positive" if label=="Positive" else ("result-negative" if label=="Negative" else "result-neutral")  # Banner style
        st.markdown(                                                                # Render prediction banner
            f"""
            <div class="result-banner {css_cls}">
                <div style="font-size:28px">{_emoji(label)}</div>
                <div>
                    <div style="font-weight:800;font-size:20px">Predicted Sentiment: {label}</div>
                    <div>This review expresses {label.lower()} sentiment.</div>
                    <div style="opacity:.7;margin-top:6px">Confidence: <b>{conf:.3f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("**Prediction Confidence**")                                     # Confidence header
        for lbl, p in zip(labels, proba):                                            # For each class probability
            _conf_row(lbl, p*100)                                                    # Draw bar row

        st.markdown(                                                                 # Footer line with metrics
            f"<p class='small'>Using <b>{model_name}</b> • Macro F1: {F1S[model_name]*100:.1f}% · "
            f"Neg F1: {pcs['Negative']:.1f}% · Neu F1: {pcs['Neutral']:.1f}% · Pos F1: {pcs['Positive']:.1f}%</p>",
            unsafe_allow_html=True
        )
    elif go:                                                                         # If button clicked but no text
        st.warning("Please enter a review first.")                                   # Prompt user to enter text
    st.markdown('</div>', unsafe_allow_html=True)                                    # Close card
