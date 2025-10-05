# app.py — Streamlit Sentiment Analysis with TF‑IDF + Logistic Regression
# - One-file app: will load saved models if present; else train from CSV automatically
# - Provides: single-text prediction, batch CSV upload, charts (distribution + word clouds),
#             optional (re)training button, and evaluation metrics
#
# Project structure (suggested):
# sentiment_analysis_project/
#   ├── app/
#   │    └── app.py                <-- this file
#   ├── data/
#   │    └── Amazon_Reviews.csv    <-- place dataset here (reviewText, overall)
#   ├── models/
#   └── results/
#
# Run:  streamlit run app/app.py

import os
import re
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------
# Config & Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl")
DEFAULT_DATASET = os.path.join(DATA_DIR, "Amazon_Reviews.csv")

# -----------------------
# Utilities
# -----------------------
STOPWORDS = set()
try:
    # Try to load NLTK stopwords if available; otherwise fall back to a small static set
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {
        "a","an","the","and","or","but","if","to","of","in","on","for","with","at","by","from",
        "is","it","this","that","these","those","am","are","was","were","be","been","being","as",
        "i","me","my","we","our","you","your","he","him","his","she","her","they","them","their"
    }


def label_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating in [4, 5]:
        return "Positive"
    else:
        return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)       # remove HTML
    text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None):
    """Load dataset either from uploaded CSV or default path. Returns cleaned df with labels."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists(DEFAULT_DATASET):
        df = pd.read_csv(DEFAULT_DATASET)
    else:
        return None

    # Keep only needed columns if present
    cols = df.columns.str.lower()
    if 'reviewtext' in cols and 'overall' in cols:
        # Map back to original casing
        review_col = df.columns[cols.get_loc('reviewtext')]
        overall_col = df.columns[cols.get_loc('overall')]
        df = df[[review_col, overall_col]].rename(columns={review_col: 'reviewText', overall_col: 'overall'})
    else:
        # Try common alternatives
        candidates_text = [c for c in df.columns if c.lower() in {"review", "text", "content", "review_text", "reviewtext"}]
        candidates_label = [c for c in df.columns if c.lower() in {"overall", "rating", "stars", "score"}]
        if candidates_text and candidates_label:
            df = df[[candidates_text[0], candidates_label[0]]].rename(columns={candidates_text[0]: 'reviewText', candidates_label[0]: 'overall'})
        else:
            return None

    df['sentiment'] = df['overall'].apply(label_sentiment)
    df = df.dropna(subset=['sentiment', 'reviewText']).copy()
    df['cleaned'] = df['reviewText'].apply(clean_text)
    df = df[df['cleaned'].str.len() > 0]
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_or_train_model(df: pd.DataFrame, retrain: bool = False):
    """Load TF-IDF + LogisticRegression from disk; train if missing or retrain=True."""
    if (not retrain) and os.path.exists(TFIDF_PATH) and os.path.exists(MODEL_PATH):
        tfidf = joblib.load(TFIDF_PATH)
        model = joblib.load(MODEL_PATH)
        return tfidf, model

    # Train
    X = df['cleaned']
    y = df['sentiment']

    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    # Persist
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(model, MODEL_PATH)

    return tfidf, model


def plot_distribution(df: pd.DataFrame):
    counts = df['sentiment'].value_counts().reindex(["Positive", "Negative"]).fillna(0)
    fig, ax = plt.subplots(figsize=(5,3))
    counts.plot(kind='bar', ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig, clear_figure=True)


def make_wordcloud(text: str, title: str):
    if not text.strip():
        st.info(f"No text available for {title} word cloud.")
        return
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc)
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


def export_report(y_true, y_pred):
    rep = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"]) if len(set(y_true))==2 else confusion_matrix(y_true, y_pred)

    st.subheader("Classification Report")
    st.code(rep)

    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["Actual Positive","Actual Negative"], columns=["Pred Positive","Pred Negative"]))


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Customer Sentiment Analysis", page_icon="🗣️", layout="wide")
st.title("🗣️ Customer Sentiment Analysis — Amazon Reviews")
st.caption("TF‑IDF + Logistic Regression • Quick training • Charts • Word Clouds • Batch scoring")

with st.sidebar:
    st.header("Data Source")
    up = st.file_uploader("Upload CSV (optional)", type=["csv"], help="Must include review text and rating columns.")
    process_file = st.button("📂 Process Uploaded File")   # 👈 ADD THIS BUTTON HERE

    st.markdown("""**Expected columns:**
- *reviewText* (or review/text/content)
- *overall* (or rating/stars/score) → we convert 1–2 = Negative, 4–5 = Positive, drop 3
""")

    st.header("Training")
    retrain_flag = st.button("🔁 Retrain model now")


# Load data only when user clicks Process
_df = None
if process_file and up is not None:
    _df = load_dataset(uploaded_file=up)
elif up is None and os.path.exists(DEFAULT_DATASET):
    _df = load_dataset()   # fallback to default dataset

if _df is None or _df.empty:
    st.warning("No dataset found. Upload a CSV and click '📂 Process Uploaded File', or place 'Amazon_Reviews.csv' under data/.")
    st.stop()


# Load or train model
with st.spinner("Preparing model…"):
    tfidf, model = load_or_train_model(_df, retrain=retrain_flag)

# Tabs
predict_tab, insights_tab, batch_tab = st.tabs(["🔮 Predict", "📊 Insights", "📦 Batch Scoring"]) 

with predict_tab:
    st.subheader("Single Review Prediction")
    user_text = st.text_area("Enter a product review:", height=140, placeholder="Type or paste any review text…")
    if st.button("Predict Sentiment", use_container_width=True):
        if user_text.strip():
            cleaned = clean_text(user_text)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = getattr(model, "predict_proba", lambda X: None)(vec)
            st.success(f"Predicted Sentiment: **{pred}**")
            if proba is not None:
                st.write({"Positive": float(proba[0, list(model.classes_).index("Positive")]) if "Positive" in model.classes_ else None,
                          "Negative": float(proba[0, list(model.classes_).index("Negative")]) if "Negative" in model.classes_ else None})
        else:
            st.info("Please enter some text to predict.")

with insights_tab:
    st.subheader("Dataset Insights")

    col1, col2 = st.columns(2)
    with col1:
        plot_distribution(_df)
    with col2:
        st.write("Sample reviews:")
        st.dataframe(_df[['reviewText','overall','sentiment']].head(10))

    st.markdown("---")
    st.subheader("Word Clouds")
    pos_text = " ".join(_df[_df['sentiment']=="Positive"]["cleaned"].tolist())
    neg_text = " ".join(_df[_df['sentiment']=="Negative"]["cleaned"].tolist())

    c1, c2 = st.columns(2)
    with c1:
        make_wordcloud(pos_text, "Positive Reviews")
    with c2:
        make_wordcloud(neg_text, "Negative Reviews")

with batch_tab:
    st.subheader("Batch Scoring & Evaluation")
    st.write("Upload a CSV to score new reviews or evaluate on a labeled set.")
    eval_file = st.file_uploader("Upload CSV for Scoring/Evaluation", type=["csv"], key="eval")

    if eval_file is not None:
        df_new = pd.read_csv(eval_file)
        # Try to detect text column
        text_col = None
        for cand in ["reviewText","review","text","content","review_text","reviewtext"]:
            if cand in df_new.columns:
                text_col = cand
                break
        if text_col is None:
            st.error("Could not find a text column. Include a column like 'reviewText' or 'text'.")
        else:
            df_new['cleaned'] = df_new[text_col].astype(str).apply(clean_text)
            X_new = tfidf.transform(df_new['cleaned'])
            preds = model.predict(X_new)
            df_new['prediction'] = preds
            st.download_button("⬇️ Download predictions", data=df_new.to_csv(index=False).encode('utf-8'), file_name="predictions.csv", mime="text/csv")

            # If labels exist, evaluate
            label_col = None
            for cand in ["overall","rating","stars","score","sentiment"]:
                if cand in df_new.columns:
                    label_col = cand
                    break
            if label_col is not None:
                if label_col.lower() == 'sentiment':
                    true_labels = df_new[label_col]
                else:
                    true_labels = df_new[label_col].apply(lambda r: label_sentiment(int(r)) if pd.notnull(r) else None)
                mask = true_labels.notna()
                if mask.any():
                    y_true = true_labels[mask]
                    y_pred = df_new.loc[mask, 'prediction']
                    export_report(y_true, y_pred)
                else:
                    st.info("No evaluable labels found in this file (only 1–2–4–5 ratings or explicit Positive/Negative supported).")
            else:
                st.info("No label column found — generated predictions only.")

st.markdown("---")
st.caption("Built with scikit‑learn · Streamlit · WordCloud. For a BERT/LSTM upgrade, integrate a REST endpoint or a saved transformer model and swap the predictor.")
