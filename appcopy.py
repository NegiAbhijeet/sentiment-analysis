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

STOPWORDS = set()
try:
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
    text = re.sub(r"<.*?>", " ", text)     
    text = re.sub(r"[^a-z\s]", " ", text)   
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None, path=None, chunksize=100000):
    if uploaded_file:
        reader = pd.read_csv(uploaded_file, chunksize=chunksize)
    elif path:
        reader = pd.read_csv(path, chunksize=chunksize)
    else:
        return None

    chunks = []
    for chunk in reader:
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df

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

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

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


st.set_page_config(page_title="Customer Sentiment Analysis", page_icon="🗣️", layout="wide")
st.title("🗣️ Customer Sentiment Analysis ")
st.caption("TF‑IDF + Logistic Regression • Quick training • Charts • Word Clouds • Batch scoring")

mode = st.radio(
    "Choose how you want to use the app:",
    ["🛍️ Customer Mode (for buyers)", "📊 Business Mode (for analysts)"],
    horizontal=True
)

if "Customer Mode" in mode:
    st.header("🛍️ Product Search & Sentiment Analysis")

    product_name = st.text_input("Enter product name", placeholder="e.g., Adidas shoes, iPhone 15, Nike shirt")
    platform = st.selectbox("Select website", [ "Flipkart", "Myntra"])

    platform_files = {
        "Flipkart": os.path.join(DATA_DIR, "Flipkart_Reviews.csv"),
        "Myntra": os.path.join(DATA_DIR, "Myntra_Reviews.csv"),
    }

    selected_csv = platform_files.get(platform)
    st.write(f"📂 Using dataset: `{os.path.basename(selected_csv)}`")

    if not os.path.exists(selected_csv):
        st.error(f"❌ Dataset for {platform} not found at:\n{selected_csv}")
        st.stop()


    
    if not os.path.exists(selected_csv):
        st.error(f"❌ Dataset for {platform} not found at: {selected_csv}")
        st.stop()

    df_all = pd.read_csv(selected_csv, nrows=200000, encoding='utf-8', on_bad_lines='skip')
    df_all.columns = df_all.columns.str.lower().str.strip()
    df_all.rename(columns={
        "reviewtext": "reviewtext",
        "summary": "summary",
        "productid": "productid"  # optional, just keeps consistent lowercase
    }, inplace=True)


    if df_all is None or df_all.empty:
        st.error(f"⚠️ Unable to load dataset for {platform}. Check file format (must have review text).")
        st.stop()

    # 🔧 Normalize column names for cross-platform compatibility
    df_all.columns = df_all.columns.str.lower().str.strip()
    df_all.rename(columns={
        "productname": "product",
        "product_title": "product",
        "review": "reviewtext",
        "review_text": "reviewtext",
        "rate": "overall",
        "rating": "overall",
        "summary": "summary"
    }, inplace=True)

# Step 5️⃣ — Filter reviews by product name
    if product_name.strip():
        product_name = product_name.lower().strip()
        search_terms = product_name.split()

        df_filtered = pd.DataFrame()  # default empty

        # 🔹 Check which columns exist for searching
        text_columns = []
        for col in ["product", "summary", "reviewtext"]:
            if col in df_all.columns:
                text_columns.append(col)

        if not text_columns:
            st.error("❌ No suitable columns found for filtering (expected 'product', 'summary', or 'reviewtext').")
            st.stop()

        # 🔹 Smarter matching: all words must appear in the same column
        mask = pd.Series(False, index=df_all.index)
        for col in text_columns:
            mask_col = df_all[col].astype(str).str.lower()
            mask_col = mask_col.apply(lambda x: all(term in x for term in search_terms))
            mask = mask | mask_col

        df_filtered = df_all[mask]

        # 🧩 Show how many reviews were matched
        if df_filtered.empty:
            st.warning(f"No reviews found for '{product_name}' on {platform}. Try another product.")
        else:
            st.success(f"✅ Found {len(df_filtered)} reviews mentioning '{product_name}' on {platform}.")
            st.dataframe(df_filtered.head(5))  # show sample reviews

            # Step 6️⃣ — Sentiment prediction
            if "cleaned" not in df_filtered.columns:
                df_filtered["cleaned"] = (
                    df_filtered[text_columns[0]].astype(str).str.lower()
                )

            tfidf, model = load_or_train_model(df_filtered)
            X_vec = tfidf.transform(df_filtered["cleaned"].fillna(""))
            df_filtered["predicted_sentiment"] = model.predict(X_vec)
            # Step 6️⃣ — Sentiment prediction
            # Make sure the text field exists and is cleaned before transformation
            if "cleaned" not in df_filtered.columns and "reviewtext" in df_filtered.columns:
                df_filtered["cleaned"] = df_filtered["reviewtext"].astype(str).str.lower()

            tfidf, model = load_or_train_model(df_filtered)
            X_vec = tfidf.transform(df_filtered["cleaned"].fillna(""))
            df_filtered["predicted_sentiment"] = model.predict(X_vec)


            # Step 7️⃣ — Overview metrics
            st.subheader(f"Sentiment Overview for '{product_name}' on {platform}")
            pos_rate = (df_filtered["predicted_sentiment"] == "Positive").mean() * 100
            neg_rate = 100 - pos_rate
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positive Sentiment", f"{pos_rate:.1f}%")
            with col2:
                st.metric("Negative Sentiment", f"{neg_rate:.1f}%")
            
            

            # Step 8️⃣ — Word Clouds
            st.markdown("### Word Clouds")
            c1, c2 = st.columns(2)
            with c1:
                make_wordcloud(" ".join(df_filtered[df_filtered["predicted_sentiment"] == "Positive"]["cleaned"]), "Positive")
            with c2:
                make_wordcloud(" ".join(df_filtered[df_filtered["predicted_sentiment"] == "Negative"]["cleaned"]), "Negative")

        # Step 9️⃣ — Display sample reviews / ratings
        if "reviewtext" in df_filtered.columns and "predicted_sentiment" in df_filtered.columns:
            st.markdown("### Example Reviews")
            st.dataframe(df_filtered[["reviewtext", "predicted_sentiment"]].head(10))
        elif "overall" in df_filtered.columns:
            st.markdown("### Example Ratings")

            # ✅ Only show columns that actually exist
            display_cols = [col for col in ["product", "summary", "reviewtext", "overall"] if col in df_filtered.columns]
            if display_cols:
                st.dataframe(df_filtered[display_cols].head(10))
            else:
                st.info("No suitable columns found to display ratings for this dataset.")
        else:
            st.info("No textual reviews or ratings to display for this dataset.")


    else:
        st.info("Enter a product name above to start the analysis.")

    # 📊 Business Mode (for analysts)
elif "Business Mode" in mode:
    st.title("📊 Business Sentiment Dashboard")

    # Select platform
    uploaded_platform = st.selectbox("Select Platform", ["Flipkart", "Myntra"])

    # Define dataset paths (adjust DATA_DIR as per your app setup)
    platform_files = {
        "Flipkart": os.path.join(DATA_DIR, "Flipkart_Reviews.csv"),
        "Myntra": os.path.join(DATA_DIR, "Myntra_Reviews.csv"),
    }

    # Get correct file path
    file_path = platform_files.get(uploaded_platform)

    # Load dataset safely
    if os.path.exists(file_path):
        df_all = pd.read_csv(file_path, nrows=200000, encoding="utf-8", on_bad_lines="skip")
        st.success(f"📂 Loaded {uploaded_platform} dataset successfully!")
    else:
        st.error(f"❌ Dataset for {uploaded_platform} not found at: {file_path}")
        st.stop()

    # Normalize column names
    df_all.columns = df_all.columns.str.lower().str.strip()
    df_all.rename(columns={"productname": "product", "name": "product"}, inplace=True)

    st.success(f"✅ Loaded {len(df_all)} records from {uploaded_platform}")

    # 📊 Dataset Overview
    st.subheader("🧾 Dataset Overview")
    st.dataframe(df_all.head(5))

    # 🏆 Top 10 Highest Rated Products
    # 🏆 Top 10 Highest Rated Products
    if "product" in df_all.columns and "overall" in df_all.columns:
        # Convert 'overall' to numeric safely
        df_all["overall"] = pd.to_numeric(df_all["overall"], errors="coerce")

        # Drop rows with missing or invalid ratings
        df_valid = df_all.dropna(subset=["overall"])

        if not df_valid.empty:
            sentiment_avg = (
                df_valid.groupby("product")["overall"].mean().reset_index()
                .sort_values("overall", ascending=False)
                .head(10)
            )

            st.subheader("🏆 Top 10 Highest Rated Products")
            st.bar_chart(data=sentiment_avg, x="product", y="overall")
        else:
            st.warning("No valid numeric ratings found in this dataset.")


    # 🔥 Most Reviewed Products
    if "product" in df_all.columns:
        review_count = df_all["product"].value_counts().head(10)
        st.subheader("🔥 Most Reviewed Products")
        st.bar_chart(review_count)
        
