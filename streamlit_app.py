"""SentiLens | Streamlit Community Cloud entrypoint."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment_engine import LABELS, load_original_model, predict

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="SentiLens | Review Intelligence", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2.2rem;}
.hero {background: linear-gradient(115deg,#101b33,#162e50 60%,#1a484f);
       border:1px solid #285073;border-radius:22px;padding:2rem 2.2rem;margin-bottom:1.5rem;}
.hero h1{font-size:2.7rem;line-height:1.1;margin:0;color:#f5f8ff;}
.hero p{color:#bfd1df;margin:.9rem 0 0;font-size:1.08rem;}
.eyebrow{color:#70e0c4!important;letter-spacing:.14em;font-weight:700;font-size:.77rem;}
.smallmuted{font-size:.86rem;color:#97aec3;}
.result{border-radius:16px;padding:1rem 1.3rem;margin:1rem 0;
        background:#132238;border:1px solid #2a435d;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading classifier...")
def get_artifacts():
    return load_original_model(ROOT)

try:
    ARTIFACTS = get_artifacts()
except Exception as exc:
    ARTIFACTS = None
    st.sidebar.error("The trained artifacts could not be loaded. Using demo mode.")
    st.sidebar.caption(type(exc).__name__)

mode_name = "Trained ML model" if ARTIFACTS is not None else "Illustrative demo · rules-based"

with st.sidebar:
    st.markdown("### 🔎 SentiLens")
    st.caption("Review sentiment intelligence")
    st.markdown("**Analysis engine**")
    if ARTIFACTS is None:
        st.warning("Demo mode — `artifacts/NLP.pkl` was not included.")
        st.caption("Predictions are illustrative keyword rules, not your notebook's trained model. No model accuracy or confidence is claimed.")
    else:
        st.success("Notebook-trained classifier loaded")
        st.caption("Logistic Regression · TF-IDF · 3 sentiment labels")
    st.divider()
    st.markdown("**Labels**")
    st.markdown("🟢 Positive  ·  ⚪ Neutral  ·  🔴 Negative")
    st.divider()
    st.caption("Built from a review-analysis notebook. Developed as a data science portfolio demonstration.")

st.markdown("""
<div class="hero"><div class="eyebrow">NATURAL LANGUAGE PROCESSING / REVIEW ANALYTICS</div>
<h1>Turn reviews into insights.</h1>
<p>Classify customer feedback, inspect sentiment patterns, and analyze batches of reviews in seconds.</p></div>
""", unsafe_allow_html=True)

st.caption(f"Current analysis mode: **{mode_name}**")

tab1, tab2, tab3 = st.tabs(["📝 Analyze a review", "📊 Batch insights", "🧠 Method & limitations"])
with tab1:
    left, right = st.columns([3, 2], gap="large")
    with left:
        examples = {
            "Write my own": "",
            "Positive example": "The quality is excellent, delivery was fast, and I absolutely love it!",
            "Neutral example": "The package arrived yesterday. It is a standard item with a basic finish.",
            "Negative example": "Very disappointing quality. The device broke quickly and customer support was terrible.",
        }
        choice = st.selectbox("Try a sample review", list(examples))
        review = st.text_area("Customer review", value=examples[choice], height=170, max_chars=5000,
                              placeholder="Paste a product review or customer comment...")
        analyze_clicked = st.button("Analyze sentiment", type="primary", use_container_width=True)
        st.caption("English-language text only. Up to 5,000 characters per review.")
    with right:
        st.markdown("#### Sentiment result")
        if analyze_clicked:
            if not review.strip():
                st.warning("Enter a review to analyze.")
            else:
                try:
                    result = predict(review, ARTIFACTS)
                    marks = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
                    st.markdown(f'<div class="result"><div class="smallmuted">PREDICTED CLASS</div><h2>{marks.get(result["label"], "🔎")} {result["label"]}</h2></div>', unsafe_allow_html=True)
                    if result.get("scores"):
                        st.markdown("**Model class probabilities**")
                        for label in ["Positive", "Neutral", "Negative"]:
                            val = float(result["scores"].get(label, 0.0))
                            st.write(f"{label}: {val:.1%}")
                            st.progress(min(1.0, max(0.0, val)))
                        st.caption("Probabilities are model outputs, not guaranteed real-world correctness.")
                    else:
                        st.info("Illustrative keyword classification. No statistical confidence is available.")
                        matched = result.get("matched", [])
                        if matched:
                            st.write("**Matched words**: " + ", ".join(f"{m['word']} ({m['effect']})" for m in matched[:14]))
                        else:
                            st.write("No sentiment-bearing keywords found. The default is Neutral.")
                except Exception as exc:
                    st.error(f"Could not analyze this review ({type(exc).__name__}).")
                    st.caption("Check that the model and vectorizer are compatible and required NLTK resources are available.")
        else:
            st.info("Enter a review and select **Analyze sentiment** to see the result.")

with tab2:
    st.markdown("#### Analyze feedback at scale")
    st.write("Upload a CSV with a `review` column (up to 1,000 rows). Download labeled results.")
    with open(ROOT / "examples" / "reviews.csv", "rb") as file:
        st.download_button("Download sample CSV", data=file.read(), file_name="reviews_sample.csv", mime="text/csv")
    upload = st.file_uploader("Upload review CSV", type="csv", key="batch_upload")
    if upload is not None:
        try:
            if upload.size > 2_000_000:
                st.error("Upload must be below 2 MB.")
            else:
                data = pd.read_csv(upload)
                if "review" not in data.columns:
                    st.error("Missing column: `review`.")
                elif len(data) > 1000:
                    st.error("Please upload no more than 1,000 rows.")
                elif data.empty:
                    st.warning("Your CSV contains no rows.")
                else:
                    if data["review"].isna().any() or not data["review"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                        st.error("Each row must have a non-empty review.")
                    else:
                        results = [predict(str(t)[:5000], ARTIFACTS)["label"] for t in data["review"]]
                        output = data.copy()
                        output["sentiment"] = results
                        a, b, c = st.columns(3)
                        for col, label in zip((a, b, c), ("Positive", "Neutral", "Negative")):
                            col.metric(label, int((output["sentiment"] == label).sum()))
                        counts = output["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
                        st.bar_chart(counts, horizontal=True, x_label="Reviews", y_label="Sentiment")
                        st.dataframe(output, use_container_width=True, hide_index=True)
                        st.download_button("Download classified reviews", data=output.to_csv(index=False).encode("utf-8"),
                                           file_name="sentilens_results.csv", mime="text/csv", type="primary")
                        if ARTIFACTS is None:
                            st.warning("These CSV results were produced by demo rules, not the original trained classifier.")
        except Exception as exc:
            st.error(f"Unable to analyze CSV ({type(exc).__name__}). Verify that it is a valid UTF-8 CSV.")

with tab3:
    st.markdown("#### Project methodology")
    st.markdown("""
    The supplied notebook processes customer reviews by combining review title and body,
    cleaning text, removing stop words, and lemmatizing before converting reviews into
    TF-IDF features. It compares **Multinomial Naive Bayes**, **Logistic Regression**, and
    **Linear SVM**, then explores class balancing using **SMOTE** and hyperparameter tuning
    with **GridSearchCV**.

    **Notebook-recorded held-out test accuracy:** 78.82% for the tuned Logistic Regression
    model on a 288-row test split. This is a notebook output, **not independently verified
    by this deployed app**. It must not be attributed to the demo fallback.

    **Known limitations:** small original dataset (1,440 reviews), English-only design,
    uneven class frequencies, possible vocabulary drift, and weak handling of irony,
    negation, mixed feedback, and domain shifts. The notebook used SMOTE before
    cross-validation during tuning; reported cross-validation performance may therefore
    be optimistic due to data leakage across folds.

    **Deployment note:** the supplied files contained `Vector.pkl` but not the corresponding
    `NLP.pkl` trained estimator. Until `artifacts/NLP.pkl` is added to this repository,
    the app operates in an **explicitly labeled, rule-based demo mode**.
    """)
    st.caption("Code and preprocessing methods are documented in the repository README.")
