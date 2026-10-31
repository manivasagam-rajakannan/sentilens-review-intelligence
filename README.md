# sentilens-review-intelligence
# SentiLens — Customer Review Sentiment Intelligence

**A Streamlit dashboard for reviewing sentiment in customer feedback.** Built from a 1,440-review NLP notebook using TF-IDF, classification models, class-imbalance techniques, and hyperparameter tuning.

**Project author:** Manivasagam Rajakannan  
**Stack:** Python · Streamlit · NLP / NLTK · TF-IDF · scikit-learn · pandas

## What it does

- Classifies individual English-language product reviews as **Positive**, **Neutral**, or **Negative**.
- Supports sample reviews, explanation of matched words in demo mode, and **batch CSV analysis** up to 1,000 rows.
- Displays aggregate sentiment counts and provides downloadable results.
- Loads the original tuned Logistic Regression model when the matching `artifacts/NLP.pkl` is present.
- Clearly labels a **rule-based demonstration** if the original model is not provided.

> **Important — missing original model:** The supplied project includes `Vector.pkl` but **not** `NLP.pkl`. The app runs in labeled demo mode until the notebook-trained model file is added. Demo predictions are NOT validated ML predictions and do NOT inherit the notebook's reported performance.

## Notebook methodology

1. Load the review dataset (1,440 rows) and combine review title and body.
2. Lowercase, tokenize, remove stop words, strip punctuation/digits, and verb-lemmatize.
3. Derive three sentiment classes from 1–5 star ratings (1–2 Negative, 3 Neutral, 4–5 Positive).
4. Split train/test (80/20), fit TF-IDF, and compare Multinomial Naive Bayes, Logistic Regression, and linear SVM.
5. Apply SMOTE to the training features, then tune Logistic Regression using GridSearchCV.
6. Export trained model and vectorizer to `.pkl` files.

### Historical notebook results (not a new evaluation)

| Notebook model | Held-out test accuracy |
|---|---:|
| Multinomial Naive Bayes | 71.53% |
| Logistic Regression | 79.17% |
| Linear SVM | 79.17% |
| Logistic Regression + SMOTE | 78.47% |
| Tuned Logistic Regression + SMOTE | **78.82%** |

The test set contains 288 rows. The notebook recorded **0.9086 CV weighted-F1**, but SMOTE was performed before cross-validation, which can inflate the validation metric. The held-out score should not be confused with the demo mode's accuracy (unmeasured).

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To enable the original classifier, copy the **trusted** notebook-generated model to `artifacts/NLP.pkl`; the matching `Vector.pkl` is already included. The app then switches to model mode. Verify that your model and vectorizer were created together and that dependency versions are compatible. Never load pickle files from unknown sources.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload all the files **without changing this folder structure**.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) using GitHub and select **Create app**.
3. Choose the repository, branch `main`, and main file **`streamlit_app.py`**.
4. Choose Python 3.12 if needed, and click **Deploy**.
5. The app provides a real `.streamlit.app` URL only after successful deployment.

See [Streamlit's official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Input data format

A CSV file must contain a header named `review`. Example: [`examples/reviews.csv`](examples/reviews.csv). Maximum 1,000 rows and 2 MB per upload. Reviews are processed in the Streamlit server session; avoid uploading sensitive personal information.

## Project structure

```text
sentilens-review-intelligence/
├── streamlit_app.py            # Streamlit app entrypoint
├── sentiment_engine.py         # Original inference + rules-only demo fallback
├── artifacts/Vector.pkl        # Uploaded vectorizer (not usable alone for ML inference)
├── notebooks/                  # Original notebook and original Streamlit script
├── examples/reviews.csv        # Test input for batch mode
├── tests/                      # Unit tests for fallback and error handling
├── requirements.txt            # App dependencies
├── requirements-train.txt      # Extra training dependencies
└── .streamlit/config.toml      # Theme
```

## Constraints and responsible use

- The original training dataset and trained estimator are not included in the supplied upload; this repository cannot independently retrain or reproduce all historical results without them.
- Inference supports English text. Keywords alone cannot reliably interpret nuanced language, sarcasm, or negation.
- Do not use this demonstration as a basis for high-stakes employment, healthcare, financial, or eligibility decisions.
- Do not represent fallback-demo output as a validated machine-learning inference.

## Future enhancements

Rebuild with a complete review dataset, split data before resampling, use an `imblearn.Pipeline` within cross-validation, calibrate class probabilities, test performance on new domains, and add confidence monitoring and feedback collection.
