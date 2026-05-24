# Forensic Investigation Platform
## AI-Enhanced Keyword Search (Topic 1)

### Overview
This platform demonstrates how AI improves forensic keyword searching beyond naive
string matching.  Three AI techniques work together:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Semantic Encoding** | `all-MiniLM-L6-v2` (Sentence Transformers) | Maps text to 384-d dense vectors capturing *meaning*, not just exact words |
| **Query Expansion** | WordNet (NLTK) | Adds synonyms so "transfer" also matches "remittance", "wire", "conveyance" |
| **Active Learning** | Linear SVM (scikit-learn) | Re-ranks results using the investigator's 👍/👎 feedback |

### File Structure
```
.
├── app.py                  # Streamlit UI
├── semantic_search.py      # FAISS index + main search pipeline
├── feedback_manager.py     # SVM active learning loop
├── text_extractor.py       # PDF / DOCX / HTML / TXT extraction
├── text_preprocessor.py    # Text cleaning + lemmatisation
├── report_generator.py     # HTML + JSON report generation
├── requirements.txt        # Python dependencies
└── data/                   # Created at runtime (FAISS index, feedback log)
```

### Setup & Run

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501

### How the AI pipeline works

```
User Query
    │
    ▼
WordNet synonym expansion        ← broadens vocabulary recall
    │
    ▼
SentenceTransformer encoding    ← captures semantic meaning
    │
    ▼
FAISS cosine-similarity search  ← fast nearest-neighbour retrieval
    │
    ▼
SVM personalisation blend       ← reweights using feedback history
(0.7 × semantic + 0.3 × SVM)
    │
    ▼
Ranked results + preview
```

### Key design decisions

- **`pypdf` not `PyPDF2`** – PyPDF2 was deprecated; pypdf is the maintained fork.
- **Light preprocessing for embeddings** – Sentence-transformers are trained on
  natural text.  Removing stopwords or stemming *hurts* embedding quality.
  Heavy NLP normalisation is reserved for the TF-IDF vectorizer inside the SVM.
- **Vectorizer cached per (session, query)** – The original code re-fitted the
  TF-IDF vectorizer on every scoring call, causing vocabulary-dimension mismatches
  when `transform()` was called after a re-fit.  The fixed version caches the
  fitted (vectorizer, svm) pair and only retrains when new feedback arrives.
- **`ReportGenerator` is a proper class** – The original generated HTML inline
  inside `app.py` (duplicated logic).  The generator now owns all report logic
  and produces a dark-theme HTML file suitable for professional submission.
