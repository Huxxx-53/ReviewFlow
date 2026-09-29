# ReviewFlow

CS 315 – Application Development and Emerging Technologies · Activity 3

**TMDB Movie Dataset Analysis + Dataset-Grounded Chatbot**

- **TMDB** is the source of truth for all stats, charts, and answers.
- **Groq** is used only on the server to phrase chatbot replies from retrieved dataset facts.
- The Groq API key stays in a server-side `.env` file — users never paste a key in the UI.

## Project structure

```
genai-movie-app/
├── app.py                 # Streamlit app (analysis + chatbot)
├── clean_dataset.py       # Validate / dedupe the TMDB CSV
├── fetch_real_dataset.py  # Pull real reviews from TMDB
├── requirements.txt
├── .env.example           # Template for server-side secrets
├── .gitignore
├── README.md
└── data/
    └── movie_reviews.csv  # Validated TMDB reviews (≥ 520 rows)
```

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
```

Edit `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq key at https://console.groq.com/keys (no credit card).

Optional (only to re-fetch TMDB data):

```
TMDB_API_KEY=your_tmdb_api_key_here
```

## 2. Run locally

```bash
streamlit run app.py
```

No API key fields appear in the browser. The backend reads `GROQ_API_KEY` from `.env`
(or from Streamlit Cloud secrets).

## 3. What it does

Three tabs:

1. **📋 Data** — metrics and the filtered TMDB review table (genre / year / rating filters).
2. **📊 Analysis** — click **Analyze** to run:
   - Progress: Loading Data → Validating → Analyzing → Extracting Keywords → Generating Insights → Complete
   - Sections: Dataset Overview, Rating Analysis, Genre Analysis, Sentiment Analysis,
     Keyword Extraction, Key Insights
   - Full-dataset statistics stay accurate; a random sample is only used for the preview table.
3. **💬 Chatbot** — dataset-only Q&A:
   - **User Question → Search/Query Dataset → Retrieve Relevant Data → Groq Generates Answer**
   - No general knowledge or web search. If data is missing:
     *"I couldn't find that information in the current TMDB dataset."*

Sentiment labels (with color + text):

- 🟢 Positive (rating 7–10)
- ⚪ Neutral (rating 5–6)
- 🔴 Negative (rating 1–4)

## 4. Dataset (TMDB)

The shipped CSV is real TMDB user reviews. To refresh or expand it:

```bash
# set TMDB_API_KEY in .env first
python fetch_real_dataset.py
python clean_dataset.py
```

`clean_dataset.py` removes duplicates / empty / invalid rows and keeps only legitimate
TMDB records (target: at least 520 rows — never fabricates data).

The fetcher also stores each movie's TMDB popularity score and vote count. The chatbot
can rank a genre's movies by TMDB popularity, while noting this is a changing platform
metric rather than a definitive measure of global fame. Existing CSVs need to be
re-fetched and cleaned before this metric is available in the app.

**Required attribution:** *"This product uses the TMDB API but is not endorsed or
certified by TMDB."*
https://www.themoviedb.org/documentation/api/terms-of-use

## 5. Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository (`.env` is gitignored — do not commit keys).
2. Go to https://share.streamlit.io → **New app** → point at `app.py`.
3. Under **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy.

## 6. Data flow

```
Browser (Streamlit UI)
   │  never sees GROQ_API_KEY
   ▼
app.py (server)
   ├── loads data/movie_reviews.csv  (TMDB source of truth)
   ├── Analysis: pandas stats + keyword frequency + charts
   └── Chatbot: query dataframe → evidence facts → Groq (server-side) → answer
```
