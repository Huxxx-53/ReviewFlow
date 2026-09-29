"""
CS 315 - Application Development and Emerging Technologies
Activity 3: GenAI App on a New Dataset (TMDB Movie Reviews)

TMDB Movie Dataset Analysis + Dataset-Grounded Chatbot

- TMDB CSV is the source of truth for all stats, charts, and answers.
- Groq is used only on the server to phrase chatbot replies from retrieved data.
- The Groq API key lives in .env (GROQ_API_KEY) and is never shown to the browser.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Server-side secrets (never expose to the UI)
# ----------------------------------------------------------------------
load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#95a5a6",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "it", "was", "were", "this", "that", "with", "as", "at", "by",
    "be", "are", "i", "its", "so", "if", "than", "too", "very", "you",
    "your", "my", "me", "we", "our", "they", "them", "their", "he", "she",
    "his", "her", "not", "no", "just", "even", "all", "some", "more",
    "into", "from", "film", "films", "movie", "movies", "one", "two",
    "also", "have", "has", "had", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "about", "up", "out", "when", "what", "which", "who", "whom", "how",
    "why", "where", "there", "here", "then", "these", "those",
    "such", "only", "own", "same", "other", "over", "after", "before",
    "between", "through", "during", "above", "below", "again", "further",
    "once", "any", "both", "each", "few", "most", "s", "t", "don", "now",
    "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "like",
    "really", "get", "got", "much", "still", "well", "way", "back", "see",
    "seen", "watch", "watching", "time", "story", "character", "characters",
    "scene", "scenes", "while", "make", "made", "first", "also",
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movie_reviews.csv")

# ----------------------------------------------------------------------
# Page setup (must be the first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(page_title="ReviewFlow", page_icon="🌊", layout="wide")


def get_groq_api_key() -> str:
    """Load GROQ_API_KEY from environment / .env / Streamlit secrets only."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        return ""


GROQ_API_KEY = get_groq_api_key()

st.title("🌊 ReviewFlow")
st.caption(
    "TMDB Movie Dataset Analysis + Dataset-Grounded Chatbot · "
    "TMDB is the source of truth · Groq only helps phrase answers from retrieved data."
)


# ----------------------------------------------------------------------
# Data loading & enrichment
# ----------------------------------------------------------------------
def rating_to_sentiment(rating: int) -> str:
    if rating >= 7:
        return "Positive"
    if rating <= 4:
        return "Negative"
    return "Neutral"


def sentiment_badge(label: str) -> str:
    if label == "Positive":
        return "🟢 Positive"
    if label == "Negative":
        return "🔴 Negative"
    return "⚪ Neutral"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["review_text"], keep="first")
    df = df.dropna(subset=["review_text", "movie_title", "genre", "rating"])
    df["review_text"] = (
        df["review_text"].astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()
    )
    df = df[df["review_text"].str.len() >= 40]
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    for column in ("tmdb_popularity", "tmdb_vote_count"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["rating"].between(1, 10)]
    df["rating"] = df["rating"].round().astype(int)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)
    df["sentiment"] = df["rating"].map(rating_to_sentiment)
    df["sentiment_label"] = df["sentiment"].map(sentiment_badge)
    return df.reset_index(drop=True)


def extract_keywords(texts: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Frequency of meaningful words after stopword removal."""
    counts: Counter = Counter()
    for text in texts.dropna().astype(str):
        words = re.findall(r"[a-zA-Z']+", text.lower())
        for w in words:
            if len(w) <= 3 or w in STOPWORDS:
                continue
            counts[w] += 1
    rows = counts.most_common(top_n)
    return pd.DataFrame(rows, columns=["keyword", "frequency"])


def build_insights(df: pd.DataFrame, kw_df: pd.DataFrame) -> list[str]:
    """Compute Key Insights from the actual filtered dataset (not hard-coded)."""
    if df.empty:
        return ["No movies match the current filters, so no insights are available."]

    n = len(df)
    n_movies = int(df["movie_title"].nunique())
    avg_rating = df["rating"].mean()
    top_genres = df["genre"].value_counts().head(3)
    genre_txt = ", ".join(f"{g} ({c})" for g, c in top_genres.items())
    sent = df["sentiment"].value_counts()
    sent_txt = ", ".join(
        f"{sentiment_badge(s)}: {int(sent.get(s, 0))} ({sent.get(s, 0) / n * 100:.0f}%)"
        for s in ["Positive", "Neutral", "Negative"]
    )
    high = int((df["rating"] >= 8).sum())
    low = int((df["rating"] <= 3).sum())
    mid = n - high - low
    best = df.loc[df["rating"].idxmax()]
    worst = df.loc[df["rating"].idxmin()]
    year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())
    top_kw = ", ".join(
        f"{r.keyword} ({int(r.frequency)})" for r in kw_df.head(5).itertuples()
    ) if not kw_df.empty else "n/a"

    return [
        f"Analyzed **{n}** TMDB reviews across **{n_movies}** unique movies "
        f"(release years {year_min}–{year_max}).",
        f"Most common genres: **{genre_txt}**.",
        f"Average rating: **{avg_rating:.2f} / 10**.",
        f"Rating distribution: **{high}** high (8–10), **{mid}** mid (4–7), "
        f"**{low}** low (1–3).",
        f"Sentiment distribution — {sent_txt}.",
        f"Highest-rated in this view: **{best['movie_title']}** "
        f"({int(best['rating'])}/10). Lowest-rated: **{worst['movie_title']}** "
        f"({int(worst['rating'])}/10).",
        f"Most common keywords in descriptions: **{top_kw}**.",
    ]


df = load_data(DATA_PATH)

# ----------------------------------------------------------------------
# Sidebar: filters only (no API key UI)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Filters")
    genres = sorted(df["genre"].dropna().unique())
    selected_genres = st.multiselect("Genre", genres, default=genres)

    years = sorted(df["release_year"].dropna().unique())
    selected_years = st.multiselect("Release year", years, default=years)

    min_rating, max_rating = st.slider("Rating range", 1, 10, (1, 10))

    filtered_df = df[
        df["genre"].isin(selected_genres)
        & df["release_year"].isin(selected_years)
        & df["rating"].between(min_rating, max_rating)
    ].reset_index(drop=True)

    st.metric("Reviews matching filters", len(filtered_df))

    st.divider()
    st.header("🎛️ Sample display size")
    st.caption(
        "Full-dataset statistics stay accurate. This only controls how many "
        "random sample rows are shown in the analysis table."
    )
    max_possible = max(len(filtered_df), 1)
    default_sample = min(25, max_possible)
    sample_size = st.number_input(
        "Sample rows to preview",
        min_value=1,
        max_value=max_possible,
        value=default_sample,
        step=5,
    )

    st.divider()
    st.caption(
        "API keys are loaded server-side from `.env` (`GROQ_API_KEY`). "
        "They are never shown in this UI."
    )
    if GROQ_API_KEY:
        st.success("Groq backend: configured")
    else:
        st.warning(
            "Groq backend: not configured. Copy `.env.example` to `.env` "
            "and set GROQ_API_KEY to enable the chatbot."
        )


# ----------------------------------------------------------------------
# Groq client (backend only)
# ----------------------------------------------------------------------
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    except Exception as e:
        st.sidebar.error(f"Could not initialize Groq client: {e}")
        return None


# ----------------------------------------------------------------------
# Dataset query helpers for the chatbot
# ----------------------------------------------------------------------
UNRELATED_REPLY = "I only answer questions about the TMDB movie dataset in this app."

NOT_FOUND_REPLY = """### Answer
I couldn't find that information in the current TMDB dataset.

### Relevant Data
- Nothing in the filtered TMDB reviews matched this request.

### Short Insight
If a movie, year, or genre is missing here, it simply is not in the loaded dataset."""

MOVIE_QUESTION_HINTS = re.compile(
    r"\b(movie|movies|film|films|cinema|tmdb|genre|genres|rating|ratings|"
    r"review|reviews|release|year|highest|lowest|average|avg|best|worst|"
    r"liked|love[d]?|hate[d]?|favorite|favourite|popular|popularity|famous|fame|recommend|"
    r"watch|watched|watching|how many|top rated|top-rated|common|"
    r"keyword|theme|sentiment|positive|negative|neutral|dataset|"
    r"adventure|animation|romance|fantasy|mystery|crime|family|"
    r"history|music|horror|comedy|drama|action|thriller|sci-?fi|"
    r"science fiction|which one|tell me about|what about)\b",
    re.IGNORECASE,
)

OFF_TOPIC_HINTS = re.compile(
    r"\b(weather|forecast|temperature|recipe|cook|cooking|homework|"
    r"python code|javascript|programming|capital of|president of|"
    r"bitcoin|cryptocurrency|stock price|translate|write a poem|"
    r"who won the|soccer score|basketball score|math problem|"
    r"solve for x)\b",
    re.IGNORECASE,
)


def _movie_mentions(question: str, titles: list[str]) -> list[str]:
    q = re.sub(r"[^\w\s:'\-]", " ", question.lower())
    hits = []
    for title in titles:
        tl = title.lower()
        if len(tl) <= 2:
            if re.search(rf"\b{re.escape(tl)}\b", q):
                hits.append(title)
        elif tl in q:
            hits.append(title)
    hits.sort(key=len, reverse=True)
    return hits[:5]


def _classify_dataset_topic(question: str, titles: list[str]) -> bool | None:
    """Ask Groq only YES/NO: is this about movies in the loaded TMDB list?"""
    client = get_groq_client()
    if client is None:
        return None
    sample = ", ".join(titles[:80])
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You only classify the user's sentence. Reply with YES or NO. "
                    "YES = they are asking, in any wording, about movies, ratings, "
                    "genres, years, reviews, or this TMDB collection. "
                    "NO = anything else (weather, news, homework, cooking, trivia "
                    "not about these films), or a film that is clearly not in the list. "
                    "Known titles include: " + sample
                )},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=5,
        )
        text = (response.choices[0].message.content or "").strip().upper()
        if text.startswith("YES"):
            return True
        if text.startswith("NO"):
            return False
    except Exception:
        return None
    return None


def is_about_dataset(question: str, data: pd.DataFrame) -> bool:
    """Accept any sentence; return False only when it is not about this TMDB set."""
    q = question.strip()
    if not q:
        return False
    titles = list(data["movie_title"].dropna().unique()) if not data.empty else []
    if _movie_mentions(q, titles):
        return True
    if not data.empty:
        for genre in data["genre"].dropna().unique():
            if re.search(rf"\b{re.escape(str(genre))}\b", q, re.IGNORECASE):
                return True
    if MOVIE_QUESTION_HINTS.search(q):
        return True
    if OFF_TOPIC_HINTS.search(q):
        return False
    classified = _classify_dataset_topic(q, titles)
    if classified is not None:
        return classified
    return False


def _wants_reviews(question: str) -> bool:
    q = question.lower()
    return bool(re.search(r"\breviews?\b", q)) or any(
        phrase in q for phrase in ("what did people say", "review text", "snippets")
    )


def _majority_sentiment(series: pd.Series) -> str:
    counts = series.value_counts()
    if counts.empty:
        return "Neutral"
    return str(counts.index[0])


def unique_movies(df: pd.DataFrame) -> pd.DataFrame:
    """One row per movie title so the same film is not listed for every review."""
    if df.empty:
        return pd.DataFrame(
            columns=["movie_title", "genre", "release_year", "avg_rating",
                     "review_count", "sentiment"]
        )
    aggregations = {
        "genre": ("genre", "first"),
        "release_year": ("release_year", "first"),
        "avg_rating": ("rating", "mean"),
        "review_count": ("review_text", "size"),
        "sentiment": ("sentiment", _majority_sentiment),
    }
    for column in ("tmdb_popularity", "tmdb_vote_count"):
        if column in df.columns:
            aggregations[column] = (column, "max")
    grouped = (
        df.groupby("movie_title", as_index=False)
        .agg(**aggregations)
    )
    grouped["release_year"] = grouped["release_year"].astype(int)
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    return grouped


def _movies_to_table(
    movies: pd.DataFrame, limit: int = 10, show_popularity: bool = False
) -> str:
    if movies.empty:
        return "_No matching movies._"
    headers = ["Movie", "Year", "Genre"]
    separators = ["---"] * 3
    if show_popularity:
        headers.extend(["TMDB popularity", "TMDB votes"])
        separators.extend(["---:", "---:"])
    headers.extend(["Avg. rating", "Reviews", "Sentiment"])
    separators.extend(["---:", "---:", "---"])
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separators) + " |"]
    for _, r in movies.head(limit).iterrows():
        values = [str(r["movie_title"]), str(int(r["release_year"])), str(r["genre"])]
        if show_popularity:
            values.extend([
                f"{float(r['tmdb_popularity']):.2f}",
                str(int(r["tmdb_vote_count"])) if pd.notna(r.get("tmdb_vote_count")) else "n/a",
            ])
        values.extend([
            f"{float(r['avg_rating']):.1f} / 10",
            str(int(r["review_count"])),
            sentiment_badge(str(r["sentiment"])),
        ])
        rows.append("| " + " | ".join(values) + " |")
    extra = len(movies) - limit
    md = "\n".join(rows)
    if extra > 0:
        md += f"\n\n_Showing {limit} of {len(movies)} unique movies._"
    return md


def _sentiment_bullets(df: pd.DataFrame) -> list[str]:
    n = max(len(df), 1)
    bullets = []
    for label in ["Positive", "Neutral", "Negative"]:
        count = int((df["sentiment"] == label).sum())
        pct = count / n * 100
        bullets.append(f"- {sentiment_badge(label)}: **{count}** reviews ({pct:.0f}%)")
    return bullets


def _dataset_snapshot(data: pd.DataFrame) -> dict:
    sent = data["sentiment"].value_counts()
    genres = data.groupby("genre")["movie_title"].nunique().sort_values(ascending=False)
    return {
        "n_reviews": int(len(data)),
        "n_movies": int(data["movie_title"].nunique()),
        "avg_rating": float(data["rating"].mean()) if len(data) else 0.0,
        "positive": int(sent.get("Positive", 0)),
        "neutral": int(sent.get("Neutral", 0)),
        "negative": int(sent.get("Negative", 0)),
        "top_genres": [(g, int(c)) for g, c in genres.head(5).items()],
    }


def query_dataset(question: str, data: pd.DataFrame) -> dict:
    """Retrieve structured TMDB facts. Movies are unique unless reviews are requested."""
    q = question.strip()
    q_lower = q.lower()
    related = is_about_dataset(q, data)
    payload: dict = {
        "related": related,
        "not_found": False,
        "intent": "overview",
        "want_reviews": _wants_reviews(q),
        "movies": pd.DataFrame(),
        "reviews": pd.DataFrame(),
        "focus_genre": None,
        "focus_year": None,
        "mentioned_title": None,
        "keywords": pd.DataFrame(),
        "snapshot": {},
        "missing_reason": "",
    }

    if data.empty:
        payload["not_found"] = True
        payload["missing_reason"] = "The filtered dataset is empty."
        payload["related"] = True
        return payload

    payload["snapshot"] = _dataset_snapshot(data)
    titles = sorted(data["movie_title"].unique(), key=len, reverse=True)
    mentioned = _movie_mentions(q, titles)
    if mentioned:
        payload["related"] = True

    movies = unique_movies(data)
    asked_highest = any(
        w in q_lower for w in (
            "highest", "best", "top rated", "top-rated", "greatest",
            "liked the most", "love the most", "favorite", "favourite",
            "most popular", "stand out", "top pick", "worth watching",
        )
    )
    asked_lowest = any(
        w in q_lower for w in ("lowest", "worst", "lowest-rated", "lowest rated")
    )
    asked_avg = any(w in q_lower for w in ("average", "avg", "mean rating"))
    asked_sentiment = any(
        w in q_lower for w in ("sentiment", "positive", "negative", "neutral")
    )
    asked_keywords = any(
        w in q_lower for w in ("keyword", "theme", "common word")
    )
    asked_count = "how many" in q_lower and any(
        w in q_lower for w in ("movie", "review", "film")
    )
    asked_common_genre = any(
        p in q_lower for p in ("common genre", "most common genre", "popular genre")
    )
    asked_popularity = any(term in q_lower for term in ("famous", "fame", "popularity")) or (
        "popular" in q_lower and not asked_common_genre
    )
    asked_recommendation = any(
        phrase in q_lower
        for phrase in (
            "recommend", "recommendation", "suggest", "suggestion",
            "what should i watch", "what should we watch", "pick a movie",
        )
    )

    genre_counts = data.groupby("genre")["movie_title"].nunique().sort_values(ascending=False)
    matched_genres = [g for g in genre_counts.index if g.lower() in q_lower]

    year_match = re.search(r"\b(\d{4})\b", q)
    year_question = (
        "released in" in q_lower
        or "release year" in q_lower
        or ("year" in q_lower and year_match is not None)
        or (year_match is not None and payload["related"])
    )

    if asked_popularity:
        popularity_data = data
        if matched_genres:
            payload["focus_genre"] = matched_genres[0]
            popularity_data = popularity_data[
                popularity_data["genre"] == payload["focus_genre"]
            ]
        if year_question and year_match:
            payload["focus_year"] = int(year_match.group(1))
            popularity_data = popularity_data[
                popularity_data["release_year"] == payload["focus_year"]
            ]
        payload["related"] = True
        if "tmdb_popularity" not in popularity_data.columns:
            payload["not_found"] = True
            payload["missing_reason"] = (
                "TMDB popularity is not present in this CSV yet. Re-fetch the dataset "
                "with fetch_real_dataset.py, then run clean_dataset.py."
            )
        else:
            popularity_movies = unique_movies(popularity_data).dropna(
                subset=["tmdb_popularity"]
            )
            if popularity_movies.empty:
                payload["not_found"] = True
                payload["missing_reason"] = (
                    "No TMDB popularity values are available for the matching movies."
                )
            else:
                sort_columns = ["tmdb_popularity"]
                if "tmdb_vote_count" in popularity_movies.columns:
                    sort_columns.append("tmdb_vote_count")
                payload["intent"] = "popularity"
                payload["movies"] = popularity_movies.sort_values(
                    sort_columns, ascending=[False] * len(sort_columns)
                ).head(8)
                payload["snapshot"] = _dataset_snapshot(popularity_data)
    elif asked_recommendation:
        recommendation_data = data
        if matched_genres:
            payload["focus_genre"] = matched_genres[0]
            recommendation_data = recommendation_data[
                recommendation_data["genre"] == payload["focus_genre"]
            ]
        if year_question and year_match:
            payload["focus_year"] = int(year_match.group(1))
            recommendation_data = recommendation_data[
                recommendation_data["release_year"] == payload["focus_year"]
            ]
        payload["related"] = True
        recommendation_movies = unique_movies(recommendation_data)
        if recommendation_movies.empty:
            payload["not_found"] = True
            payload["missing_reason"] = "No movies match those recommendation filters."
        else:
            established = recommendation_movies[recommendation_movies["review_count"] >= 5]
            if not established.empty:
                recommendation_movies = established
            payload["intent"] = "recommendation"
            payload["movies"] = recommendation_movies.sort_values(
                ["avg_rating", "review_count", "movie_title"],
                ascending=[False, False, True],
            ).head(3)
            payload["snapshot"] = _dataset_snapshot(recommendation_data)
    elif asked_highest:
        payload["intent"] = "highest"
        payload["movies"] = movies.sort_values(
            ["avg_rating", "review_count", "movie_title"],
            ascending=[False, False, True],
        ).head(8)
    elif asked_lowest:
        payload["intent"] = "lowest"
        payload["movies"] = movies.sort_values(
            ["avg_rating", "review_count", "movie_title"],
            ascending=[True, False, True],
        ).head(8)
    elif matched_genres:
        payload["intent"] = "genre"
        payload["focus_genre"] = matched_genres[0]
        payload["related"] = True
        genre_df = data[data["genre"] == payload["focus_genre"]]
        payload["movies"] = unique_movies(genre_df).sort_values(
            "avg_rating", ascending=False
        )
        payload["snapshot"] = _dataset_snapshot(genre_df)
        payload["snapshot"]["parent_reviews"] = int(len(data))
        payload["snapshot"]["parent_movies"] = int(data["movie_title"].nunique())
    elif year_question and year_match:
        year = int(year_match.group(1))
        payload["intent"] = "year"
        payload["focus_year"] = year
        payload["related"] = True
        year_df = data[data["release_year"] == year]
        if year_df.empty:
            payload["not_found"] = True
            payload["missing_reason"] = (
                f"No movies in this dataset have a release year of {year}."
            )
        else:
            payload["movies"] = unique_movies(year_df).sort_values("movie_title")
            payload["snapshot"] = _dataset_snapshot(year_df)
    elif mentioned:
        payload["intent"] = "movie"
        payload["mentioned_title"] = mentioned[0]
        sub = data[data["movie_title"] == mentioned[0]]
        payload["movies"] = unique_movies(sub)
        payload["snapshot"] = _dataset_snapshot(sub)
        if payload["want_reviews"]:
            payload["reviews"] = sub.sort_values("rating", ascending=False).head(4)
    elif asked_avg:
        payload["intent"] = "average"
        payload["movies"] = movies.sort_values("avg_rating", ascending=False).head(5)
    elif asked_sentiment:
        payload["intent"] = "sentiment"
        payload["movies"] = movies.sort_values("avg_rating", ascending=False).head(5)
    elif asked_keywords:
        payload["intent"] = "keywords"
        payload["keywords"] = extract_keywords(data["review_text"], top_n=8)
    elif asked_common_genre or asked_count:
        payload["intent"] = "overview"
        payload["movies"] = movies.sort_values("review_count", ascending=False).head(5)
    else:
        payload["intent"] = "overview"
        payload["movies"] = movies.sort_values("avg_rating", ascending=False).head(5)

    if payload["want_reviews"] and payload["reviews"].empty and not payload["not_found"]:
        source = data
        if payload.get("focus_genre"):
            source = data[data["genre"] == payload["focus_genre"]]
        elif payload.get("focus_year") is not None:
            source = data[data["release_year"] == payload["focus_year"]]
        payload["reviews"] = source.sort_values("rating", ascending=False).head(3)

    if not payload["related"]:
        return {"related": False, "not_found": False}

    return payload


def _local_answer_and_insight(question: str, payload: dict, data: pd.DataFrame) -> tuple[str, str]:
    snap = payload.get("snapshot") or _dataset_snapshot(data)
    movies: pd.DataFrame = payload.get("movies", pd.DataFrame())
    intent = payload.get("intent", "overview")
    n_movies = snap.get("n_movies", 0)
    n_reviews = snap.get("n_reviews", 0)
    avg = snap.get("avg_rating", 0)

    if intent == "popularity" and not movies.empty:
        top = movies.iloc[0]
        scope = f"the {payload['focus_genre']} movies" if payload.get("focus_genre") else "movies"
        answer = (
            f"By TMDB's popularity score, **{top['movie_title']}** ranks highest among "
            f"{scope} in this dataset, with a score of **{top['tmdb_popularity']:.2f}**."
        )
        insight = (
            "TMDB popularity is a changing platform metric, not a definitive measure of "
            "overall fame."
        )
    elif intent == "recommendation" and not movies.empty:
        top = movies.iloc[0]
        scope = f" in {payload['focus_genre']}" if payload.get("focus_genre") else ""
        answer = (
            f"I recommend **{top['movie_title']}**{scope}. It averages "
            f"**{top['avg_rating']:.1f} / 10** from **{int(top['review_count'])}** "
            "reviews in this dataset."
        )
        insight = (
            "Recommendations are ranked by average TMDB review rating, prioritizing "
            "movies with at least five collected reviews when available."
        )
    elif intent == "highest" and not movies.empty:
        top = movies.iloc[0]
        answer = (
            f"**{top['movie_title']}** is the highest-rated unique movie in this view, "
            f"with an average TMDB rating of **{top['avg_rating']:.1f} / 10** "
            f"({sentiment_badge(str(top['sentiment']))})."
        )
        insight = (
            "Ratings are averaged per movie so the same title is not listed once for every review."
        )
    elif intent == "lowest" and not movies.empty:
        low = movies.iloc[0]
        answer = (
            f"**{low['movie_title']}** is the lowest-rated unique movie in this view, "
            f"with an average TMDB rating of **{low['avg_rating']:.1f} / 10** "
            f"({sentiment_badge(str(low['sentiment']))})."
        )
        insight = "A low average can still come from only a few reviews — check the review count in the table."
    elif intent == "genre" and payload.get("focus_genre"):
        g = payload["focus_genre"]
        answer = (
            f"There are **{n_movies} unique {g} movies** in this dataset "
            f"({n_reviews} reviews). Their average rating is **{avg:.2f} / 10**."
        )
        insight = f"{g} is counted from unique titles, not from repeating the same movie in every review."
    elif intent == "year" and payload.get("focus_year") is not None:
        year = payload["focus_year"]
        answer = (
            f"**{n_movies} unique movies** in this dataset were released in **{year}**, "
            f"based on **{n_reviews}** TMDB reviews."
        )
        insight = f"Only films that appear in the loaded TMDB reviews for {year} are listed."
    elif intent == "movie" and payload.get("mentioned_title"):
        title = payload["mentioned_title"]
        answer = (
            f"**{title}** appears in this dataset with **{n_reviews}** review(s) "
            f"and an average rating of **{avg:.2f} / 10**."
        )
        insight = (
            "Review snippets are shown only when you ask about reviews; otherwise the movie is listed once."
        )
    elif intent == "average":
        answer = (
            f"The average TMDB rating across **{n_reviews}** reviews "
            f"({n_movies} unique movies) is **{avg:.2f} / 10**."
        )
        insight = "This average uses every matching review, not a sample."
    elif intent == "sentiment":
        answer = (
            f"Out of **{n_reviews}** reviews, **{snap['positive']}** are "
            f"{sentiment_badge('Positive')}, **{snap['neutral']}** are "
            f"{sentiment_badge('Neutral')}, and **{snap['negative']}** are "
            f"{sentiment_badge('Negative')}."
        )
        insight = "Sentiment comes from the TMDB rating: 7–10 positive, 5–6 neutral, 1–4 negative."
    elif intent == "keywords":
        kw = payload.get("keywords", pd.DataFrame())
        if kw.empty:
            answer = "No frequent keywords could be extracted from the current descriptions."
        else:
            top = ", ".join(f"**{r.keyword}** ({int(r.frequency)})" for r in kw.head(5).itertuples())
            answer = f"The most common themes in the movie descriptions are {top}."
        insight = "Keywords are frequent meaningful words after common stopwords are removed."
    else:
        genres = snap.get("top_genres") or []
        genre_txt = ", ".join(f"**{g}** ({c} movies)" for g, c in genres[:3]) or "n/a"
        answer = (
            f"This TMDB view contains **{n_movies} unique movies** and **{n_reviews}** reviews, "
            f"with an average rating of **{avg:.2f} / 10**. Most common genres: {genre_txt}."
        )
        insight = "Counts of movies use unique titles so popular films are not counted more than once."
    return answer, insight


def _relevant_data_markdown(payload: dict, data: pd.DataFrame) -> str:
    parts: list[str] = []
    snap = payload.get("snapshot") or {}
    movies: pd.DataFrame = payload.get("movies", pd.DataFrame())
    intent = payload.get("intent")

    if intent == "recommendation":
        parts.append("- Ranking: average TMDB review rating, with review count used as a confidence signal")
        if payload.get("focus_genre"):
            parts.append(f"- Genre: **{payload['focus_genre']}**")
        if payload.get("focus_year") is not None:
            parts.append(f"- Release year: **{payload['focus_year']}**")
        parts.append(f"- Matching movies: **{snap.get('n_movies', 0)}**")
    elif intent == "popularity":
        parts.append("- Ranking metric: TMDB popularity score (not a direct measure of global fame)")
        if payload.get("focus_genre"):
            parts.append(f"- Genre: **{payload['focus_genre']}**")
        if payload.get("focus_year") is not None:
            parts.append(f"- Release year: **{payload['focus_year']}**")
        parts.append(f"- Matching movies: **{snap.get('n_movies', 0)}**")
    elif intent == "genre" and payload.get("focus_genre"):
        g = payload["focus_genre"]
        parts.append(
            f"- Unique **{g}** movies: **{snap.get('n_movies', 0)}**"
        )
        parts.append(f"- {g} reviews: **{snap.get('n_reviews', 0)}**")
        parts.append(f"- Average {g} rating: **{snap.get('avg_rating', 0):.2f} / 10**")
        if snap.get("parent_movies"):
            parts.append(
                f"- Full filtered dataset: {snap['parent_movies']} unique movies "
                f"({snap.get('parent_reviews', 0)} reviews)"
            )
    elif intent in {"average", "overview", "sentiment", "highest", "lowest", "keywords"}:
        parts.append(f"- Unique movies: **{snap.get('n_movies', data['movie_title'].nunique())}**")
        parts.append(f"- Reviews: **{snap.get('n_reviews', len(data))}**")
        parts.append(f"- Average rating: **{snap.get('avg_rating', 0):.2f} / 10**")
        if intent in {"overview", "sentiment"}:
            parts.extend(_sentiment_bullets(data))
        if intent == "overview" and snap.get("top_genres"):
            parts.append("- Most common genres (unique movies):")
            for g, c in snap["top_genres"][:5]:
                parts.append(f"  - **{g}**: {c}")
    elif intent == "year":
        parts.append(f"- Release year: **{payload.get('focus_year')}**")
        parts.append(f"- Unique movies that year: **{snap.get('n_movies', 0)}**")
        parts.append(f"- Reviews that year: **{snap.get('n_reviews', 0)}**")
    elif intent == "movie":
        title = payload.get("mentioned_title")
        movie_rows = data[data["movie_title"] == title] if title else data.iloc[0:0]
        parts.append(f"- Title: **{title}**")
        parts.append(f"- Reviews found: **{snap.get('n_reviews', 0)}**")
        parts.append(f"- Average rating: **{snap.get('avg_rating', 0):.2f} / 10**")
        parts.extend(_sentiment_bullets(movie_rows))

    if intent == "keywords":
        kw = payload.get("keywords", pd.DataFrame())
        if not kw.empty:
            parts.append("")
            parts.append("| Keyword | Times used |")
            parts.append("| --- | --- |")
            for r in kw.itertuples():
                parts.append(f"| {r.keyword} | {int(r.frequency)} |")

    if intent != "keywords" and not movies.empty:
        heading = "Unique movies"
        if intent == "recommendation":
            heading = "Top recommendations"
        elif intent == "popularity":
            heading = "Movies ranked by TMDB popularity"
        elif intent == "highest":
            heading = "Highest-rated unique movies"
        elif intent == "lowest":
            heading = "Lowest-rated unique movies"
        parts.append("")
        parts.append(f"**{heading}**")
        parts.append(
            _movies_to_table(movies, limit=10, show_popularity=intent == "popularity")
        )

    reviews = payload.get("reviews", pd.DataFrame())
    if payload.get("want_reviews") and isinstance(reviews, pd.DataFrame) and not reviews.empty:
        parts.append("")
        parts.append("**Sample reviews** (shown because you asked about reviews)")
        for _, r in reviews.iterrows():
            snippet = str(r["review_text"])[:160].replace("\n", " ")
            parts.append(
                f"- {sentiment_badge(str(r['sentiment']))} · **{r['movie_title']}** "
                f"({int(r['rating'])}/10): _{snippet}…_"
            )

    return "\n".join(p for p in parts if p is not None)


def format_chat_reply(question: str, payload: dict, data: pd.DataFrame) -> str:
    """Student-friendly markdown: Answer → Relevant Data → Short Insight."""
    if payload.get("not_found"):
        reason = payload.get("missing_reason") or "Nothing matched this request."
        return (
            "I can't determine that from the current TMDB dataset.\n\n"
            "**What is missing**\n"
            f"- {reason}\n\n"
            "I can answer using only the movies and reviews available in this dataset."
        )

    answer, insight = _local_answer_and_insight(question, payload, data)
    relevant = _relevant_data_markdown(payload, data)
    groq_bits = _groq_polish_answer_insight(question, answer, insight, relevant)
    if groq_bits:
        answer, insight = groq_bits
    return (
        f"{answer.strip()}\n\n"
        f"**Dataset evidence**\n{relevant.strip()}\n\n"
        f"_{insight.strip()}_"
    )


def _groq_polish_answer_insight(
    question: str, answer: str, insight: str, relevant: str
) -> tuple[str, str] | None:
    """Optional: rewrite Answer + Insight in plain English. Layout stays local."""
    client = get_groq_client()
    if client is None:
        return None
    system = (
        "You are a conversational assistant answering questions about a TMDB movie dataset. "
        "Rewrite ONLY the Answer (1-3 natural, direct sentences) and Short Insight (1 concise sentence). "
        "Use ONLY the facts already given. Do not invent movies, years, or numbers. "
        "Answer the question directly, use a friendly conversational tone, and avoid canned phrasing. "
        "Preserve the scope and caveats of any metric; TMDB popularity is not global fame. "
        "Do not dump raw field names like Positive=661 or Genre counts:. "
        "Use 🟢 Positive, ⚪ Neutral, and 🔴 Negative when sentiment is mentioned. "
        "If facts are insufficient, reply with NOT_FOUND. "
        "Return exactly:\nANSWER:\n...\nINSIGHT:\n..."
    )
    user = (
        f"Question: {question}\n\n"
        f"Draft answer:\n{answer}\n\n"
        f"Draft insight:\n{insight}\n\n"
        f"Facts (do not copy as a dump; they are already shown to the student):\n{relevant}"
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        text = (response.choices[0].message.content or "").strip()
        if GROQ_API_KEY:
            text = text.replace(GROQ_API_KEY, "[REDACTED]")
        if "NOT_FOUND" in text and "ANSWER:" not in text:
            return None
        ans_m = re.search(r"ANSWER:\s*(.*?)\s*INSIGHT:", text, re.DOTALL | re.IGNORECASE)
        ins_m = re.search(r"INSIGHT:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
        if not ans_m or not ins_m:
            return None
        new_a, new_i = ans_m.group(1).strip(), ins_m.group(1).strip()
        if not new_a or not new_i:
            return None
        return new_a, new_i
    except Exception:
        return None


def answer_from_dataset(question: str, data: pd.DataFrame) -> str:
    """User Question → Query Dataset → Retrieve → formatted (Groq may polish wording)."""
    retrieved = query_dataset(question, data)

    if not retrieved.get("related"):
        return UNRELATED_REPLY

    if retrieved.get("not_found"):
        return format_chat_reply(question, retrieved, data)

    return format_chat_reply(question, retrieved, data)


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_data, tab_analysis, tab_chat = st.tabs(
    ["📋 Data", "📊 Analysis", "💬 Chatbot"]
)

# --- Tab: Data ----------------------------------------------------------
with tab_data:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total reviews (dataset)", len(df))
    m2.metric("Filtered reviews", len(filtered_df))
    m3.metric(
        "Unique movies",
        int(filtered_df["movie_title"].nunique()) if len(filtered_df) else 0,
    )
    m4.metric(
        "Avg. rating",
        f"{filtered_df['rating'].mean():.1f}" if len(filtered_df) else "—",
    )

    st.subheader("TMDB Review Data")
    st.caption(
        "Each **row is one TMDB user review**, not one movie. Popular films appear "
        "many times because several people reviewed them. The title, genre, and year "
        "can look the same; the reviewer, rating, and review text are different."
    )
    if filtered_df.empty:
        st.warning(
            "No reviews match your current filters — try widening the genre, "
            "year, or rating range in the sidebar."
        )
    else:
        view = st.radio(
            "Table view",
            ["Unique movies (one row per film)", "All reviews"],
            horizontal=True,
            help="Use unique movies if the repeated titles look like duplicates.",
        )
        if view.startswith("Unique"):
            movies = unique_movies(filtered_df).sort_values(
                "review_count", ascending=False
            )
            movies = movies.rename(columns={
                "movie_title": "movie_title",
                "release_year": "release_year",
                "avg_rating": "avg_rating",
                "review_count": "review_count",
                "sentiment": "sentiment",
            })
            movies["sentiment_label"] = movies["sentiment"].map(sentiment_badge)
            st.dataframe(
                movies[[
                    "movie_title", "genre", "release_year",
                    "avg_rating", "review_count", "sentiment_label",
                ]],
                width="stretch",
                height=420,
            )
        else:
            show_cols = [
                "movie_title", "genre", "release_year", "reviewer",
                "rating", "sentiment_label", "review_text",
            ]
            st.dataframe(filtered_df[show_cols], width="stretch", height=420)

# --- Tab: Analysis ------------------------------------------------------
with tab_analysis:
    st.subheader("Analyze the TMDB Dataset")
    st.caption(
        "Statistics use the **full filtered dataset**. A random sample is only "
        "used for the preview table so results are not identical every run."
    )

    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_button = st.button(
            "▶️ Analyze", type="primary", disabled=filtered_df.empty
        )
    with info_col:
        if filtered_df.empty:
            st.warning("Widen the sidebar filters before analyzing.")
        else:
            st.info(
                f"Will analyze all **{len(filtered_df)}** filtered reviews "
                f"(preview sample: **{min(sample_size, len(filtered_df))}** random rows)."
            )

    if run_button:
        steps = [
            "Loading Data",
            "Validating",
            "Analyzing",
            "Extracting Keywords",
            "Generating Insights",
            "Complete",
        ]
        status = st.empty()
        bar = st.progress(0)

        status.markdown(
            "**Loading Data** → Validating → Analyzing → Extracting Keywords → "
            "Generating Insights → Complete"
        )
        bar.progress(1 / len(steps), text="Loading Data...")
        working = filtered_df.copy()
        time.sleep(0.15)

        status.markdown(
            "Loading Data → **Validating** → Analyzing → Extracting Keywords → "
            "Generating Insights → Complete"
        )
        bar.progress(2 / len(steps), text="Validating...")
        working = working.dropna(subset=["movie_title", "review_text", "rating", "genre"])
        working = working[working["rating"].between(1, 10)]
        n_valid = len(working)
        time.sleep(0.15)

        status.markdown(
            "Loading Data → Validating → **Analyzing** → Extracting Keywords → "
            "Generating Insights → Complete"
        )
        bar.progress(3 / len(steps), text="Analyzing...")
        working["sentiment"] = working["rating"].map(rating_to_sentiment)
        working["sentiment_label"] = working["sentiment"].map(sentiment_badge)
        preview_n = min(int(sample_size), len(working))
        sample_preview = working.sample(n=preview_n, random_state=None).sort_values(
            "rating", ascending=False
        )
        time.sleep(0.15)

        status.markdown(
            "Loading Data → Validating → Analyzing → **Extracting Keywords** → "
            "Generating Insights → Complete"
        )
        bar.progress(4 / len(steps), text="Extracting Keywords...")
        kw_df = extract_keywords(working["review_text"], top_n=20)
        time.sleep(0.15)

        status.markdown(
            "Loading Data → Validating → Analyzing → Extracting Keywords → "
            "**Generating Insights** → Complete"
        )
        bar.progress(5 / len(steps), text="Generating Insights...")
        insights = build_insights(working, kw_df)
        time.sleep(0.15)

        status.markdown(
            "Loading Data → Validating → Analyzing → Extracting Keywords → "
            "Generating Insights → **Complete**"
        )
        bar.progress(1.0, text="Complete")

        st.session_state["analysis"] = {
            "df": working,
            "sample": sample_preview,
            "keywords": kw_df,
            "insights": insights,
            "n_valid": n_valid,
        }
        st.success(f"Analysis complete on {n_valid} validated TMDB reviews.")

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.caption("Click **Analyze** to run the dashboard.")
    else:
        adf: pd.DataFrame = analysis["df"]
        kw_df: pd.DataFrame = analysis["keywords"]
        sample_preview: pd.DataFrame = analysis["sample"]

        # ---- 1. Dataset Overview ----
        st.markdown("## 1. Dataset Overview")
        st.caption(
            "High-level counts for the filtered TMDB reviews. Useful for confirming "
            "how much data the rest of the charts are based on."
        )
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Reviews analyzed", len(adf))
        o2.metric("Unique movies", int(adf["movie_title"].nunique()))
        o3.metric("Genres", int(adf["genre"].nunique()))
        o4.metric("Avg rating", f"{adf['rating'].mean():.2f}")

        st.markdown("**Random sample preview** (changes each Analyze run)")
        st.dataframe(
            sample_preview[
                ["movie_title", "genre", "release_year", "rating",
                 "sentiment_label", "review_text"]
            ],
            width="stretch",
            height=260,
        )

        # ---- 2. Rating Analysis ----
        st.markdown("## 2. Rating Analysis")
        st.caption(
            "Shows how TMDB user ratings (1–10) are distributed. Useful for spotting "
            "whether the collection leans toward highly rated or poorly rated films."
        )
        rating_counts = adf["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig_rating = px.bar(
            rating_counts,
            x="rating",
            y="count",
            title="Rating Distribution (1–10)",
            labels={"rating": "Rating", "count": "Number of reviews"},
            text="count",
        )
        fig_rating.update_traces(marker_color="#3498db", textposition="outside")
        fig_rating.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig_rating, width="stretch")

        # ---- 3. Genre Analysis ----
        st.markdown("## 3. Genre Analysis")
        st.caption(
            "Compares how many reviews each genre contributes and their average rating. "
            "Useful for explaining which genres dominate the dataset."
        )
        g1, g2 = st.columns(2)
        with g1:
            genre_counts = adf["genre"].value_counts().reset_index()
            genre_counts.columns = ["genre", "count"]
            fig_g = px.bar(
                genre_counts.sort_values("count"),
                x="count",
                y="genre",
                orientation="h",
                title="Reviews per Genre",
                labels={"count": "Number of reviews", "genre": "Genre"},
            )
            fig_g.update_traces(marker_color="#1abc9c")
            st.plotly_chart(fig_g, width="stretch")
        with g2:
            genre_avg = (
                adf.groupby("genre", as_index=False)["rating"]
                .mean()
                .sort_values("rating")
            )
            fig_ga = px.bar(
                genre_avg,
                x="rating",
                y="genre",
                orientation="h",
                title="Average Rating by Genre",
                labels={"rating": "Average rating", "genre": "Genre"},
            )
            fig_ga.update_traces(marker_color="#9b59b6")
            st.plotly_chart(fig_ga, width="stretch")

        # ---- 4. Sentiment Analysis ----
        st.markdown("## 4. Sentiment Analysis")
        st.caption(
            "Sentiment is derived from the TMDB rating: "
            "🟢 Positive (7–10), ⚪ Neutral (5–6), 🔴 Negative (1–4). "
            "Labels are shown with color so the chart stays accessible."
        )
        s1, s2 = st.columns(2)
        with s1:
            sent_order = ["Positive", "Neutral", "Negative"]
            sent_counts = (
                adf["sentiment"]
                .value_counts()
                .reindex(sent_order, fill_value=0)
                .reset_index()
            )
            sent_counts.columns = ["sentiment", "count"]
            sent_counts["label"] = sent_counts["sentiment"].map(sentiment_badge)
            fig_s = px.bar(
                sent_counts,
                x="label",
                y="count",
                color="sentiment",
                title="Sentiment Distribution",
                labels={"label": "Sentiment", "count": "Number of reviews"},
                color_discrete_map=SENTIMENT_COLORS,
                text="count",
            )
            fig_s.update_layout(showlegend=False)
            fig_s.update_traces(textposition="outside")
            st.plotly_chart(fig_s, width="stretch")
        with s2:
            by_genre = (
                adf.groupby(["genre", "sentiment"]).size().reset_index(name="count")
            )
            by_genre["sentiment"] = pd.Categorical(
                by_genre["sentiment"], categories=sent_order, ordered=True
            )
            chart = (
                alt.Chart(by_genre)
                .mark_bar()
                .encode(
                    x=alt.X("genre:N", title="Genre"),
                    y=alt.Y("count:Q", title="Number of reviews"),
                    color=alt.Color(
                        "sentiment:N",
                        title="Sentiment",
                        scale=alt.Scale(
                            domain=sent_order,
                            range=[SENTIMENT_COLORS[s] for s in sent_order],
                        ),
                        legend=alt.Legend(
                            labelExpr=(
                                "{'Positive': '🟢 Positive', "
                                "'Neutral': '⚪ Neutral', "
                                "'Negative': '🔴 Negative'}[datum.value]"
                            )
                        ),
                    ),
                    tooltip=["genre", "sentiment", "count"],
                )
                .properties(title="Sentiment by Genre")
            )
            st.altair_chart(chart, width="stretch")

        # ---- 5. Keyword Extraction ----
        st.markdown("## 5. Keyword Extraction")
        st.info(
            "Keyword extraction identifies frequently occurring meaningful words "
            "in movie descriptions to reveal common themes across the dataset."
        )
        st.caption(
            "Common stopwords (the, a, is, …) are removed. The bar chart shows "
            "the most frequent remaining words — useful for spotting recurring themes."
        )
        if kw_df.empty:
            st.warning("No keywords could be extracted.")
        else:
            fig_kw = px.bar(
                kw_df.sort_values("frequency"),
                x="frequency",
                y="keyword",
                orientation="h",
                title="Top Keywords by Frequency",
                labels={"frequency": "Frequency", "keyword": "Keyword"},
                text="frequency",
            )
            fig_kw.update_traces(marker_color="#e67e22", textposition="outside")
            st.plotly_chart(fig_kw, width="stretch")
            st.dataframe(kw_df, width="stretch", height=280)

        # ---- 6. Key Insights ----
        st.markdown("## 6. Key Insights")
        st.caption(
            "Short findings calculated automatically from the current filtered dataset."
        )
        for line in analysis["insights"]:
            st.markdown(f"- {line}")

# --- Tab: Chatbot -------------------------------------------------------
with tab_chat:
    st.subheader("Dataset-Grounded Chatbot")
    st.caption(
        "Type any sentence. If it is about movies in this TMDB dataset, you will get a "
        "dataset-grounded answer. If it is not, the chatbot replies: "
        '*"I only answer questions about the TMDB movie dataset in this app."*'
    )
    st.info(
        "Each reply is formatted as **Answer → Relevant Data → Short Insight**. "
        "Movies are grouped by unique title (duplicates from multiple reviews are not listed "
        "unless you ask about reviews). Groq may polish the wording; all facts come from TMDB."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    user_question = st.chat_input("Ask anything — answers are only about this TMDB movie dataset...")

    if user_question:
        st.session_state["chat_history"].append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🤖 _Searching the TMDB dataset..._")
            answer = answer_from_dataset(user_question, filtered_df)
            placeholder.markdown(answer)
        st.session_state["chat_history"].append(("assistant", answer))

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ About this dataset"):
    st.markdown(
        f"""
This product uses the **TMDB API** but is not endorsed or certified by TMDB.

The app loads **{len(df)}** validated TMDB review records across
**{df['movie_title'].nunique()}** real movies from `data/movie_reviews.csv`.

- Re-fetch: `python fetch_real_dataset.py` (needs `TMDB_API_KEY`)
- Re-clean: `python clean_dataset.py`
- Groq key: set `GROQ_API_KEY` in a local `.env` file (see `.env.example`).
  On Streamlit Community Cloud, add the same name under **Settings → Secrets**.

Sentiment is derived from TMDB ratings (Positive 7–10, Neutral 5–6, Negative 1–4).
Keyword extraction uses word frequency after stopword removal.
Groq is used only to phrase chatbot answers from retrieved dataset facts.
        """
    )
