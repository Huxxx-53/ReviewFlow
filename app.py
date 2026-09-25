"""
CS 315 - Application Development and Emerging Technologies
Activity 3: GenAI App on a New Dataset (Movie Reviews)

A Streamlit app that:
  1. Loads and cleans a movie reviews dataset with Pandas.
  2. Uses a GenAI API (Groq's free models, or OpenAI GPT) to classify
     review sentiment and extract keywords/topics from each review.
  3. Lets the user filter by genre / year / rating.
  4. Visualizes results with Plotly and Altair.
  5. Includes a simple chatbot (with a "thinking" indicator) that answers
     questions about the dataset.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this folder to a public GitHub repo, then create a new app on
    https://share.streamlit.io pointing at app.py. Add your Groq key
    as a secret named GROQ_API_KEY (or OPENAI_API_KEY for OpenAI) under
    Settings -> Secrets.
"""

import os
import re
import json
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "it", "was", "were", "this", "that", "with", "as", "at", "by",
    "be", "are", "i", "its", "so", "if", "than", "too", "very", "you",
    "my", "me", "we", "not", "no", "just", "even", "all", "some", "more",
    "into", "from", "film", "movie", "one",
}

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="ReviewFlow", page_icon="🌊", layout="wide")
st.title("🌊 ReviewFlow")
st.caption(
    "Activity 3 · Pandas + Streamlit + a GenAI API (Groq or OpenAI) for sentiment "
    "analysis, keyword extraction, visualization, and a dataset chatbot."
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movie_reviews.csv")


# ----------------------------------------------------------------------
# 1. Load and clean the dataset
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["review_text", "movie_title"])
    df["review_text"] = df["review_text"].str.strip()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    return df


df = load_data(DATA_PATH)

# ----------------------------------------------------------------------
# Sidebar: API key + filters
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox(
        "GenAI provider",
        ["Groq (free)", "OpenAI (paid)"],
        help="Groq offers a free API tier — no credit card, no charges. "
             "Use it if you don't have OpenAI credits.",
    )
    if provider.startswith("Groq"):
        api_key = st.text_input(
            "Groq API key", type="password",
            help="Get a free key at https://console.groq.com/keys. "
                 "On Streamlit Cloud, set this as a secret named GROQ_API_KEY instead.",
            value=os.environ.get("GROQ_API_KEY", ""),
        )
        GENAI_MODEL = "openai/gpt-oss-120b"
        GENAI_BASE_URL = "https://api.groq.com/openai/v1"
    else:
        api_key = st.text_input(
            "OpenAI API key", type="password",
            help="Needed for GenAI sentiment analysis and the chatbot. "
                 "On Streamlit Cloud, set this as a secret named OPENAI_API_KEY instead.",
            value=os.environ.get("OPENAI_API_KEY", ""),
        )
        GENAI_MODEL = "gpt-4o-mini"
        GENAI_BASE_URL = None

    st.divider()
    st.header("🔍 Filters")
    genres = sorted(df["genre"].unique())
    selected_genres = st.multiselect("Genre", genres, default=genres)

    years = sorted(df["release_year"].unique())
    selected_years = st.multiselect("Release year", years, default=years)

    min_rating, max_rating = st.slider("Rating range", 0, 10, (0, 10))

    filtered_df = df[
        df["genre"].isin(selected_genres)
        & df["release_year"].isin(selected_years)
        & df["rating"].between(min_rating, max_rating)
    ].reset_index(drop=True)

    st.metric("Reviews matching filters", len(filtered_df))

    st.divider()
    st.header("🎛️ Analysis sample size")
    st.caption(
        "Free API tiers are rate-limited, so pick how many of the "
        "filtered reviews to actually send to the GenAI model."
    )
    max_possible = max(len(filtered_df), 1)
    default_sample = min(30, max_possible)
    sample_size = st.number_input(
        "Reviews to analyze", min_value=1, max_value=max_possible,
        value=default_sample, step=5,
    )

# ----------------------------------------------------------------------
# 2. GenAI integration: combined sentiment + keyword extraction
# ----------------------------------------------------------------------
def get_openai_client():
    if not api_key:
        return None
    try:
        from openai import OpenAI
        if GENAI_BASE_URL:
            return OpenAI(api_key=api_key, base_url=GENAI_BASE_URL)
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Could not initialize GenAI client: {e}")
        return None


def genai_analyze(client, text: str) -> dict:
    """Single GenAI call: returns {'sentiment': ..., 'keywords': [...]}."""
    response = client.chat.completions.create(
        model=GENAI_MODEL,
        messages=[
            {"role": "system", "content": (
                "You analyze a movie review. Reply with ONLY valid JSON, no "
                "other text, in exactly this shape: "
                '{"sentiment": "Positive|Neutral|Negative", '
                '"keywords": ["kw1", "kw2", "kw3"]}. '
                "Keywords should be 3 to 5 short phrases describing what the "
                "review is actually about (e.g. \"slow pacing\", \"great acting\")."
            )},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(raw)
    sentiment = str(data.get("sentiment", "Neutral")).strip().title()
    keywords = data.get("keywords", [])
    if isinstance(keywords, list):
        keywords = ", ".join(str(k).strip() for k in keywords if str(k).strip())
    return {"sentiment": sentiment, "keywords": keywords}


def fallback_analyze(text: str, rating: int) -> dict:
    """Offline fallback so the app still works without an API key or on failure."""
    if rating >= 7:
        sentiment = "Positive"
    elif rating <= 4:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    common = [w for w, _ in Counter(words).most_common(4)]
    keywords = ", ".join(common) if common else "n/a"
    return {"sentiment": sentiment, "keywords": keywords}


def build_chat_context(context_df, max_chars: int = 9000, text_len: int = 180) -> str:
    """Build a CSV context for the chatbot that stays within free-tier token
    limits: truncates each review's text and stops adding rows once the
    character budget is hit (roughly max_chars // 4 tokens)."""
    cols = [c for c in ["movie_title", "genre", "rating", "sentiment", "keywords", "review_text"]
            if c in context_df.columns]
    trimmed = context_df[cols].copy()
    if "review_text" in trimmed.columns:
        trimmed["review_text"] = trimmed["review_text"].astype(str).str.slice(0, text_len)

    lines = [",".join(cols)]
    total_chars = len(lines[0])
    for _, row in trimmed.iterrows():
        line = ",".join(str(row[c]).replace(",", ";").replace("\n", " ") for c in cols)
        if total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Tabbed layout
# ----------------------------------------------------------------------
tab_data, tab_analysis, tab_charts, tab_chat = st.tabs(
    ["📋 Data", "🤖 Run Analysis", "📈 Charts", "💬 Chatbot"]
)

# --- Tab: Data ----------------------------------------------------------
with tab_data:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total reviews (dataset)", len(df))
    m2.metric("Filtered reviews", len(filtered_df))
    m3.metric("Unique movies", filtered_df["movie_title"].nunique())
    m4.metric("Avg. rating", f"{filtered_df['rating'].mean():.1f}" if len(filtered_df) else "—")

    st.subheader("Review Data")
    analyzed_df_state = st.session_state.get("analyzed_df")
    if analyzed_df_state is not None and not analyzed_df_state.empty:
        display_df = analyzed_df_state
    else:
        display_df = filtered_df
    extra_cols = [c for c in ["sentiment", "keywords"] if c in display_df.columns]
    if display_df.empty:
        st.warning("No reviews match your current filters — try widening the genre, "
                   "year, or rating range in the sidebar.")
    else:
        st.dataframe(
            display_df[["movie_title", "genre", "release_year", "reviewer", "rating"]
                       + extra_cols + ["review_text"]],
            width='stretch', height=420,
        )

# --- Tab: Run Analysis ---------------------------------------------------
with tab_analysis:
    st.subheader("Run Sentiment Analysis & Keyword Extraction")
    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_button = st.button("▶️ Analyze reviews", type="primary", disabled=filtered_df.empty)
    with info_col:
        if filtered_df.empty:
            st.warning("No reviews match your current filters — widen the genre, year, "
                       "or rating range in the sidebar before analyzing.")
        elif not api_key:
            st.info(
                "No API key entered — using an offline fallback (rating-based "
                "sentiment, word-frequency keywords) so you can still demo the app."
            )
        else:
            st.info(f"Will analyze **{sample_size}** of the **{len(filtered_df)}** filtered reviews.")

    if run_button:
        client = get_openai_client()
        sample_df = filtered_df.sample(n=min(sample_size, len(filtered_df)), random_state=1).sort_index()
        sentiments, keywords = [], []
        progress = st.progress(0, text="Analyzing reviews...")
        errors = 0
        for i, (_, row) in enumerate(sample_df.iterrows()):
            if client:
                try:
                    result = genai_analyze(client, row["review_text"])
                except Exception:
                    errors += 1
                    result = fallback_analyze(row["review_text"], row["rating"])
            else:
                result = fallback_analyze(row["review_text"], row["rating"])
            sentiments.append(result["sentiment"])
            keywords.append(result["keywords"])
            progress.progress((i + 1) / len(sample_df), text=f"Analyzing review {i + 1} of {len(sample_df)}...")
        progress.empty()
        sample_df = sample_df.copy()
        sample_df["sentiment"] = sentiments
        sample_df["keywords"] = keywords
        st.session_state["analyzed_df"] = sample_df
        if errors:
            st.warning(f"{errors} review(s) fell back to offline analysis due to GenAI errors.")
        st.success(f"Analyzed {len(sample_df)} reviews. See the Data and Charts tabs.")

    analyzed_df = st.session_state.get("analyzed_df")
    if analyzed_df is not None:
        st.dataframe(
            analyzed_df[["movie_title", "genre", "rating", "sentiment", "keywords", "review_text"]],
            width='stretch', height=380,
        )
    else:
        st.caption("Results will appear here after you click **Analyze reviews**.")

# --- Tab: Charts ----------------------------------------------------------
with tab_charts:
    analyzed_df = st.session_state.get("analyzed_df")
    if analyzed_df is not None:
        st.subheader("Sentiment Overview")
        c1, c2 = st.columns(2)
        with c1:
            sentiment_counts = analyzed_df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            fig = px.bar(
                sentiment_counts, x="sentiment", y="count", color="sentiment",
                title="Sentiment Distribution (Plotly)",
                color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
            )
            st.plotly_chart(fig, width='stretch')
        with c2:
            by_genre = analyzed_df.groupby(["genre", "sentiment"]).size().reset_index(name="count")
            chart = (
                alt.Chart(by_genre).mark_bar().encode(
                    x=alt.X("genre:N", title="Genre"),
                    y=alt.Y("count:Q", title="Number of reviews"),
                    color=alt.Color("sentiment:N", title="Sentiment"),
                    tooltip=["genre", "sentiment", "count"],
                ).properties(title="Sentiment by Genre (Altair)")
            )
            st.altair_chart(chart, width='stretch')

        st.subheader("Rating Over Time")
        ts = analyzed_df.dropna(subset=["review_date"]).sort_values("review_date")
        st.line_chart(ts.set_index("review_date")["rating"])

        st.subheader("Top Keywords")
        all_keywords = []
        for kw_str in analyzed_df["keywords"].dropna():
            all_keywords.extend([k.strip().lower() for k in str(kw_str).split(",") if k.strip()])
        if all_keywords:
            kw_counts = Counter(all_keywords).most_common(15)
            kw_df = pd.DataFrame(kw_counts, columns=["keyword", "count"])
            kc1, kc2 = st.columns([2, 1])
            with kc1:
                fig_kw = px.bar(
                    kw_df.sort_values("count"), x="count", y="keyword", orientation="h",
                    title="Most Common Keywords",
                )
                st.plotly_chart(fig_kw, width='stretch')
            with kc2:
                st.markdown("**Keywords by movie**")
                for movie in analyzed_df["movie_title"].unique():
                    movie_kws = analyzed_df.loc[analyzed_df["movie_title"] == movie, "keywords"]
                    tags = ", ".join(sorted(set(
                        k.strip() for row in movie_kws.dropna() for k in str(row).split(",") if k.strip()
                    )))
                    if tags:
                        st.markdown(f"- **{movie}**: {tags}")
        else:
            st.info("No keywords extracted yet.")
    else:
        st.info("Go to the **Run Analysis** tab and click **Analyze reviews** first.")

# --- Tab: Chatbot -----------------------------------------------------------
with tab_chat:
    st.subheader("Ask the Dataset Chatbot")
    st.caption(
        "Ask things like \"Which movie got the most negative reviews?\" or "
        "\"Summarize the Horror reviews.\" Answers use whatever's been analyzed so far."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.write(msg)

    user_question = st.chat_input("Ask a question about the filtered reviews...")

    if user_question:
        st.session_state["chat_history"].append(("user", user_question))
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🤖 _Thinking..._")
            client = get_openai_client()
            context_df = st.session_state.get("analyzed_df", filtered_df)
            if client:
                context = build_chat_context(context_df)
                try:
                    with st.spinner("Consulting the GenAI model..."):
                        response = client.chat.completions.create(
                            model=GENAI_MODEL,
                            messages=[
                                {"role": "system", "content": (
                                    "You answer questions about the movie review dataset "
                                    "provided as CSV below, which may include sentiment and "
                                    "extracted keyword columns. Be concise and cite movie "
                                    "titles when relevant.\n\n" + context
                                )},
                                {"role": "user", "content": user_question},
                            ],
                            temperature=0.3,
                            max_tokens=300,
                        )
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"GenAI call failed: {e}"
            else:
                answer = "Enter your API key in the sidebar to enable the chatbot."
            placeholder.markdown(answer)
        st.session_state["chat_history"].append(("assistant", answer))

# ----------------------------------------------------------------------
# Footer / dataset citation
# ----------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ About this dataset"):
    st.markdown(
        f"""
This app currently ships with a **synthetic** sample dataset
(`data/movie_reviews.csv`, **{len(df)} rows** across **{df['movie_title'].nunique()} fictional films**),
generated for this assignment so the app works offline with no download required.
Its structure (title, genre, year, reviewer, rating, date, review text) is modeled
after real, citable movie-review datasets:

- **Large Movie Review Dataset** (Maas, A. L., Daly, R. E., Pham, P. T., Huang, D.,
  Ng, A. Y., & Potts, C., 2011). *Learning Word Vectors for Sentiment Analysis*,
  ACL 2011. https://ai.stanford.edu/~amaas/data/sentiment/
- **Rotten Tomatoes Movie Reviews** (Kaggle). https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

**Want real movies and real reviews instead?** Run `fetch_real_dataset.py`
(included in this project) with a free TMDB API key — it pulls real movie
titles, genres, and real user reviews with genuine ratings directly from
[TMDB](https://www.themoviedb.org/) and overwrites this CSV in the same
format, no code changes needed. See the README for the 2-minute setup.
If you use that data, credit TMDB per their terms: *"This product uses
the TMDB API but is not endorsed or certified by TMDB."*
        """
    )
