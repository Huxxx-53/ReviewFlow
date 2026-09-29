"""Smoke tests for ReviewFlow core logic (no Streamlit server required)."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re
import sys
from collections import Counter

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data" / "movie_reviews.csv"
APP = ROOT / "app.py"


def test_dataset():
    df = pd.read_csv(DATA)
    assert len(df) >= 520, f"expected >=520 rows, got {len(df)}"
    assert df.duplicated(subset=["review_text"]).sum() == 0
    assert df["rating"].between(1, 10).all()
    assert df["movie_title"].str.len().gt(0).all()
    assert df["review_text"].str.len().ge(40).all()
    print(f"[OK] dataset: {len(df)} rows, {df['movie_title'].nunique()} movies")


def test_app_static():
    src = APP.read_text(encoding="utf-8")
    ast.parse(src)
    assert 'type="password"' not in src
    assert "load_dotenv" in src
    assert "GROQ_API_KEY" in src
    assert "I couldn't find that information in the current TMDB dataset." in src
    assert "Keyword extraction identifies frequently occurring meaningful words" in src
    assert "Loading Data" in src
    assert "Generating Insights" in src
    assert "Dataset Overview" in src
    assert "Key Insights" in src
    print("[OK] app.py static checks")


def test_analysis_helpers():
    # Import functions from app without executing Streamlit page logic fully:
    # We re-implement the tiny pure helpers here to avoid Streamlit runtime.
    df = pd.read_csv(DATA)

    def rating_to_sentiment(rating: int) -> str:
        if rating >= 7:
            return "Positive"
        if rating <= 4:
            return "Negative"
        return "Neutral"

    df["sentiment"] = df["rating"].map(rating_to_sentiment)
    assert set(df["sentiment"]) <= {"Positive", "Neutral", "Negative"}

    stopwords = {"the", "a", "an", "and", "or", "but", "of", "to", "in", "is", "it", "film", "movie"}
    counts: Counter = Counter()
    for text in df["review_text"].astype(str):
        for w in re.findall(r"[a-zA-Z']+", text.lower()):
            if len(w) > 3 and w not in stopwords:
                counts[w] += 1
    top = counts.most_common(5)
    assert top and all(w not in stopwords for w, _ in top)
    print(f"[OK] analysis helpers; top keywords={top}")


def test_chat_query_via_app_module():
    """Load app functions by mocking streamlit enough to import query helpers."""
    import types

    class Ctx:
        def __init__(self, **attrs):
            for k, v in attrs.items():
                setattr(self, k, v)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def metric(self, *a, **k):
            return None

        def markdown(self, *a, **k):
            return None

        def progress(self, *a, **k):
            return self

    st = types.ModuleType("streamlit")
    st.session_state = {}
    st.secrets = {}
    st.set_page_config = lambda **k: None
    st.title = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.header = lambda *a, **k: None
    st.subheader = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.write = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.success = lambda *a, **k: None
    st.error = lambda *a, **k: None
    st.divider = lambda *a, **k: None
    st.metric = lambda *a, **k: None
    st.button = lambda *a, **k: False
    st.radio = lambda label, options, index=0, **k: options[index]
    st.chat_input = lambda *a, **k: None
    st.chat_message = lambda *a, **k: Ctx()
    st.empty = lambda: Ctx()
    st.progress = lambda *a, **k: Ctx()
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
    st.altair_chart = lambda *a, **k: None
    st.columns = lambda n: [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda labels: [Ctx() for _ in labels]
    st.sidebar = Ctx(
        header=st.header,
        caption=st.caption,
        divider=st.divider,
        metric=st.metric,
        success=st.success,
        warning=st.warning,
        error=st.error,
        multiselect=lambda *a, **k: list(a[1]) if len(a) > 1 else [],
        slider=lambda *a, **k: (1, 10),
        number_input=lambda *a, **k: 25,
    )
    st.multiselect = lambda label, options, default=None: list(options)
    st.slider = lambda *a, **k: (1, 10)
    st.number_input = lambda *a, **k: 25
    st.expander = lambda *a, **k: Ctx()

    def cache_data(fn=None, **kwargs):
        if fn is None:
            return lambda f: f
        return fn

    st.cache_data = cache_data
    sys.modules["streamlit"] = st

    spec = importlib.util.spec_from_file_location("reviewflow_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = mod.load_data(str(DATA))
    unrelated = mod.answer_from_dataset("What is the capital of France?", data)
    assert unrelated == "I only answer questions about the TMDB movie dataset in this app."

    natural = mod.answer_from_dataset(
        "Could you tell me which one people liked the most?", data
    )
    assert "**Dataset evidence**" in natural
    assert "### Answer" not in natural
    assert "Genre counts:" not in natural

    avg_ans = mod.answer_from_dataset("What is the average rating?", data)
    assert "**Dataset evidence**" in avg_ans
    assert "### Relevant Data" not in avg_ans
    assert "average" in avg_ans.lower() and "/ 10" in avg_ans
    assert "Genre counts:" not in avg_ans
    assert "Positive=" not in avg_ans

    high_ans = mod.answer_from_dataset("What is the highest-rated movie?", data)
    assert "| Movie |" in high_ans
    titles = re.findall(r"^\| ([^|]+) \|", high_ans, re.MULTILINE)
    movie_titles = [t.strip() for t in titles if t.strip() not in {"Movie", "---"}]
    assert len(movie_titles) == len(set(movie_titles))

    popularity_data = data.copy()
    ordered_titles = sorted(data["movie_title"].unique())
    title_scores = {title: float(index + 1) for index, title in enumerate(ordered_titles)}
    title_votes = {title: (index + 1) * 100 for index, title in enumerate(ordered_titles)}
    popularity_data["tmdb_popularity"] = popularity_data["movie_title"].map(title_scores)
    popularity_data["tmdb_vote_count"] = popularity_data["movie_title"].map(title_votes)
    horror_movies = mod.unique_movies(
        popularity_data[popularity_data["genre"].str.lower() == "horror"]
    )
    expected_popular = horror_movies.sort_values(
        ["tmdb_popularity", "tmdb_vote_count"], ascending=[False, False]
    ).iloc[0]["movie_title"]
    popular_ans = mod.answer_from_dataset(
        "What is the most famous horror movie?", popularity_data
    )
    assert expected_popular in popular_ans
    assert "TMDB popularity" in popular_ans
    assert "not a definitive measure" in popular_ans
    legacy_data = data.drop(columns=["tmdb_popularity", "tmdb_vote_count"], errors="ignore")
    assert "TMDB popularity is not present in this CSV yet" in mod.answer_from_dataset(
        "What is the most famous horror movie?", legacy_data
    )

    horror_movies = mod.unique_movies(data[data["genre"].str.lower() == "horror"])
    established_horror = horror_movies[horror_movies["review_count"] >= 5]
    if not established_horror.empty:
        horror_movies = established_horror
    expected_recommendation = horror_movies.sort_values(
        ["avg_rating", "review_count", "movie_title"],
        ascending=[False, False, True],
    ).iloc[0]["movie_title"]
    recommendation = mod.answer_from_dataset("Recommend me a horror movie", data)
    assert expected_recommendation in recommendation
    assert "I recommend" in recommendation
    assert "Top recommendations" in recommendation
    assert "**Dataset evidence**" in recommendation
    assert "There are **" not in recommendation

    missing = mod.answer_from_dataset("Which movies were released in 1800?", data)
    assert "couldn't find" in missing.lower() or "no movies" in missing.lower()

    print("[OK] chatbot query paths")
    print("  unrelated:", unrelated[:80].replace("\n", " "))
    print("  average snippet:", avg_ans[:120].replace("\n", " "))
    print("  year 1800:", missing[:80].replace("\n", " "))


if __name__ == "__main__":
    test_dataset()
    test_app_static()
    test_analysis_helpers()
    test_chat_query_via_app_module()
    print("\nALL TESTS PASSED")
