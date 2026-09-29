"""
Clean the TMDB movie-reviews CSV used by the app.

Removes unreleased/stub listings, duplicates, empty/corrupted rows, and
synthetic leftover text. Does NOT fabricate rows.

Set CLEAN_VERIFY_TMDB=1 to also check each title against the TMDB API
(slower; requires TMDB_API_KEY).
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "movie_reviews.csv")
MIN_TEXT_LEN = 40
MIN_TITLE_LEN = 1
MIN_REVIEWS_CURRENT_YEAR = 8
FAKE_TITLE_PATTERNS = re.compile(
    r"\b(?:test|fake|dummy|sample|lorem|ipsum|asdf|xxx|untitled|n/?a)\b",
    re.IGNORECASE,
)
SYNTHETIC_OPENERS = [
    "an absolutely stunning experience from start to finish",
    "this completely exceeded my expectations",
    "a perfectly watchable but unremarkable film",
    "overall, the fight choreography was exceptional",
]
TMDB_BASE = "https://api.themoviedb.org/3"


def _tmdb_released_map(pairs: list[tuple[str, int]], api_key: str) -> dict[tuple[str, int], bool]:
    today = date.today()
    cache: dict[tuple[str, int], bool] = {}
    for i, (title, year) in enumerate(pairs, start=1):
        ok = False
        try:
            resp = requests.get(
                f"{TMDB_BASE}/search/movie",
                params={
                    "api_key": api_key,
                    "query": title,
                    "year": year,
                    "include_adult": "false",
                },
                timeout=20,
            )
            time.sleep(0.28)
            resp.raise_for_status()
            results = resp.json().get("results") or []
        except requests.exceptions.RequestException as e:
            print(f"  TMDB lookup failed for '{title}' ({year}): {type(e).__name__}", flush=True)
            cache[(title, year)] = False
            continue

        title_l = title.lower().strip()
        for r in results:
            cand = (r.get("title") or r.get("original_title") or "").lower().strip()
            if cand != title_l:
                continue
            raw = (r.get("release_date") or "").strip()
            if len(raw) < 10:
                continue
            try:
                released = datetime.strptime(raw[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
            if released.year != int(year) or released > today:
                continue
            if int(r.get("vote_count") or 0) < 50:
                continue
            ok = True
            break
        cache[(title, year)] = ok
        status = "kept" if ok else "removed"
        print(f"  [{i}/{len(pairs)}] {title} ({year}): {status}", flush=True)
    return cache


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "review_id", "movie_title", "genre", "release_year",
        "reviewer", "rating", "review_date", "review_text",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")

    before = len(df)
    df = df.copy()
    today = date.today()

    for col in ["movie_title", "genre", "reviewer", "review_text"]:
        df[col] = df[col].astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    df = df.dropna(subset=["movie_title", "genre", "release_year", "rating", "review_text"])
    df = df[df["movie_title"].str.len() >= MIN_TITLE_LEN]
    df = df[df["genre"].str.len() >= 1]
    df = df[df["review_text"].str.len() >= MIN_TEXT_LEN]
    df = df[~df["review_text"].str.lower().isin(["nan", "none", "null", "n/a"])]

    df = df[df["rating"].between(1, 10)]
    df["rating"] = df["rating"].round().astype(int)

    df = df[df["release_year"].between(1900, today.year)]
    df["release_year"] = df["release_year"].astype(int)
    df = df[df["review_date"].isna() | (df["review_date"].dt.date <= today)]

    df = df[~df["movie_title"].str.contains(FAKE_TITLE_PATTERNS, na=False)]
    df = df[~df["reviewer"].str.lower().isin(["", "nan", "none", "null", "n/a", "test"])]

    text_l = df["review_text"].str.lower()
    synth_mask = False
    for opener in SYNTHETIC_OPENERS:
        synth_mask = synth_mask | text_l.str.contains(re.escape(opener), na=False)
    df = df[~synth_mask]

    df = df.drop_duplicates(subset=["review_text"], keep="first")
    df = df.drop_duplicates(
        subset=["movie_title", "reviewer", "review_date", "rating"], keep="first"
    )

    # TMDB popular lists include unreleased stubs. Keep prior years, and only
    # current-year titles that already have several real reviews in this file.
    counts = df.groupby("movie_title")["review_text"].transform("size")
    established = df["release_year"] < today.year
    current_real = (df["release_year"] == today.year) & (counts >= MIN_REVIEWS_CURRENT_YEAR)
    removed_stubs = int((~(established | current_real)).sum())
    df = df[established | current_real]
    print(f"Removed {removed_stubs} reviews from unreleased/stub titles "
          f"(kept {today.year} films only if they have "
          f">= {MIN_REVIEWS_CURRENT_YEAR} reviews).", flush=True)

    if os.environ.get("CLEAN_VERIFY_TMDB", "").strip() in {"1", "true", "True"}:
        api_key = os.environ.get("TMDB_API_KEY", "").strip()
        if not api_key:
            print("CLEAN_VERIFY_TMDB is set but TMDB_API_KEY is missing.", flush=True)
        else:
            pairs = sorted(set(zip(df["movie_title"], df["release_year"])))
            print(f"Checking {len(pairs)} titles against TMDB...", flush=True)
            released = _tmdb_released_map(pairs, api_key)
            keep = [
                released.get((row.movie_title, int(row.release_year)), False)
                for row in df.itertuples()
            ]
            dropped = int((~pd.Series(keep, index=df.index)).sum())
            df = df[keep]
            print(f"Removed {dropped} reviews that TMDB did not confirm as released.", flush=True)

    df = df.sort_values(["release_year", "movie_title", "review_date"], kind="stable")
    df = df.reset_index(drop=True)
    df["review_id"] = df.index + 1
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    after = len(df)
    print(f"Cleaned dataset: {before} -> {after} rows "
          f"({before - after} removed), "
          f"{df['movie_title'].nunique()} unique movies.", flush=True)
    if after < 520:
        print(
            "WARNING: fewer than 520 legitimate rows remain. "
            "Re-run fetch_real_dataset.py (it now skips unreleased TMDB listings).",
            flush=True,
        )
    return df[required]


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"No dataset found at {DATA_PATH}. "
            "Run fetch_real_dataset.py first to pull real TMDB reviews."
        )
    raw = pd.read_csv(DATA_PATH)
    cleaned = clean_dataframe(raw)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    cleaned.to_csv(DATA_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(cleaned)} validated TMDB records to {DATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
