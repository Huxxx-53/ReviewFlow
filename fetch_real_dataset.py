"""
Fetch a REAL movie-reviews dataset from The Movie Database (TMDB) API.

This replaces the synthetic data/movie_reviews.csv with real movie titles,
real genres, and real user-submitted reviews pulled live from TMDB.

Run this on YOUR OWN machine (it needs internet access) — not inside a
sandboxed AI environment. It does not run automatically; you run it once
to (re)generate data/movie_reviews.csv.

------------------------------------------------------------------------
SETUP (2 minutes)
------------------------------------------------------------------------
1. Go to https://www.themoviedb.org/ and create a free account.
2. Go to Settings -> API -> Request an API Key (choose "Developer").
   Approval is usually instant.
3. Copy your "API Key (v3 auth)" value.
4. Either:
     export TMDB_API_KEY="your_key_here"      (Mac/Linux)
     setx TMDB_API_KEY "your_key_here"         (Windows)
   or paste it directly into API_KEY below.
5. Install requests if you don't already have it:
     pip install requests
6. Run:
     python fetch_real_dataset.py

------------------------------------------------------------------------
ATTRIBUTION (required by TMDB's terms of use)
------------------------------------------------------------------------
"This product uses the TMDB API but is not endorsed or certified by TMDB."
Keep this line in your README / submission when you use this data.
https://www.themoviedb.org/documentation/api/terms-of-use

------------------------------------------------------------------------
NOTES ON DATA QUALITY
------------------------------------------------------------------------
- Only reviews that include a real numeric TMDB user rating (1-10) are
  kept, so every "rating" in the resulting CSV is a genuine user score,
  not a guess. Many TMDB reviews don't include a rating, so this script
  pages through a large number of popular/top-rated movies to collect
  enough rated reviews to reach TARGET_ROWS.
- TMDB reviews are sparse per movie (many movies have zero), so this can
  take a few minutes and several hundred API calls. TMDB's free tier
  allows ~40 requests per 10 seconds, which this script respects.
- Unreleased TMDB listings (no release date, or a date in the future)
  are skipped so the CSV only contains movies that have already come out.
"""

import os
import csv
import time
from datetime import date

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("TMDB_API_KEY", "")  # set in .env — never commit the real key
BASE_URL = "https://api.themoviedb.org/3"
TARGET_ROWS = 1000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "movie_reviews.csv")
REQUEST_DELAY = 0.28  # stay comfortably under 40 req / 10 sec
MAX_RETRIES = 4
CHECKPOINT_EVERY = 50  # rows; saves partial progress so a crash doesn't lose everything


def api_get(path, **params):
    """GET with automatic retries for transient network errors (SSL resets,
    connection drops, timeouts) — common on flaky wifi and not a real bug."""
    params["api_key"] = API_KEY
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)
            time.sleep(REQUEST_DELAY)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = 2 * attempt
                print(f"    (network hiccup: {type(e).__name__}, retrying in {wait}s "
                      f"— attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
    raise last_error


def get_genre_map():
    data = api_get("/genre/movie/list")
    return {g["id"]: g["name"] for g in data["genres"]}


def iter_candidate_movies(genre_map, max_pages=40):
    """Yield popular + top-rated movies that have already been released."""
    today = date.today().isoformat()
    seen_ids = set()
    for endpoint in ["/movie/popular", "/movie/top_rated"]:
        for page in range(1, max_pages + 1):
            try:
                data = api_get(endpoint, page=page)
            except requests.exceptions.RequestException as e:
                print(f"  Skipping {endpoint} page {page} after repeated network "
                      f"errors ({type(e).__name__}); moving on.")
                break
            results = data.get("results", [])
            if not results:
                break
            for m in results:
                if m["id"] in seen_ids:
                    continue
                seen_ids.add(m["id"])
                if m.get("adult"):
                    continue
                release_date = (m.get("release_date") or "").strip()
                if len(release_date) < 10 or release_date > today:
                    continue  # skip unreleased / undated TMDB listings
                if int(m.get("vote_count") or 0) < 50:
                    continue  # skip stubs with no real audience yet
                genre_names = [genre_map.get(gid, "") for gid in m.get("genre_ids", [])]
                genre_names = [g for g in genre_names if g]
                if not genre_names:
                    continue
                yield {
                    "id": m["id"],
                    "title": m["title"],
                    "genre": genre_names[0],
                    "year": release_date[:4],
                    "popularity": m.get("popularity"),
                    "vote_count": m.get("vote_count"),
                }


def get_rated_reviews(movie_id, max_pages=3):
    """Return reviews that include a real numeric author rating."""
    rows = []
    for page in range(1, max_pages + 1):
        data = api_get(f"/movie/{movie_id}/reviews", page=page)
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            rating = (r.get("author_details") or {}).get("rating")
            if rating is None:
                continue  # skip reviews without a genuine numeric rating
            content = (r.get("content") or "").strip().replace("\n", " ")
            if len(content) < 20:
                continue
            rows.append({
                "reviewer": r.get("author", "anonymous"),
                "rating": round(float(rating)),
                "review_date": (r.get("created_at") or "")[:10],
                "review_text": content[:1000],  # keep rows manageable
            })
        if page >= data.get("total_pages", 1):
            break
    return rows


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "review_id", "movie_title", "genre", "release_year",
            "tmdb_popularity", "tmdb_vote_count",
            "reviewer", "rating", "review_date", "review_text",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not API_KEY:
        raise SystemExit(
            "No TMDB API key found. Set the TMDB_API_KEY environment variable "
            "or paste your key into API_KEY in this script, then run again."
        )

    print("Fetching genre list...")
    genre_map = get_genre_map()

    all_rows = []
    review_id = 1
    last_checkpoint = 0
    print(f"Collecting real reviews until we reach {TARGET_ROWS} rows...")

    for movie in iter_candidate_movies(genre_map):
        if len(all_rows) >= TARGET_ROWS:
            break
        try:
            reviews = get_rated_reviews(movie["id"])
        except requests.exceptions.RequestException as e:
            print(f"  Skipping '{movie['title']}' after repeated network errors "
                  f"({type(e).__name__}).")
            continue
        for rev in reviews:
            all_rows.append({
                "review_id": review_id,
                "movie_title": movie["title"],
                "genre": movie["genre"],
                "release_year": movie["year"],
                "tmdb_popularity": movie["popularity"],
                "tmdb_vote_count": movie["vote_count"],
                **rev,
            })
            review_id += 1
        if reviews:
            print(f"  {movie['title']} ({movie['year']}): +{len(reviews)} rated reviews "
                  f"[total: {len(all_rows)}]")

        # Save partial progress periodically so a crash doesn't lose everything.
        if len(all_rows) - last_checkpoint >= CHECKPOINT_EVERY:
            save_csv(all_rows, OUTPUT_PATH)
            last_checkpoint = len(all_rows)
            print(f"    (checkpoint saved: {len(all_rows)} rows written so far)")

    if not all_rows:
        raise SystemExit("No rated reviews were found. Try increasing max_pages in "
                          "iter_candidate_movies().")

    save_csv(all_rows, OUTPUT_PATH)

    print(f"\nDone. Wrote {len(all_rows)} real reviews to {OUTPUT_PATH}")
    print('Remember to credit TMDB: "This product uses the TMDB API but is not '
          'endorsed or certified by TMDB." — https://www.themoviedb.org/')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped early — whatever was collected before this point has "
              "already been saved (checkpoints save every "
              f"{CHECKPOINT_EVERY} rows).")
