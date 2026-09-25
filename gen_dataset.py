import random
import csv
from datetime import date, timedelta

random.seed(7)

genre_vocab = {
    "Action": {
        "pos": ["the fight choreography was exceptional", "the stunt work felt genuinely dangerous",
                "the set pieces kept escalating in the best way", "the pacing never let up"],
        "neg": ["the action scenes were edited into incoherence", "the stakes never felt real",
                "the villain was forgettable", "it leaned too hard on CGI over practical stunts"],
        "neu": ["the action is competent but nothing new", "a few set pieces stand out, the rest blur together"],
    },
    "Drama": {
        "pos": ["the lead performance carried real emotional weight", "the script trusted its audience",
                "every quiet scene landed", "the character arcs felt earned"],
        "neg": ["it leaned on melodrama instead of earning its emotion", "the pacing dragged badly",
                "the dialogue felt overwritten", "the resolution felt unearned"],
        "neu": ["it's competently acted but familiar territory", "a few scenes resonate, most feel routine"],
    },
    "Comedy": {
        "pos": ["the timing was impeccable", "the jokes actually landed one after another",
                "the cast had real chemistry", "it was smarter than the trailer suggested"],
        "neg": ["most of the jokes fell flat", "it tried too hard to be quirky",
                "the humor felt dated", "the pacing killed the comedic momentum"],
        "neu": ["a handful of laughs, mostly forgettable", "fine for a background watch"],
    },
    "Horror": {
        "pos": ["the sound design alone made it unsettling", "the atmosphere never let up",
                "the scares were earned rather than cheap", "the tension built expertly"],
        "neg": ["it leaned entirely on jump scares", "the pacing lost all tension by the third act",
                "the plot made little sense by the end", "it wasn't remotely scary"],
        "neu": ["a few effective scares, the rest is filler", "atmospheric but forgettable"],
    },
    "Sci-Fi": {
        "pos": ["the world-building was genuinely immersive", "the ideas felt fresh and ambitious",
                "the visuals were stunning", "the concept was executed with real care"],
        "neg": ["the plot had too many holes", "the world-building never got explained",
                "the pacing dragged through exposition", "the ideas outran the execution"],
        "neu": ["ambitious but uneven", "some strong ideas buried in a messy plot"],
    },
    "Thriller": {
        "pos": ["the twists actually landed", "the tension never let up",
                "it kept me guessing until the end", "the pacing was relentless"],
        "neg": ["the twist was predictable from the start", "the plot fell apart under scrutiny",
                "the pacing dragged in the middle", "the ending felt rushed"],
        "neu": ["decent but predictable if you've seen a few of these", "solid setup, weaker payoff"],
    },
    "Fantasy": {
        "pos": ["the world felt lived-in and detailed", "the production design was gorgeous",
                "the mythology was genuinely compelling", "it balanced spectacle and story well"],
        "neg": ["it leaned on tired genre tropes", "the world-building overwhelmed the story",
                "the pacing suffered under all the lore", "the CGI looked inconsistent"],
        "neu": ["visually nice but narratively familiar", "some fun ideas, uneven execution"],
    },
    "Romance": {
        "pos": ["the leads had real chemistry", "it earned its emotional beats",
                "the dialogue felt natural and warm", "the pacing let the relationship breathe"],
        "neg": ["the chemistry never quite clicked", "it leaned on cliches",
                "the conflict felt manufactured", "the pacing rushed the relationship"],
        "neu": ["sweet but formulaic", "a pleasant watch, nothing more"],
    },
    "Animation": {
        "pos": ["the animation style was gorgeous", "it balanced humor and heart well",
                "the voice cast was excellent", "it worked for kids and adults alike"],
        "neg": ["the story felt aimed only at very young kids", "the humor missed more than it hit",
                "the pacing dragged for an animated feature", "the plot was thin"],
        "neu": ["visually nice but a fairly standard story", "fine family entertainment, nothing more"],
    },
    "Documentary": {
        "pos": ["the access to its subjects was remarkable", "it presented a complex issue clearly",
                "the editing kept it consistently engaging", "it never felt one-sided"],
        "neg": ["it felt one-sided throughout", "the pacing dragged in the middle stretch",
                "it barely scratched the surface of its subject", "the structure was confusing"],
        "neu": ["informative but a bit dry", "solid research, flat presentation"],
    },
}

positive_openers = [
    "An absolutely stunning experience from start to finish.",
    "This completely exceeded my expectations.",
    "One of the best films I've seen this year.",
    "A genuinely moving and well-crafted piece of work.",
    "I was hooked from the very first scene.",
    "Beautifully made and deeply satisfying.",
    "A masterclass in its genre.",
    "Left the theater with a huge smile on my face.",
]
negative_openers = [
    "I really wanted to like this more than I did.",
    "This was a disappointing watch overall.",
    "Not what I expected, and not in a good way.",
    "It had potential but fell flat.",
    "A frustrating misfire.",
    "I struggled to stay engaged the whole time.",
    "This didn't work for me at all.",
    "A forgettable entry, unfortunately.",
]
neutral_openers = [
    "A perfectly watchable but unremarkable film.",
    "Solid but not particularly memorable.",
    "Decent for a one-time watch.",
    "Middle-of-the-road in almost every way.",
    "Neither great nor terrible, just okay.",
]

title_parts_a = ["Skyline", "Midnight", "Iron", "Whispers of", "Static", "Paper", "Silent", "Crimson", "Broken",
                  "Last", "Hidden", "Velvet", "Golden", "Shattered", "Echoes of", "Rusted", "Winter's", "Neon",
                  "Forgotten", "Ashen", "Amber", "Faded", "Distant", "Endless", "Quiet", "Northern", "Scarlet",
                  "Hollow", "Wandering", "Ember"]
title_parts_b = ["Horizon", "Ledger", "Tide", "Home", "Frequency", "Kingdoms", "Harbor", "Oath", "Glass",
                  "Signal", "Garden", "Empire", "Crossing", "Vow", "Reverie", "Current", "Light", "Static",
                  "Hollow", "Bloom", "Drift", "Ruin", "Passage", "Vein", "Chorus", "Fields", "Line", "Season",
                  "Threshold", "Echo"]

genres = list(genre_vocab.keys())
movies = []
seen = set()
while len(movies) < 75:
    title = f"{random.choice(title_parts_a)} {random.choice(title_parts_b)}"
    if title not in seen:
        seen.add(title)
        movies.append({"title": title, "genre": random.choice(genres), "year": random.randint(2018, 2024)})

first_names = ["J.", "M.", "A.", "K.", "R.", "T.", "S.", "L.", "D.", "E.", "F.", "G.", "H.", "I.", "N.", "P.",
               "B.", "C.", "O.", "V.", "W.", "Y.", "Z.", "Q.", "U."]
last_names = ["Cruz", "Reyes", "Santos", "Bautista", "Dizon", "Mercado", "Ramos", "Flores", "Garcia", "Villanueva",
              "Torres", "Aquino", "Navarro", "Domingo", "Salazar", "Pascual", "Castillo", "Manalo", "Ocampo",
              "Espino", "Ibarra", "Valdez", "Fajardo", "Sarmiento", "Lim", "Tan", "Rivera", "Gomez", "Cortez", "Vega",
              "Mendoza", "Agbayani", "Del Rosario", "Panganiban", "Quiambao"]

rows = []
start_date = date(2019, 1, 1)

for movie in movies:
    vocab = genre_vocab[movie["genre"]]
    num_reviews = random.randint(15, 22)
    for _ in range(num_reviews):
        bucket = random.choices(["pos", "neg", "neu"], weights=[0.5, 0.3, 0.2])[0]
        if bucket == "pos":
            opener = random.choice(positive_openers)
            detail = random.choice(vocab["pos"])
            rating = random.randint(7, 10)
        elif bucket == "neg":
            opener = random.choice(negative_openers)
            detail = random.choice(vocab["neg"])
            rating = random.randint(1, 4)
        else:
            opener = random.choice(neutral_openers)
            detail = random.choice(vocab["neu"])
            rating = random.randint(5, 6)

        text = f"{opener} Overall, {detail}."
        reviewer = f"{random.choice(first_names)} {random.choice(last_names)}"
        review_date = start_date + timedelta(days=random.randint(0, 365 * 6))

        rows.append({
            "movie_title": movie["title"], "genre": movie["genre"], "release_year": movie["year"],
            "reviewer": reviewer, "rating": rating, "review_date": review_date.isoformat(),
            "review_text": text,
        })

random.shuffle(rows)
seen_pairs = set()
unique_rows = []
for r in rows:
    key = (r["movie_title"], r["review_text"])
    if key not in seen_pairs:
        seen_pairs.add(key)
        unique_rows.append(r)

print(f"Generated pool: {len(unique_rows)} unique rows across {len(movies)} movies.")
rows = unique_rows[:1050]

for i, r in enumerate(rows, start=1):
    r["review_id"] = i

with open("data/movie_reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["review_id", "movie_title", "genre", "release_year", "reviewer", "rating", "review_date", "review_text"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows.")
