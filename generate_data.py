"""
generate_data.py — builds the synthetic dataset bundled in data/.

We generate ~50 users and ~80 books from TRUE underlying latent factor
vectors (a "taste" vector per user, a "style" vector per book), the same
kind of hidden structure that real collaborative-filtering data has. Each
user only rates a random subset of books (sparse matrix, like real life),
and observed ratings are the latent dot-product plus biases plus noise,
rounded to the nearest half-star and clipped to [1, 5].

This script is provided for reproducibility / transparency; the CSVs it
produces are already checked into data/, so you do not need to re-run it.
"""
from __future__ import annotations

import csv
import numpy as np

N_USERS = 50
N_BOOKS = 80
N_FACTORS = 5
MIN_RATINGS_PER_USER = 15
MAX_RATINGS_PER_USER = 40
SEED = 7

GENRES = [
    "Mystery", "Fantasy", "Sci-Fi", "Romance", "Thriller",
    "Historical", "Horror", "Biography", "Adventure", "Drama",
]
AUTHOR_FIRST = [
    "Elena", "Marcus", "Priya", "Jonas", "Amara", "Diego", "Yuki", "Fiona",
    "Idris", "Sofia", "Owen", "Nadia", "Leif", "Rosa", "Kenji", "Tara",
]
AUTHOR_LAST = [
    "Voss", "Okafor", "Lindqvist", "Marsh", "Petrov", "Iyer", "Callahan",
    "Nakamura", "Reyes", "Whitfield", "Abara", "Solberg", "Doyle", "Chen",
]


def generate_books(rng):
    books = []
    for book_id in range(N_BOOKS):
        genre = GENRES[book_id % len(GENRES)]
        title = f"The {genre} of {AUTHOR_LAST[(book_id * 3) % len(AUTHOR_LAST)]}'s Shadow" \
            if book_id % 2 == 0 else f"{genre} at the Edge of {AUTHOR_FIRST[book_id % len(AUTHOR_FIRST)]}'s World"
        author = f"{rng.choice(AUTHOR_FIRST)} {rng.choice(AUTHOR_LAST)}"
        books.append({"book_id": book_id, "title": title, "author": author})
    return books


def generate_ratings(rng):
    # Latent taste/style vectors -> the genuine collaborative structure.
    user_factors = rng.normal(scale=1.0, size=(N_USERS, N_FACTORS))
    book_factors = rng.normal(scale=1.0, size=(N_BOOKS, N_FACTORS))
    user_bias = rng.normal(scale=0.4, size=N_USERS)
    book_bias = rng.normal(scale=0.4, size=N_BOOKS)
    global_mean = 3.4

    rows = []
    for u in range(N_USERS):
        n_rated = rng.integers(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER + 1)
        rated_books = rng.choice(N_BOOKS, size=n_rated, replace=False)
        for b in rated_books:
            signal = global_mean + user_bias[u] + book_bias[b] + user_factors[u] @ book_factors[b]
            noisy = signal + rng.normal(scale=0.35)
            rating = float(np.clip(round(noisy * 2) / 2, 1.0, 5.0))
            rows.append((u, int(b), rating))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    books = generate_books(rng)
    ratings = generate_ratings(rng)

    with open("data/books.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["book_id", "title", "author"])
        writer.writeheader()
        writer.writerows(books)

    with open("data/ratings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "book_id", "rating"])
        writer.writerows(ratings)

    print(f"Wrote {len(books)} books and {len(ratings)} ratings "
          f"({len(ratings) / (N_USERS * N_BOOKS):.1%} density).")


if __name__ == "__main__":
    main()
