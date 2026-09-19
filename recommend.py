"""
recommend.py — prints the top-N predicted-highest-rated UNRATED books for a
given user, using the trained collaborative-filtering model.

Usage:
    python recommend.py --user-id 7
    python recommend.py --user-id 7 --top-n 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from data_utils import load_books, load_ratings, ratings_by_user
from mf import MFModel, recommend

MODEL_PATH = "model.npz"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found — running train.py first...\n")
        subprocess.run([sys.executable, "train.py"], check=True)
        print()


def main():
    parser = argparse.ArgumentParser(description="Recommend books for a user (collaborative filtering).")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    ensure_model()

    model = MFModel.load(args.model)
    triples = load_ratings()
    books = load_books()
    by_user = ratings_by_user(triples)

    already_rated = by_user.get(args.user_id, set())
    all_book_ids = list(books.keys())

    if args.user_id < 0 or args.user_id >= model.n_users:
        print(f"User {args.user_id} is unseen/out of range for this model "
              f"(trained on users 0..{model.n_users - 1}).")
        print("Falling back to popularity-based cold-start recommendations.\n")

    top_ids = recommend(model, args.user_id, already_rated, all_book_ids, top_n=args.top_n)

    print(f"Top {len(top_ids)} recommendations for user {args.user_id} "
          f"({len(already_rated)} books already rated):\n")
    for rank, book_id in enumerate(top_ids, start=1):
        info = books[book_id]
        if 0 <= args.user_id < model.n_users:
            score = model.predict(args.user_id, book_id)
        else:
            score = model.mu + model.b_i[book_id]
        print(f"  {rank}. \"{info['title']}\" by {info['author']}  "
              f"(book_id={book_id}, predicted rating={score:.2f})")


if __name__ == "__main__":
    main()
