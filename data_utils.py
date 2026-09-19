"""Shared helpers for loading the bundled CSV dataset."""
from __future__ import annotations

import csv
import numpy as np


def load_ratings(path="data/ratings.csv"):
    """Returns an (n, 3) float array of (user_id, book_id, rating) rows."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["user_id"]), float(row["book_id"]), float(row["rating"])))
    return np.array(rows, dtype=float)


def load_books(path="data/books.csv"):
    """Returns a dict: book_id (int) -> {"title": ..., "author": ...}."""
    books = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            books[int(row["book_id"])] = {"title": row["title"], "author": row["author"]}
    return books


def train_val_split(triples, val_frac=0.15, seed=0):
    """Random row-wise split of observed (user, item, rating) triples."""
    rng = np.random.default_rng(seed)
    n = len(triples)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return triples[train_idx], triples[val_idx]


def ratings_by_user(triples):
    """dict: user_id (int) -> set of book_ids that user rated."""
    out = {}
    for u, i, _ in triples:
        out.setdefault(int(u), set()).add(int(i))
    return out
