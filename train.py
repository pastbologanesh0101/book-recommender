"""
train.py — trains the matrix-factorization recommender on data/ratings.csv
and reports RMSE on a held-out validation split.

Usage:
    python train.py
    python train.py --epochs 60 --factors 12 --lr 0.01 --reg 0.02
"""
from __future__ import annotations

import argparse

import numpy as np

from data_utils import load_books, load_ratings, train_val_split
from mf import MFModel

MODEL_PATH = "model.npz"


def build_dims(triples):
    n_users = int(triples[:, 0].max()) + 1
    n_items = int(triples[:, 1].max()) + 1
    return n_users, n_items


def main():
    parser = argparse.ArgumentParser(description="Train the collaborative-filtering book recommender.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--factors", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--model-out", default=MODEL_PATH)
    args = parser.parse_args()

    triples = load_ratings()
    books = load_books()
    n_users, n_items = build_dims(triples)
    train_triples, val_triples = train_val_split(triples, val_frac=args.val_frac, seed=0)

    print(f"Loaded {len(triples)} ratings from {n_users} users x {len(books)} books "
          f"({len(triples) / (n_users * n_items):.1%} density)")
    print(f"Train rows: {len(train_triples)}  Val rows: {len(val_triples)}")

    model = MFModel(n_users=n_users, n_items=n_items, n_factors=args.factors,
                     lr=args.lr, reg=args.reg, seed=42)

    # Baseline: always predict the global mean rating.
    global_mean = float(train_triples[:, 2].mean())
    baseline_val_rmse = float(np.sqrt(np.mean((val_triples[:, 2] - global_mean) ** 2)))

    history = model.fit(train_triples, n_epochs=args.epochs, val_triples=val_triples, verbose=True)

    model.save(args.model_out)

    print()
    print("=" * 50)
    print(f"Final train RMSE:        {history['train_rmse'][-1]:.4f}")
    print(f"Final validation RMSE:   {history['val_rmse'][-1]:.4f}")
    print(f"Baseline (mean) RMSE:    {baseline_val_rmse:.4f}")
    improvement = 100 * (1 - history["val_rmse"][-1] / baseline_val_rmse)
    print(f"Improvement over baseline: {improvement:.1f}%")
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
