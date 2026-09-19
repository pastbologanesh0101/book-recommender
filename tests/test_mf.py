"""
Unit tests for the from-scratch matrix-factorization collaborative filter.

Run with:  python -m pytest tests/  (or  python -m unittest discover tests)
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_books, load_ratings, train_val_split, ratings_by_user
from mf import MFModel, recommend


def make_synthetic_triples(n_users=20, n_items=30, n_factors=4, seed=1, density=0.4):
    """Small, fast synthetic dataset with genuine latent structure, for tests."""
    rng = np.random.default_rng(seed)
    user_f = rng.normal(scale=1.0, size=(n_users, n_factors))
    item_f = rng.normal(scale=1.0, size=(n_items, n_factors))
    rows = []
    for u in range(n_users):
        n_rated = max(3, int(density * n_items))
        items = rng.choice(n_items, size=n_rated, replace=False)
        for i in items:
            signal = 3.4 + user_f[u] @ item_f[i]
            rating = float(np.clip(round((signal + rng.normal(scale=0.3)) * 2) / 2, 1.0, 5.0))
            rows.append((u, int(i), rating))
    return np.array(rows, dtype=float), n_users, n_items


class TestTrainingConverges(unittest.TestCase):
    def test_training_reduces_rmse_over_epochs(self):
        triples, n_users, n_items = make_synthetic_triples()
        model = MFModel(n_users=n_users, n_items=n_items, n_factors=4, lr=0.02, reg=0.02, seed=0)
        history = model.fit(triples, n_epochs=25)
        # Loss should trend down: last epoch clearly better than first.
        self.assertLess(history["train_rmse"][-1], history["train_rmse"][0])
        # And a strict, non-noisy check: average of the final 5 epochs beats
        # the average of the first 5 epochs by a real margin.
        early = np.mean(history["train_rmse"][:5])
        late = np.mean(history["train_rmse"][-5:])
        self.assertLess(late, early * 0.9)


class TestPredictiveAccuracy(unittest.TestCase):
    def test_model_beats_mean_baseline_on_held_out_data(self):
        triples, n_users, n_items = make_synthetic_triples(n_users=30, n_items=40, density=0.5)
        train_triples, val_triples = train_val_split(triples, val_frac=0.2, seed=0)

        model = MFModel(n_users=n_users, n_items=n_items, n_factors=5, lr=0.02, reg=0.05, seed=0)
        model.fit(train_triples, n_epochs=40)
        model_rmse = model.rmse(val_triples)

        baseline_pred = train_triples[:, 2].mean()
        baseline_rmse = float(np.sqrt(np.mean((val_triples[:, 2] - baseline_pred) ** 2)))

        # Quantitative assertion: model must clearly beat the trivial baseline.
        self.assertLess(model_rmse, baseline_rmse)
        self.assertLess(model_rmse, baseline_rmse * 0.85)


class TestRecommend(unittest.TestCase):
    def setUp(self):
        self.triples, self.n_users, self.n_items = make_synthetic_triples(n_users=15, n_items=25)
        self.model = MFModel(n_users=self.n_users, n_items=self.n_items, n_factors=4, seed=0)
        self.model.fit(self.triples, n_epochs=20)
        self.by_user = ratings_by_user(self.triples)
        self.all_items = list(range(self.n_items))

    def test_recommend_excludes_already_rated_books(self):
        user_id = 0
        rated = self.by_user[user_id]
        recs = recommend(self.model, user_id, rated, self.all_items, top_n=5)
        self.assertTrue(set(recs).isdisjoint(rated))

    def test_recommend_returns_valid_book_ids(self):
        user_id = 1
        rated = self.by_user[user_id]
        recs = recommend(self.model, user_id, rated, self.all_items, top_n=5)
        for book_id in recs:
            self.assertIn(book_id, self.all_items)

    def test_recommend_returns_requested_count(self):
        user_id = 2
        rated = self.by_user[user_id]
        recs = recommend(self.model, user_id, rated, self.all_items, top_n=5)
        # Enough unrated items exist, so we should get exactly top_n back.
        self.assertEqual(len(recs), 5)


class TestColdStart(unittest.TestCase):
    def setUp(self):
        self.triples, self.n_users, self.n_items = make_synthetic_triples(n_users=15, n_items=25)
        self.model = MFModel(n_users=self.n_users, n_items=self.n_items, n_factors=4, seed=0)
        self.model.fit(self.triples, n_epochs=15)
        self.all_items = list(range(self.n_items))

    def test_unseen_user_id_does_not_crash_and_uses_popularity_fallback(self):
        """A user id far outside the trained range (e.g. brand new signup):
        recommend() must not crash, and falls back to popularity ranking
        (highest global_mean + item_bias), documented in mf.recommend()."""
        unseen_user_id = self.n_users + 500
        recs = recommend(self.model, unseen_user_id, rated_item_ids=set(),
                          all_item_ids=self.all_items, top_n=5)
        self.assertEqual(len(recs), 5)
        expected_popularity_order = sorted(
            self.all_items, key=lambda iid: self.model.mu + self.model.b_i[iid], reverse=True
        )[:5]
        self.assertEqual(recs, expected_popularity_order)

    def test_user_with_one_rating_does_not_crash(self):
        """A user who has rated only a single book still gets a sensible,
        non-crashing personalized recommendation list (their latent vector
        is noisy but present since they exist in the trained model)."""
        # Retrain a tiny model where one user has only 1 rating.
        triples = list(self.triples)
        sparse_triples = [t for t in triples if t[0] != 3] + [(3, 0, 4.0)]
        sparse_triples = np.array(sparse_triples, dtype=float)
        model = MFModel(n_users=self.n_users, n_items=self.n_items, n_factors=4, seed=1)
        model.fit(sparse_triples, n_epochs=15)

        recs = recommend(model, 3, rated_item_ids={0}, all_item_ids=self.all_items, top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertNotIn(0, recs)


class TestSaveLoad(unittest.TestCase):
    def test_save_load_roundtrips_predictions(self):
        triples, n_users, n_items = make_synthetic_triples(n_users=10, n_items=15)
        model = MFModel(n_users=n_users, n_items=n_items, n_factors=3, seed=2)
        model.fit(triples, n_epochs=15)

        before = np.array([[model.predict(u, i) for i in range(n_items)] for u in range(n_users)])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.npz")
            model.save(path)
            loaded = MFModel.load(path)

        after = np.array([[loaded.predict(u, i) for i in range(n_items)] for u in range(n_users)])
        np.testing.assert_allclose(before, after, rtol=1e-10, atol=1e-10)


class TestPredictionBounds(unittest.TestCase):
    def test_predictions_stay_within_rating_scale(self):
        triples, n_users, n_items = make_synthetic_triples(n_users=12, n_items=18)
        model = MFModel(n_users=n_users, n_items=n_items, n_factors=6, lr=0.05, reg=0.0, seed=3)
        model.fit(triples, n_epochs=30)
        for u in range(n_users):
            for i in range(n_items):
                pred = model.predict(u, i)
                self.assertGreaterEqual(pred, 1.0)
                self.assertLessEqual(pred, 5.0)


class TestBundledDataset(unittest.TestCase):
    def test_bundled_csvs_load_and_are_consistent(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ratings_path = os.path.join(repo_root, "data", "ratings.csv")
        books_path = os.path.join(repo_root, "data", "books.csv")

        triples = load_ratings(ratings_path)
        books = load_books(books_path)

        self.assertGreater(len(triples), 500)
        self.assertGreaterEqual(len(books), 50)

        max_book_id = int(triples[:, 1].max())
        self.assertIn(max_book_id, books)
        # Ratings should be within the 1-5 star scale.
        self.assertTrue(np.all(triples[:, 2] >= 1.0))
        self.assertTrue(np.all(triples[:, 2] <= 5.0))


if __name__ == "__main__":
    unittest.main()
