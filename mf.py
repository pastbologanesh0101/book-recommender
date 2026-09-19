"""
mf.py — Matrix factorization collaborative filtering, implemented from scratch
with numpy (SGD, no scikit-learn).

Core idea (collaborative filtering via matrix factorization):
  We approximate the sparse user x item ratings matrix R as

      R[u, i] ~= mu + b_u[u] + b_i[i] + P[u] . Q[i]

  where:
    mu     - global average rating
    b_u, b_i - per-user / per-item bias terms
    P      - (n_users x n_factors) matrix of user latent-taste vectors
    Q      - (n_items x n_factors) matrix of item latent-trait vectors

  Only OBSERVED (user, item, rating) triples are used for training. We never
  look at genres, tags, or text describing a book — the model discovers
  latent structure purely from the pattern of *who rated what how*. That is
  what makes this collaborative filtering rather than content-based
  filtering.

  Training is plain stochastic gradient descent: for every observed rating
  we compute the prediction error, then nudge b_u, b_i, P[u], Q[i] a small
  step down the gradient of the L2-regularized squared error loss.
"""
from __future__ import annotations

import numpy as np


class MFModel:
    """Matrix-factorization recommender trained with SGD."""

    def __init__(self, n_users, n_items, n_factors=10, lr=0.01, reg=0.02, seed=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg

        rng = np.random.default_rng(seed)
        # Small random init keeps early dot products near zero.
        self.P = rng.normal(scale=0.1, size=(n_users, n_factors))
        self.Q = rng.normal(scale=0.1, size=(n_items, n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.mu = 0.0

    def predict(self, u, i):
        """Predict a single (user_idx, item_idx) rating, clipped to [1, 5]."""
        pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
        return float(np.clip(pred, 1.0, 5.0))

    def predict_all_for_user(self, u):
        """Predicted rating for every item, for one user index. Shape (n_items,)."""
        raw = self.mu + self.b_u[u] + self.b_i + self.Q @ self.P[u]
        return np.clip(raw, 1.0, 5.0)

    def rmse(self, triples):
        """RMSE over an array of (user_idx, item_idx, rating) rows."""
        if len(triples) == 0:
            return float("nan")
        users = triples[:, 0].astype(int)
        items = triples[:, 1].astype(int)
        actual = triples[:, 2]
        preds = self.mu + self.b_u[users] + self.b_i[items] + np.sum(
            self.P[users] * self.Q[items], axis=1
        )
        preds = np.clip(preds, 1.0, 5.0)
        return float(np.sqrt(np.mean((preds - actual) ** 2)))

    def fit(self, train_triples, n_epochs=100, val_triples=None, verbose=False, seed=0):
        """
        Train with SGD over observed (user_idx, item_idx, rating) rows.

        Returns a dict with 'train_rmse' (and 'val_rmse' if val_triples given),
        one entry per epoch, so callers/tests can check that loss decreases.
        """
        train_triples = np.asarray(train_triples, dtype=float)
        self.mu = float(train_triples[:, 2].mean())

        rng = np.random.default_rng(seed)
        history = {"train_rmse": []}
        if val_triples is not None:
            history["val_rmse"] = []

        n = len(train_triples)
        order = np.arange(n)
        for epoch in range(n_epochs):
            rng.shuffle(order)
            for idx in order:
                u, i, r = train_triples[idx]
                u = int(u)
                i = int(i)
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
                err = r - pred

                # SGD update, L2-regularized.
                b_u_old = self.b_u[u]
                b_i_old = self.b_i[i]
                p_old = self.P[u].copy()
                q_old = self.Q[i].copy()

                self.b_u[u] += self.lr * (err - self.reg * b_u_old)
                self.b_i[i] += self.lr * (err - self.reg * b_i_old)
                self.P[u] += self.lr * (err * q_old - self.reg * p_old)
                self.Q[i] += self.lr * (err * p_old - self.reg * q_old)

            train_rmse = self.rmse(train_triples)
            history["train_rmse"].append(train_rmse)
            if val_triples is not None:
                val_rmse = self.rmse(np.asarray(val_triples, dtype=float))
                history["val_rmse"].append(val_rmse)
            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                msg = f"epoch {epoch + 1:3d}/{n_epochs}  train_rmse={train_rmse:.4f}"
                if val_triples is not None:
                    msg += f"  val_rmse={history['val_rmse'][-1]:.4f}"
                print(msg)

        return history

    def save(self, path):
        np.savez(
            path,
            P=self.P,
            Q=self.Q,
            b_u=self.b_u,
            b_i=self.b_i,
            mu=self.mu,
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            lr=self.lr,
            reg=self.reg,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        model = cls(
            n_users=int(data["n_users"]),
            n_items=int(data["n_items"]),
            n_factors=int(data["n_factors"]),
            lr=float(data["lr"]),
            reg=float(data["reg"]),
        )
        model.P = data["P"]
        model.Q = data["Q"]
        model.b_u = data["b_u"]
        model.b_i = data["b_i"]
        model.mu = float(data["mu"])
        return model


def recommend(model: MFModel, user_id, rated_item_ids, all_item_ids, top_n=5):
    """
    Return up to `top_n` recommended item ids for a user, highest predicted
    rating first, excluding items already in `rated_item_ids`.

    Cold-start behavior:
      - If `user_id` is outside the range the model was trained on (a
        brand-new/unseen user), we cannot use that user's latent vector
        (there isn't one). We fall back to a POPULARITY ranking: items
        sorted by (global_mean + item_bias) — i.e. "what does everyone
        tend to rate highly", which is a standard, sensible collaborative-
        filtering cold-start fallback. This never crashes.
      - If `user_id` is known but has very few ratings, the model still has
        a (possibly noisy/near-zero) latent vector for them from training,
        so we use the normal personalized path — predictions are simply
        less confident, which is expected and not treated as an error.
    """
    rated_set = set(rated_item_ids)
    candidates = [iid for iid in all_item_ids if iid not in rated_set]

    if user_id is None or user_id < 0 or user_id >= model.n_users:
        # Cold start: unseen user -> popularity fallback.
        scores = {iid: model.mu + model.b_i[iid] for iid in candidates}
    else:
        preds = model.predict_all_for_user(user_id)
        scores = {iid: float(preds[iid]) for iid in candidates}

    ranked = sorted(candidates, key=lambda iid: scores[iid], reverse=True)
    return ranked[:top_n]
