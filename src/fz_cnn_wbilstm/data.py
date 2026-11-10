"""Chronological PVGIS-style time-series preparation, with train-only fitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
import torch


class FuzzyCMeans:
    """Euclidean fuzzy c-means; centroids learned exclusively on training rows."""

    def __init__(self, n_clusters: int = 3, fuzzifier: float = 2.0, tol: float = 1e-4,
                 max_iter: int = 2000, seed: int = 42):
        if n_clusters < 2 or fuzzifier <= 1:
            raise ValueError("n_clusters must be >= 2 and fuzzifier > 1")
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.centers_: np.ndarray | None = None
        self.iterations_ = 0

    def predict_membership(self, x: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("FCM has not been fitted")
        x = np.asarray(x, dtype=np.float64)
        distances = np.linalg.norm(x[:, None, :] - self.centers_[None, :, :], axis=-1)
        zero = distances <= 1e-10
        safe = np.maximum(distances, 1e-10)
        powers = safe ** (-2.0 / (self.fuzzifier - 1.0))
        membership = powers / powers.sum(axis=1, keepdims=True)
        # Exactly coincident samples must have deterministic, finite memberships.
        for idx in np.where(zero.any(axis=1))[0]:
            membership[idx] = zero[idx] / zero[idx].sum()
        return membership

    def fit(self, x: np.ndarray) -> "FuzzyCMeans":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < self.n_clusters or not np.isfinite(x).all():
            raise ValueError("FCM requires a finite 2D array with at least k samples")
        rng = np.random.default_rng(self.seed)
        self.centers_ = x[rng.choice(len(x), self.n_clusters, replace=False)].copy()
        for iteration in range(1, self.max_iter + 1):
            previous = self.predict_membership(x)
            powers = previous ** self.fuzzifier
            new_centers = powers.T @ x / np.maximum(powers.sum(axis=0)[:, None], 1e-12)
            self.centers_ = new_centers
            updated = self.predict_membership(x)
            membership_shift = float(np.sum((updated - previous) ** 2))
            self.iterations_ = iteration
            if membership_shift < self.tol:
                break
        return self


@dataclass
class DataSplit:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    test_time: list[str]
    y_min: float
    y_scale: float
    metadata: dict

    def inverse_y(self, values):
        return np.asarray(values) * self.y_scale + self.y_min


def make_windows(df: pd.DataFrame, features: Sequence[str], target: str = "GI",
                 lookback: int = 96, train_fraction: float = .8,
                 val_share_train: float = .15, n_clusters: int = 3,
                 seed: int = 42, time_col: str = "time") -> DataSplit:
    if not (0 < val_share_train < .5 and .5 <= train_fraction < 1):
        raise ValueError("Invalid split fractions")
    if lookback < 27:
        raise ValueError("The three pool-3 CNN blocks require lookback >= 27")
    df = df.copy()
    if time_col in df:
        if df[time_col].isna().any() or df[time_col].duplicated().any():
            raise ValueError("Timestamps must be complete and unique")
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    required = list(dict.fromkeys(list(features) + [target]))
    absent = [c for c in required if c not in df]
    if absent:
        raise ValueError(f"Missing dataset columns: {absent}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[required].isna().any().any() or not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("Data contain NaN or non-finite values; clean before partitioning")
    n = len(df)
    train_end = int(n * train_fraction)
    val_begin = int(train_end * (1.0 - val_share_train))
    if not (lookback + 8 < val_begin < train_end - 8 < n - 8):
        raise ValueError("Dataset is too short for these lookback and split settings")

    raw = df[required].to_numpy(dtype=np.float64)
    low = raw[:val_begin].min(axis=0)
    scale = np.maximum(raw[:val_begin].max(axis=0) - low, 1e-8)
    normalized = (raw - low) / scale
    # FCM sees only the observed predictor/lag channels. Target may occur only as
    # historical context inside a window, never as the same-step forecast label.
    fcm = FuzzyCMeans(n_clusters=n_clusters, tol=1e-4, max_iter=2000, seed=seed)
    fcm.fit(normalized[:val_begin])
    memberships = fcm.predict_membership(normalized).astype(np.float32)
    inputs = np.column_stack([normalized, memberships]).astype(np.float32)
    labels = normalized[:, required.index(target)].astype(np.float32)
    times = df[time_col].astype(str).tolist() if time_col in df else [str(i) for i in range(n)]

    X, y, indices = [], [], []
    for t in range(lookback, n):
        X.append(inputs[t - lookback:t])
        y.append(labels[t])
        indices.append(t)
    X = torch.from_numpy(np.stack(X))
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))
    indices = np.asarray(indices)
    masks = {
        "train": indices < val_begin,
        "val": (indices >= val_begin) & (indices < train_end),
        "test": indices >= train_end,
    }
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("Insufficient windows for a split")
    metadata = {
        "input_features": required,
        "fcm_memberships": n_clusters,
        "features_total": inputs.shape[1],
        "training_end_row_exclusive": val_begin,
        "validation_end_row_exclusive": train_end,
        "test_start_row": train_end,
        "lookback": lookback,
        "fcm_iterations": fcm.iterations_,
        "fcm_centroids": fcm.centers_.tolist(),
        "min_train": low.tolist(),
        "range_train": scale.tolist(),
        "split_strategy": "chronological 80/20 with held-out validation inside initial 80%",
    }
    return DataSplit(
        X[masks["train"]], y[masks["train"]],
        X[masks["val"]], y[masks["val"]],
        X[masks["test"]], y[masks["test"]],
        [times[i] for i in indices[masks["test"]]],
        float(low[required.index(target)]),
        float(scale[required.index(target)]), metadata,
    )


def demo_pvgis(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic signal for software tests only (not a substitute for PVGIS)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hour = t % 24
    elev = np.maximum(0, np.sin(np.pi * (hour - 6) / 12)) * 72
    humidity = np.clip(60 + 20 * np.cos(t / 60) + rng.normal(0, 3, n), 10, 95)
    wind = np.maximum(0, 3 + np.cos(t / 9) + rng.normal(0, .5, n))
    temperature = 28 + 7 * np.sin((hour - 8) * np.pi / 12) + rng.normal(0, 1.5, n)
    gi = np.maximum(0, 12 * elev * (1 - humidity / 300) + rng.normal(0, 15, n))
    return pd.DataFrame({"time": pd.date_range("2020-01-01", periods=n, freq="h"),
                         "GI": gi, "SE": elev, "AT": temperature, "H": humidity, "WV": wind})
