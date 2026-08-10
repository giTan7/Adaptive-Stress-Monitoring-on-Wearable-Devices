"""
Windowing and PyTorch ``Dataset`` classes shared by SSL pretraining,
classifier training, and the deployment-time drift / adaptation loop.

This mirrors the windowing logic used to produce the released
``encoder.pth`` / ``scaler.pkl`` / ``meta.json`` artifacts, just organized
into reusable functions instead of being copy-pasted per notebook cell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class WindowMeta:
    columns: List[str]
    window_sec: int
    stride_sec: int
    fs: int
    window_size: int
    input_channels: int
    embed_dim: int


def load_meta(path: str) -> WindowMeta:
    import json

    with open(path, "r") as f:
        d = json.load(f)
    return WindowMeta(
        columns=d["columns"],
        window_sec=d["window_sec"],
        stride_sec=d["stride_sec"],
        fs=d["fs"],
        window_size=d["window_size"],
        input_channels=d["input_channels"],
        embed_dim=d.get("embed_dim", 256),
    )


def build_windows(arr: np.ndarray, window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a [T, C] array into overlapping [N, C, window] windows.

    Returns the windows plus the start index of each window in the
    original array, since callers often need the start index again to
    look up labels or context columns for the same span.
    """
    n = arr.shape[0]
    if n < window:
        raise ValueError(f"Signal length {n} is shorter than window size {window}")
    starts = np.arange(0, n - window + 1, stride)
    windows = np.stack([arr[s : s + window].T for s in starts])
    return windows, starts


def clean_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    sub = df[columns].copy()
    sub = sub.interpolate(limit_direction="both").fillna(0)
    return sub


def fit_scaler(df: pd.DataFrame, columns: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(clean_numeric(df, columns).values)
    return scaler


def apply_scaler(df: pd.DataFrame, columns: List[str], scaler: StandardScaler) -> np.ndarray:
    arr = scaler.transform(clean_numeric(df, columns).values)
    return pd.DataFrame(arr).interpolate(limit_direction="both").fillna(0).values


def window_labels(y_series: pd.Series, starts: np.ndarray, window: int) -> np.ndarray:
    """Majority-vote label within each window (falls back to the last sample)."""
    labels = []
    for s in starts:
        seg = y_series.iloc[s : s + window]
        mode = seg.mode()
        labels.append(mode.iloc[0] if len(mode) > 0 else seg.iloc[-1])
    return np.array(labels)


def window_context(context_df: pd.DataFrame, starts: np.ndarray, window: int) -> np.ndarray:
    """Majority-vote context values per window, one row per context column."""
    rows = []
    for s in starts:
        seg = context_df.iloc[s : s + window]
        row = [seg[c].mode().iloc[0] if len(seg[c].mode()) > 0 else seg[c].iloc[-1] for c in context_df.columns]
        rows.append(row)
    return np.array(rows)


class ContrastiveWindowDataset(Dataset):
    """Two randomly augmented views of the same window, for SSL pretraining."""

    def __init__(self, windows: np.ndarray, noise_std: float = 0.01, scale_jitter: float = 0.1):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.noise_std = noise_std
        self.scale_jitter = scale_jitter

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.noise_std * torch.randn_like(x)
        scale = (1.0 - self.scale_jitter) + 2 * self.scale_jitter * torch.rand(1)
        return x * scale + noise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        return self._augment(x), self._augment(x)


class LabelledWindowDataset(Dataset):
    """Physiological windows with a stress label and, optionally, context."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        context: Optional[np.ndarray] = None,
        subjects: Optional[np.ndarray] = None,
    ):
        self.X = torch.tensor(windows, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.C = torch.tensor(context, dtype=torch.float32) if context is not None else None
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.C is not None:
            return self.X[idx], self.C[idx], self.y[idx]
        return self.X[idx], self.y[idx]


def prepare_labelled_dataset(
    csv_path: str,
    signal_columns: List[str],
    label_column: str,
    context_columns: Optional[List[str]],
    scaler: StandardScaler,
    window_size: int,
    stride: int,
    subject_column: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List]:
    """Load a labelled CSV and turn it into model-ready windows.

    If a subject column is present, windowing is done independently per
    subject so a window never mixes signal from two different people.
    """
    df = pd.read_csv(csv_path)
    classes = sorted(df[label_column].dropna().unique().tolist())

    all_windows, all_labels, all_context, all_subjects = [], [], [], []

    groups = df.groupby(subject_column) if subject_column and subject_column in df.columns else [(None, df)]

    for subj, g in groups:
        g = g.reset_index(drop=True)
        arr = apply_scaler(g, signal_columns, scaler)
        if len(arr) < window_size:
            continue
        windows, starts = build_windows(arr, window_size, stride)
        labels = window_labels(g[label_column], starts, window_size)

        all_windows.append(windows)
        all_labels.append(labels)
        all_subjects.extend([subj] * len(windows))

        if context_columns:
            ctx = window_context(g[context_columns], starts, window_size)
            all_context.append(ctx)

    X = np.concatenate(all_windows, axis=0)
    y_raw = np.concatenate(all_labels, axis=0)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[v] for v in y_raw])

    C = np.concatenate(all_context, axis=0).astype(np.float32) if all_context else None
    subjects = np.array(all_subjects) if subject_column else None

    return X, y, C, subjects, classes


def prepare_unlabelled_windows(
    csv_path: str,
    signal_columns: List[str],
    window_size: int,
    stride: int,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Load raw signal for SSL pretraining, fitting a new scaler if one isn't given."""
    df = pd.read_csv(csv_path)
    if scaler is None:
        scaler = fit_scaler(df, signal_columns)
    arr = apply_scaler(df, signal_columns, scaler)
    windows, _ = build_windows(arr, window_size, stride)
    return windows, scaler
