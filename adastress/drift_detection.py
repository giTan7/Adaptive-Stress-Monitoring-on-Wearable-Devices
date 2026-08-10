"""
Feature-distribution drift detection.

For each embedding dimension f and stress class y, a reference
distribution P(f | y) is estimated from the training data. A
training-time baseline D_train[f, y] measures the expected
class-conditional divergence for that feature.

During deployment, a rolling buffer of recent embeddings and predicted
classes is maintained. For each feature and predicted class, the runtime
distribution R(f | y_hat) is compared with the corresponding training
reference P(f | y_hat). A feature is marked as drifting when its runtime
divergence exceeds its training-time baseline.

Drift is confirmed only when the importance-weighted mass of drifting
features exceeds the mean feature importance. Feature importance is
derived from the classifier weights, preventing low-importance features
from triggering adaptation independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


def _histogram_prob(values: np.ndarray, bin_edges: np.ndarray, eps: float) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bin_edges)
    probs = counts.astype(np.float64) + eps
    return probs / probs.sum()


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


@dataclass
class DriftReport:
    is_drift: bool
    drifting_features: List[int]
    drifting_importance_mass: float
    mean_importance: float
    per_feature_kl: Dict[int, float]


class DriftDetector:
    """Tracks per-feature, per-class distributions and flags deployment drift."""

    def __init__(self, num_features: int, num_classes: int, num_bins: int = 20, eps: float = 1e-6):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.eps = eps

        self.bin_edges: Optional[np.ndarray] = None   # [num_features, num_bins+1]
        self.p_ref: Dict[int, Dict[int, np.ndarray]] = {}     # p_ref[class][feature] -> prob vector
        self.d_train: Dict[int, Dict[int, float]] = {}        # d_train[class][feature] -> baseline KL
        self.importance: Optional[np.ndarray] = None          # [num_features]

        self.runtime_embeddings: List[np.ndarray] = []
        self.runtime_preds: List[int] = []

    # ------------------------------------------------------------------ #
    # Reference (training-time) statistics
    # ------------------------------------------------------------------ #

    def fit_reference(self, embeddings: np.ndarray, labels: np.ndarray, classifier_importance: np.ndarray) -> None:
        """Builds P(f | y) and the D_train baseline from training embeddings.

        `embeddings` is [N, num_features] (encoder / fused output, not raw
        signal), `labels` is [N] integer class ids, and
        `classifier_importance` is [num_features], typically derived from
        the classifier's first-layer weights (see `feature_importance_from_classifier`).
        """
        self.importance = classifier_importance / (classifier_importance.sum() + 1e-12)

        lo = embeddings.min(axis=0)
        hi = embeddings.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        lo, hi = lo - 0.05 * span, hi + 0.05 * span
        self.bin_edges = np.stack([np.linspace(lo[f], hi[f], self.num_bins + 1) for f in range(self.num_features)])

        self.p_ref = {}
        self.d_train = {}

        for y in range(self.num_classes):
            mask_y = labels == y
            mask_not_y = ~mask_y
            self.p_ref[y] = {}
            self.d_train[y] = {}
            for f in range(self.num_features):
                edges = self.bin_edges[f]
                p_y = _histogram_prob(embeddings[mask_y, f], edges, self.eps)
                p_not_y = _histogram_prob(embeddings[mask_not_y, f], edges, self.eps)
                self.p_ref[y][f] = p_y
                self.d_train[y][f] = _kl_divergence(p_y, p_not_y)

    def refresh_reference(self, embeddings: np.ndarray, labels: np.ndarray, classifier_importance: np.ndarray) -> None:
        """Recompute the reference distributions after a continual-learning update.

        This is meant to be called right after an EWC adaptation cycle: the
        model has changed, so what "normal" looks like in embedding space
        has changed too, and future drift checks should be measured against
        the *updated* model rather than the original training snapshot.
        """
        self.fit_reference(embeddings, labels, classifier_importance)
        self.reset_runtime_buffer()

    # ------------------------------------------------------------------ #
    # Runtime monitoring
    # ------------------------------------------------------------------ #

    def reset_runtime_buffer(self) -> None:
        self.runtime_embeddings = []
        self.runtime_preds = []

    def push(self, embedding: np.ndarray, predicted_class: int, max_buffer: int = 1000) -> None:
        self.runtime_embeddings.append(embedding)
        self.runtime_preds.append(predicted_class)
        if len(self.runtime_embeddings) > max_buffer:
            self.runtime_embeddings.pop(0)
            self.runtime_preds.pop(0)

    def buffer_size(self) -> int:
        return len(self.runtime_embeddings)

    def check(self, min_samples_per_class: int = 10) -> DriftReport:
        """Runs the divergence check against the current runtime buffer."""
        if self.bin_edges is None or self.importance is None:
            raise RuntimeError("fit_reference() must be called before check().")

        embeddings = np.stack(self.runtime_embeddings)
        preds = np.array(self.runtime_preds)

        per_feature_kl: Dict[int, float] = {}
        drifting_features: List[int] = []

        for f in range(self.num_features):
            edges = self.bin_edges[f]
            kls_for_feature = []
            for y in range(self.num_classes):
                mask = preds == y
                if mask.sum() < min_samples_per_class:
                    continue
                r_fy = _histogram_prob(embeddings[mask, f], edges, self.eps)
                d_runtime = _kl_divergence(r_fy, self.p_ref[y][f])
                kls_for_feature.append((d_runtime, self.d_train[y][f]))

            if not kls_for_feature:
                continue

            # a feature drifts if runtime divergence exceeds its own
            # training-time baseline for at least one observed class
            worst_runtime, worst_baseline = max(kls_for_feature, key=lambda t: t[0] - t[1])
            per_feature_kl[f] = worst_runtime
            if worst_runtime > worst_baseline:
                drifting_features.append(f)

        drifting_mass = float(self.importance[drifting_features].sum()) if drifting_features else 0.0
        mean_importance = float(self.importance.mean())

        return DriftReport(
            is_drift=drifting_mass > mean_importance,
            drifting_features=drifting_features,
            drifting_importance_mass=drifting_mass,
            mean_importance=mean_importance,
            per_feature_kl=per_feature_kl,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "num_bins": self.num_bins,
            "eps": self.eps,
            "bin_edges": self.bin_edges,
            "p_ref": self.p_ref,
            "d_train": self.d_train,
            "importance": self.importance,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "DriftDetector":
        state = torch.load(path, weights_only=False)
        obj = cls(state["num_features"], state["num_classes"], state["num_bins"], state["eps"])
        obj.bin_edges = state["bin_edges"]
        obj.p_ref = state["p_ref"]
        obj.d_train = state["d_train"]
        obj.importance = state["importance"]
        return obj


def feature_importance_from_classifier(classifier: torch.nn.Module) -> np.ndarray:
    """Derives per-embedding-dimension importance from the classifier's first layer.

    A larger average absolute weight leaving a given input dimension means
    the classifier relies on that dimension more, so drift in that
    dimension is more consequential than drift in a dimension the
    classifier barely uses.
    """
    first_linear = None
    for module in classifier.modules():
        if isinstance(module, torch.nn.Linear):
            first_linear = module
            break
    if first_linear is None:
        raise ValueError("Classifier has no Linear layer to derive importance from.")
    weight = first_linear.weight.detach().cpu().numpy()  # [out_dim, in_dim]
    importance = np.abs(weight).mean(axis=0)
    return importance
