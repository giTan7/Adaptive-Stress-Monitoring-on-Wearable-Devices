"""
Deployment loop for drift detection, continual adaptation, and reference refresh.

For each incoming window, the model performs inference and updates the drift
buffer. At the configured interval, the drift detector evaluates the recent
stream. If drift is confirmed, a labeled adaptation buffer is collected,
the adaptable parameters are updated using the selected continual-learning
method, and the Fisher information, reference parameters, and drift
distributions are recomputed from the updated model.

The same adaptation path is used for deployment and offline simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .continual_learning import adapt, compute_fisher, load_fisher, named_adaptable_parameters, save_fisher
from .drift_detection import DriftDetector, DriftReport, feature_importance_from_classifier


@dataclass
class AdaptationEvent:
    window_index: int
    report: DriftReport
    adaptation_history: dict


@dataclass
class AdaptationLoopConfig:
    check_every_n_windows: int = 10
    min_buffer_for_check: int = 50
    method: str = "ewc"
    ewc_lambda: float = 400.0
    l2_lambda: float = 0.01
    adapt_epochs: int = 5
    adapt_lr: float = 1e-4
    adapt_batch_size: int = 32
    fisher_num_batches: int = 50
    fisher_path: str = "checkpoints/fisher.pt"
    theta_star_path: str = "checkpoints/theta_star.pt"


LabelOracle = Callable[[np.ndarray], np.ndarray]
"""Given a batch of raw windows [N, C, T] that triggered drift, return integer
labels [N]. In a real deployment this collects opportunistic EMA responses
from the user; in the bundled demo it looks up ground truth for simulation."""


class AdaptationLoop:
    def __init__(
        self,
        model,
        drift_detector: DriftDetector,
        device: torch.device,
        cfg: AdaptationLoopConfig,
        label_oracle: LabelOracle,
        has_context: bool = False,
    ):
        self.model = model
        self.drift_detector = drift_detector
        self.device = device
        self.cfg = cfg
        self.label_oracle = label_oracle
        self.has_context = has_context

        self.window_buffer: List[np.ndarray] = []
        self.context_buffer: List[Optional[np.ndarray]] = []
        self.events: List[AdaptationEvent] = []
        self._window_count = 0

        self.fisher, self.theta_star = None, None
        try:
            self.fisher, self.theta_star = load_fisher(cfg.fisher_path, cfg.theta_star_path)
        except FileNotFoundError:
            pass  # first run: no Fisher/theta* yet, computed lazily below

    @torch.no_grad()
    def _embed_and_predict(self, x: torch.Tensor, c: Optional[torch.Tensor]):
        h = self.model.embed(x)
        fused = self.model.gate(h, c) if self.model.gate is not None else h
        logits = self.model.classifier(fused)
        preds = logits.argmax(dim=1)
        return fused.cpu().numpy(), preds.cpu().numpy()

    def process_window(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> Optional[AdaptationEvent]:
        """Feed one new window through the model, update the drift buffer,
        and every `check_every_n_windows` windows, run the divergence check
        (triggering an adaptation cycle if drift is flagged).
        """
        x = x.to(self.device)
        c = c.to(self.device) if c is not None else None

        embeddings, preds = self._embed_and_predict(x, c)
        for i in range(x.size(0)):
            self.drift_detector.push(embeddings[i], int(preds[i]))
            self.window_buffer.append(x[i].cpu().numpy())
            self.context_buffer.append(c[i].cpu().numpy() if c is not None else None)
            self._window_count += 1

        if (
            self._window_count % self.cfg.check_every_n_windows == 0
            and self.drift_detector.buffer_size() >= self.cfg.min_buffer_for_check
        ):
            report = self.drift_detector.check()
            if report.is_drift:
                history = self._run_adaptation_cycle()
                event = AdaptationEvent(self._window_count, report, history)
                self.events.append(event)
                return event
        return None

    def _build_adaptation_loader(self) -> DataLoader:
        windows = np.stack(self.window_buffer[-self.cfg.min_buffer_for_check :])
        raw_labels = self.label_oracle(windows)

        X = torch.tensor(windows, dtype=torch.float32)
        y = torch.tensor(raw_labels, dtype=torch.long)

        if self.has_context:
            contexts = self.context_buffer[-self.cfg.min_buffer_for_check :]
            C = torch.tensor(np.stack(contexts), dtype=torch.float32)
            dataset = TensorDataset(X, C, y)
        else:
            dataset = TensorDataset(X, y)

        return DataLoader(dataset, batch_size=self.cfg.adapt_batch_size, shuffle=True)

    def _run_adaptation_cycle(self) -> dict:
        print(f"[adaptation_loop] drift confirmed at window {self._window_count}, starting EWC adaptation")

        adaptation_loader = self._build_adaptation_loader()

        if self.cfg.method == "ewc" and (self.fisher is None or self.theta_star is None):
            # first adaptation cycle ever: bootstrap Fisher / theta* from the
            # current stable model before making any update
            self.fisher = compute_fisher(self.model, adaptation_loader, self.device, self.cfg.fisher_num_batches)
            self.theta_star = {name: p.detach().clone() for name, p in named_adaptable_parameters(self.model).items()}

        history = adapt(
            self.model,
            adaptation_loader,
            self.device,
            method=self.cfg.method,
            fisher=self.fisher,
            theta_star=self.theta_star,
            ewc_lambda=self.cfg.ewc_lambda,
            l2_lambda=self.cfg.l2_lambda,
            epochs=self.cfg.adapt_epochs,
            lr=self.cfg.adapt_lr,
        )

        # ---- update params, THEN recompute Fisher / theta*, THEN refresh
        # ---- the divergence reference before monitoring resumes ----
        self.fisher = compute_fisher(self.model, adaptation_loader, self.device, self.cfg.fisher_num_batches)
        self.theta_star = {name: p.detach().clone() for name, p in named_adaptable_parameters(self.model).items()}
        save_fisher(self.fisher, self.theta_star, self.cfg.fisher_path, self.cfg.theta_star_path)

        refresh_embeddings, refresh_preds = [], []
        for batch in adaptation_loader:
            if self.has_context:
                xb, cb, yb = batch
                cb = cb.to(self.device)
            else:
                xb, yb = batch
                cb = None
            xb = xb.to(self.device)
            emb, preds = self._embed_and_predict(xb, cb)
            refresh_embeddings.append(emb)
            refresh_preds.append(preds)

        refresh_embeddings = np.concatenate(refresh_embeddings, axis=0)
        refresh_preds = np.concatenate(refresh_preds, axis=0)
        importance = feature_importance_from_classifier(self.model.classifier)

        self.drift_detector.refresh_reference(refresh_embeddings, refresh_preds, importance)
        print("[adaptation_loop] model updated, Fisher/theta* saved, drift reference refreshed")

        return history
