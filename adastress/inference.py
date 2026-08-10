"""
End-to-end inference on a single window: raw signal in, stress prediction out.

This is what actually runs on the edge device. It loads the frozen encoder,
scaler, and trained classifier (+ context gate, if used) once, then exposes
a `predict` method that takes a raw [C, T] window (or [N, C, T] batch) plus
optional context and returns predicted class, confidence, and full
probabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Union

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from .models import ContextGateFusion, MLPClassifier, StressModel, load_encoder
from .utils import get_device


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: List[float]
    class_names: List[str]


class StressPredictor:
    def __init__(
        self,
        encoder_ckpt: str,
        scaler_ckpt: str,
        meta_json: str,
        classifier_ckpt: str,
        device_preference: str = "auto",
    ):
        self.device = get_device(device_preference)

        with open(meta_json, "r") as f:
            self.meta = json.load(f)

        self.scaler = joblib.load(scaler_ckpt)

        ckpt = torch.load(classifier_ckpt, map_location=self.device, weights_only=False)
        self.classes: List[str] = [str(c) for c in ckpt["classes"]]
        context_dim = ckpt.get("context_dim", 0)
        input_channels = ckpt.get("input_channels", self.meta["input_channels"])

        encoder = load_encoder(encoder_ckpt, input_channels, self.meta.get("embed_dim", 256) // 2, self.device)

        gate = None
        if context_dim and ckpt.get("gate") is not None:
            gate = ContextGateFusion(encoder.output_dim, context_dim).to(self.device)
            gate.load_state_dict(ckpt["gate"])
            gate.eval()

        classifier = MLPClassifier(encoder.output_dim, len(self.classes)).to(self.device)
        classifier.load_state_dict(ckpt["classifier"])
        classifier.eval()

        self.model = StressModel(encoder, classifier, gate).to(self.device)
        self.context_dim = context_dim

    def _preprocess(self, raw_window: np.ndarray) -> torch.Tensor:
        """raw_window: [C, T] with channels in the same order as meta['columns']."""
        scaled = self.scaler.transform(raw_window.T).T  # scaler expects [T, C]
        return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        raw_window: np.ndarray,
        context: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        x = self._preprocess(raw_window).to(self.device)

        c = None
        if self.model.gate is not None:
            if context is None:
                context = np.zeros(self.context_dim, dtype=np.float32)
            c = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)

        logits = self.model(x, c)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))

        return PredictionResult(
            predicted_class=self.classes[idx],
            predicted_index=idx,
            confidence=float(probs[idx]),
            probabilities=probs.tolist(),
            class_names=self.classes,
        )
