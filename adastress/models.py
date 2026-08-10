"""
Model definitions.

``Encoder`` is written to match ``checkpoints/encoder.pth`` layer for layer,
so that checkpoint loads directly with ``strict=True``. If you retrain the
encoder from scratch with a different width or depth, update both this file
and ``configs/config.yaml`` together.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Conv1D stack + bidirectional LSTM physiological signal encoder.

    Input:  [B, C, T]  (C physiological channels, T timesteps per window)
    Output: [B, 2 * lstm_hidden]
    """

    def __init__(self, in_channels: int, lstm_hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = 2 * lstm_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)              # [B, 128, T]
        h = h.permute(0, 2, 1)        # [B, T, 128]
        h, _ = self.lstm(h)           # [B, T, 2*hidden]
        h = h.mean(dim=1)             # temporal mean pooling -> [B, 2*hidden]
        return h


class ProjectionHead(nn.Module):
    """SSL contrastive projection head, discarded after pretraining."""

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1)


class ContextGateFusion(nn.Module):
    """Learned gate that blends the physiological embedding with context.

        g = sigmoid(W_g [h; c] + b_g)
        fused = g * h + (1 - g) * c_proj

    Context is projected into the same dimensionality as the physiological
    embedding first so the elementwise blend is well defined. If no context
    is available for a given sample, pass a zero vector; the gate will
    naturally learn to lean on the physiological embedding in that case.
    """

    def __init__(self, embed_dim: int, context_dim: int):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_proj = self.context_proj(c)
        g = torch.sigmoid(self.gate(torch.cat([h, c_proj], dim=1)))
        fused = g * h + (1 - g) * c_proj
        return fused

    def gate_values(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Returns the raw gate activations, useful for interpretability plots."""
        c_proj = self.context_proj(c)
        return torch.sigmoid(self.gate(torch.cat([h, c_proj], dim=1)))


class MLPClassifier(nn.Module):
    """Three-layer stress classifier sitting on top of the frozen embedding."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StressModel(nn.Module):
    """Deployable stress model with a frozen encoder and adaptable downstream layers.

    During continual adaptation, the encoder remains frozen. Only the
    context-fusion module and classifier are updated.
    """

    def __init__(
        self,
        encoder: Encoder,
        classifier: MLPClassifier,
        gate: Optional[ContextGateFusion] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.gate = gate
        self.classifier = classifier

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def adaptable_parameters(self):
        """Only the parameters continual learning is allowed to touch."""
        params = list(self.classifier.parameters())
        if self.gate is not None:
            params += list(self.gate.parameters())
        return params

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed(x)

        if self.gate is not None:
            if context is None:
                context = torch.zeros(
                    h.size(0),
                    self.gate.context_proj.in_features,
                    device=h.device,
                )
            h = self.gate(h, context)

        return self.classifier(h)


def load_encoder(ckpt_path: str, in_channels: int, lstm_hidden: int, device: torch.device) -> Encoder:
    """Loads ``encoder.pth``, handling both raw state_dict and wrapped ('encoder' key) formats."""
    encoder = Encoder(in_channels, lstm_hidden).to(device)
    raw = torch.load(ckpt_path, map_location=device)
    state_dict = raw["encoder"] if isinstance(raw, dict) and "encoder" in raw else raw
    encoder.load_state_dict(state_dict)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder
