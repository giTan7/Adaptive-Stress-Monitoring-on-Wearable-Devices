"""
Elastic Weight Consolidation (EWC) for drift-triggered adaptation.

Only ``model.adaptable_parameters()`` are updated during continual
adaptation; the pretrained encoder remains frozen.

Adaptation proceeds as follows:

1. ``compute_fisher(...)`` estimates diagonal Fisher information for the
   adaptable parameters using labeled adaptation data.
2. ``save_fisher(...)`` stores the Fisher information and corresponding
   parameter values (theta*).
3. When drift is detected, ``adapt(...)`` performs a small number of
   EWC-regularized updates using the adaptation buffer.
4. After adaptation, the Fisher information and theta* are recomputed
   for the updated model, and the drift detector reference distributions
   are refreshed.

The final step ensures that subsequent drift detection uses the adapted
model state as the new reference. This update sequence is orchestrated
by ``adaptation_loop.py``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

Fisher = Dict[str, torch.Tensor]
ThetaStar = Dict[str, torch.Tensor]


def _named_adaptable(model) -> Dict[str, nn.Parameter]:
    named = {}
    for name, p in model.classifier.named_parameters():
        named[f"classifier.{name}"] = p
    if model.gate is not None:
        for name, p in model.gate.named_parameters():
            named[f"gate.{name}"] = p
    return named


def named_adaptable_parameters(model) -> Dict[str, nn.Parameter]:
    """Public alias of `_named_adaptable`, so callers outside this module
    (e.g. the adaptation loop, when bootstrapping or refreshing theta*)
    build Fisher/theta* dictionaries with exactly the same keys this module
    uses internally. Do not reconstruct this naming scheme by hand elsewhere,
    a key mismatch here silently makes the EWC penalty a no-op.
    """
    return _named_adaptable(model)


@torch.no_grad()
def _snapshot(model) -> ThetaStar:
    return {name: p.detach().clone() for name, p in _named_adaptable(model).items()}


def compute_fisher(model, loader: DataLoader, device: torch.device, max_batches: int = 50) -> Fisher:
    """Estimates the diagonal empirical Fisher information for the adaptable
    parameters, using the squared gradient of the log-likelihood of the
    model's own (or true, if available) predictions.
    """
    named_params = _named_adaptable(model)
    fisher = {name: torch.zeros_like(p) for name, p in named_params.items()}

    model.classifier.eval()
    if model.gate is not None:
        model.gate.eval()

    n_batches = 0
    for batch in loader:
        if len(batch) == 3:
            xb, cb, yb = batch
            cb = cb.to(device)
        else:
            xb, yb = batch
            cb = None
        xb, yb = xb.to(device), yb.to(device)

        model.zero_grad()
        logits = model(xb, cb)
        log_probs = F.log_softmax(logits, dim=1)
        # empirical Fisher: use the model's own predicted class log-likelihood
        picked = log_probs.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)
        loss = -picked.mean()
        loss.backward()

        for name, p in named_params.items():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2

        n_batches += 1
        if n_batches >= max_batches:
            break

    for name in fisher:
        fisher[name] /= max(n_batches, 1)

    return fisher


def save_fisher(fisher: Fisher, theta_star: ThetaStar, fisher_path: str, theta_path: str) -> None:
    torch.save(fisher, fisher_path)
    torch.save(theta_star, theta_path)


def load_fisher(fisher_path: str, theta_path: str) -> tuple[Fisher, ThetaStar]:
    fisher = torch.load(fisher_path, weights_only=False)
    theta_star = torch.load(theta_path, weights_only=False)
    return fisher, theta_star


def ewc_penalty(model, fisher: Fisher, theta_star: ThetaStar) -> torch.Tensor:
    named_params = _named_adaptable(model)
    penalty = torch.tensor(0.0, device=next(iter(named_params.values())).device)
    for name, p in named_params.items():
        if name in fisher and name in theta_star:
            penalty = penalty + (fisher[name] * (p - theta_star[name]) ** 2).sum()
    return penalty


def adapt(
    model,
    adaptation_loader: DataLoader,
    device: torch.device,
    method: str = "ewc",
    fisher: Optional[Fisher] = None,
    theta_star: Optional[ThetaStar] = None,
    ewc_lambda: float = 400.0,
    l2_lambda: float = 0.01,
    epochs: int = 5,
    lr: float = 1e-4,
) -> Dict[str, List[float]]:
    """Runs a short continual-learning update on the adaptable parameters.

    `method` selects the regularization strategy compared in the paper:
        - "ewc":      Fisher-weighted penalty toward theta_star (default)
        - "l2":       plain, unweighted L2 penalty toward theta_star
        - "finetune": no penalty at all, standard fine-tuning
    """
    if method == "ewc" and (fisher is None or theta_star is None):
        raise ValueError("EWC requires both `fisher` and `theta_star`.")
    if method == "l2" and theta_star is None:
        raise ValueError("L2 regularization requires `theta_star`.")

    model.classifier.train()
    if model.gate is not None:
        model.gate.train()

    optimizer = torch.optim.Adam(model.adaptable_parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "ce_loss": [], "penalty": []}

    for epoch in range(epochs):
        epoch_loss, epoch_ce, epoch_pen = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in adaptation_loader:
            if len(batch) == 3:
                xb, cb, yb = batch
                cb = cb.to(device)
            else:
                xb, yb = batch
                cb = None
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb, cb)
            ce_loss = criterion(logits, yb)

            if method == "ewc":
                penalty = (ewc_lambda / 2) * ewc_penalty(model, fisher, theta_star)
            elif method == "l2":
                named_params = _named_adaptable(model)
                penalty = (l2_lambda / 2) * sum(
                    (p - theta_star[name]).pow(2).sum() for name, p in named_params.items()
                )
            else:  # plain fine-tuning
                penalty = torch.tensor(0.0, device=device)

            loss = ce_loss + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_pen += float(penalty.detach())
            n_batches += 1

        history["loss"].append(epoch_loss / max(n_batches, 1))
        history["ce_loss"].append(epoch_ce / max(n_batches, 1))
        history["penalty"].append(epoch_pen / max(n_batches, 1))
        print(
            f"  [adapt] epoch {epoch + 1}/{epochs} "
            f"loss={history['loss'][-1]:.4f} ce={history['ce_loss'][-1]:.4f} penalty={history['penalty'][-1]:.4f}"
        )

    return history
