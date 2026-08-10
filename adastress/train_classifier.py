"""
Stage 2: train the stress classifier (and, optionally, the context gate) on
top of the frozen, pretrained encoder.

Usage:
    python -m adastress.train_classifier --config configs/config.yaml --seed 42
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import load_config
from .datasets import LabelledWindowDataset, prepare_labelled_dataset
from .models import ContextGateFusion, Encoder, MLPClassifier, StressModel, load_encoder
from .utils import ensure_dir, save_json, set_seed


def build_model(cfg, input_channels: int, num_classes: int, context_dim: int, device) -> StressModel:
    encoder = load_encoder(cfg.paths.encoder_ckpt, input_channels, cfg.ssl.lstm_hidden, device)

    gate = None
    if cfg.classifier.use_context_gate and context_dim > 0:
        gate = ContextGateFusion(encoder.output_dim, context_dim).to(device)

    classifier = MLPClassifier(encoder.output_dim, num_classes, dropout=cfg.classifier.dropout).to(device)
    return StressModel(encoder, classifier, gate).to(device)


def evaluate(model: StressModel, loader, device, has_context: bool):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if has_context:
                xb, cb, yb = batch
                cb = cb.to(device)
            else:
                xb, yb = batch
                cb = None
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb, cb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


def run(cfg, seed: Optional[int] = None, save: bool = True) -> dict:
    seed = seed if seed is not None else cfg.seed
    set_seed(seed)
    device = torch.device("cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

    scaler = joblib.load(cfg.paths.scaler_ckpt)

    X, y, C, subjects, classes = prepare_labelled_dataset(
        csv_path=cfg.data.labelled_csv,
        signal_columns=cfg.data.signal_columns,
        label_column=cfg.data.label_column,
        context_columns=cfg.data.context_columns if cfg.classifier.use_context_gate else None,
        scaler=scaler,
        window_size=cfg.window.window_sec * cfg.window.fs,
        stride=cfg.window.stride_sec * cfg.window.fs,
        subject_column=cfg.data.subject_column,
    )

    has_context = C is not None
    context_dim = C.shape[1] if has_context else 0
    dataset = LabelledWindowDataset(X, y, C)

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.classifier.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.classifier.batch_size)

    model = build_model(cfg, input_channels=X.shape[1], num_classes=len(classes), context_dim=context_dim, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.adaptable_parameters(), lr=cfg.classifier.lr)

    best_f1 = -1.0
    best_state = None

    for epoch in range(cfg.classifier.epochs):
        model.classifier.train()
        if model.gate is not None:
            model.gate.train()

        for batch in tqdm(train_loader, desc=f"Classifier epoch {epoch + 1}/{cfg.classifier.epochs}"):
            if has_context:
                xb, cb, yb = batch
                cb = cb.to(device)
            else:
                xb, yb = batch
                cb = None
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb, cb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc, f1 = evaluate(model, val_loader, device, has_context)
        print(f"Epoch {epoch + 1}: val_acc={acc:.4f} val_macro_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                "classifier": model.classifier.state_dict(),
                "gate": model.gate.state_dict() if model.gate is not None else None,
            }

    model.classifier.load_state_dict(best_state["classifier"])
    if model.gate is not None and best_state["gate"] is not None:
        model.gate.load_state_dict(best_state["gate"])

    final_acc, final_f1 = evaluate(model, val_loader, device, has_context)

    if save:
        ensure_dir(cfg.paths.checkpoints_dir)
        torch.save(
            {
                "classifier": model.classifier.state_dict(),
                "gate": model.gate.state_dict() if model.gate is not None else None,
                "classes": classes,
                "context_dim": context_dim,
                "input_channels": X.shape[1],
            },
            cfg.paths.classifier_ckpt,
        )
        print(f"Saved classifier checkpoint to {cfg.paths.classifier_ckpt}")

    return {"seed": seed, "val_accuracy": final_acc, "val_macro_f1": final_f1, "classes": classes}


def main():
    parser = argparse.ArgumentParser(description="Train the stress classifier on top of the frozen encoder.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    result = run(load_config(args.config), seed=args.seed)
    print(result)


if __name__ == "__main__":
    main()
