#!/usr/bin/env python3
"""
Simulates deployment: streams windows through the model one at a time,
monitors for drift, and runs EWC-regularized adaptation whenever drift is
confirmed. This is the same `AdaptationLoop` a real edge deployment would
use; here it is fed from the CSV instead of a live sensor.

Labels used to trigger a real prompt on-device are simulated with a
`label_oracle` that looks up ground truth from the CSV, standing in for a
user's opportunistic EMA response.

Usage:
    python scripts/run_adaptation_demo.py --config configs/config.yaml
"""
import argparse
import os
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adastress.adaptation_loop import AdaptationLoop, AdaptationLoopConfig
from adastress.config import load_config
from adastress.datasets import prepare_labelled_dataset
from adastress.drift_detection import DriftDetector, feature_importance_from_classifier
from adastress.models import ContextGateFusion, MLPClassifier, StressModel, load_encoder
from adastress.utils import ensure_dir, get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num-windows", type=int, default=200, help="How many windows to stream through the demo.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    scaler = joblib.load(cfg.paths.scaler_ckpt)
    use_context = cfg.classifier.use_context_gate

    X, y, C, subjects, classes = prepare_labelled_dataset(
        csv_path=cfg.data.labelled_csv,
        signal_columns=cfg.data.signal_columns,
        label_column=cfg.data.label_column,
        context_columns=cfg.data.context_columns if use_context else None,
        scaler=scaler,
        window_size=cfg.window.window_sec * cfg.window.fs,
        stride=cfg.window.stride_sec * cfg.window.fs,
        subject_column=cfg.data.subject_column,
    )
    print(f"Streaming from {X.shape[0]} available windows ({len(classes)} classes: {classes})")

    context_dim = C.shape[1] if use_context and C is not None else 0

    encoder = load_encoder(cfg.paths.encoder_ckpt, X.shape[1], cfg.ssl.lstm_hidden, device)
    gate = ContextGateFusion(encoder.output_dim, context_dim).to(device) if context_dim else None
    classifier = MLPClassifier(encoder.output_dim, len(classes), dropout=cfg.classifier.dropout).to(device)

    classifier_ckpt_path = cfg.paths.classifier_ckpt
    if os.path.exists(classifier_ckpt_path):
        ckpt = torch.load(classifier_ckpt_path, map_location=device, weights_only=False)
        classifier.load_state_dict(ckpt["classifier"])
        if gate is not None and ckpt.get("gate") is not None:
            gate.load_state_dict(ckpt["gate"])
        print(f"Loaded trained classifier from {classifier_ckpt_path}")
    else:
        print("No trained classifier checkpoint found, using a freshly initialized one "
              "(train_classifier.py first for a meaningful demo).")

    model = StressModel(encoder, classifier, gate).to(device)

    # Build the initial drift reference from a held-out slice of the same data.
    ref_n = min(len(X), 300)
    with torch.no_grad():
        ref_x = torch.tensor(X[:ref_n], dtype=torch.float32).to(device)
        ref_c = torch.tensor(C[:ref_n], dtype=torch.float32).to(device) if context_dim else None
        ref_h = model.embed(ref_x)
        ref_fused = model.gate(ref_h, ref_c) if model.gate is not None else ref_h
        ref_logits = model.classifier(ref_fused)
        ref_preds = ref_logits.argmax(dim=1).cpu().numpy()

    detector = DriftDetector(num_features=encoder.output_dim, num_classes=len(classes), num_bins=cfg.drift.num_bins)
    importance = feature_importance_from_classifier(classifier)
    detector.fit_reference(ref_fused.cpu().numpy(), ref_preds, importance)

    def label_oracle(windows_batch: np.ndarray) -> np.ndarray:
        # Demo stand-in for opportunistic user prompts: looks up ground truth.
        # A real deployment would instead collect a handful of EMA responses.
        n = windows_batch.shape[0]
        idx = np.random.choice(len(y), size=n, replace=True)
        return y[idx]

    loop_cfg = AdaptationLoopConfig(
        check_every_n_windows=cfg.drift.check_every_n_windows,
        min_buffer_for_check=cfg.drift.min_buffer_for_check,
        method=cfg.continual_learning.method,
        ewc_lambda=cfg.continual_learning.ewc_lambda,
        l2_lambda=cfg.continual_learning.l2_lambda,
        adapt_epochs=cfg.continual_learning.adapt_epochs,
        adapt_lr=cfg.continual_learning.adapt_lr,
        adapt_batch_size=cfg.continual_learning.adapt_batch_size,
        fisher_num_batches=cfg.continual_learning.fisher_num_batches,
        fisher_path=cfg.paths.fisher_ckpt,
        theta_star_path=cfg.paths.theta_star_ckpt,
    )
    ensure_dir(cfg.paths.checkpoints_dir)

    loop = AdaptationLoop(model, detector, device, loop_cfg, label_oracle, has_context=bool(context_dim))

    n_stream = min(args.num_windows, X.shape[0])
    for i in range(n_stream):
        x = torch.tensor(X[i : i + 1], dtype=torch.float32)
        c = torch.tensor(C[i : i + 1], dtype=torch.float32) if context_dim else None
        event = loop.process_window(x, c)
        if event is not None:
            print(f"\n*** Adaptation event at window {event.window_index} ***")
            print(f"    drifting features: {len(event.report.drifting_features)}")
            print(f"    importance mass: {event.report.drifting_importance_mass:.4f} "
                  f"(threshold: {event.report.mean_importance:.4f})")

    detector.save(cfg.paths.drift_reference_ckpt)
    print(f"\nStreamed {n_stream} windows, {len(loop.events)} adaptation event(s) triggered.")
    print(f"Drift reference state saved to {cfg.paths.drift_reference_ckpt}")


if __name__ == "__main__":
    main()
