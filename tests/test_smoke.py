"""
End-to-end smoke test: trains a classifier for one epoch on the bundled
sample data, runs a drift check, and runs one EWC adaptation cycle. This is
not meant to produce a good model, only to catch broken imports/shapes
before you push.

Run with:  python -m pytest tests/ -v
"""
import copy
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
from adastress.train_classifier import run as train_classifier_run
from adastress.utils import get_device, set_seed

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")


def test_encoder_loads_and_runs():
    cfg = load_config(CONFIG_PATH)
    device = get_device("cpu")
    encoder = load_encoder(cfg.paths.encoder_ckpt, len(cfg.data.signal_columns), cfg.ssl.lstm_hidden, device)
    x = torch.randn(2, len(cfg.data.signal_columns), cfg.window.window_sec * cfg.window.fs)
    out = encoder(x)
    assert out.shape == (2, encoder.output_dim)


def test_classifier_trains_one_epoch():
    cfg = load_config(CONFIG_PATH)
    cfg.classifier.epochs = 1
    result = train_classifier_run(cfg, seed=0, save=True)
    assert "val_accuracy" in result
    assert 0.0 <= result["val_accuracy"] <= 1.0


def test_drift_and_adaptation_cycle_runs():
    cfg = load_config(CONFIG_PATH)
    set_seed(0)
    device = get_device("cpu")

    scaler = joblib.load(cfg.paths.scaler_ckpt)
    X, y, C, subjects, classes = prepare_labelled_dataset(
        csv_path=cfg.data.labelled_csv,
        signal_columns=cfg.data.signal_columns,
        label_column=cfg.data.label_column,
        context_columns=cfg.data.context_columns,
        scaler=scaler,
        window_size=cfg.window.window_sec * cfg.window.fs,
        stride=cfg.window.stride_sec * cfg.window.fs,
        subject_column=cfg.data.subject_column,
    )
    assert X.shape[0] > 0

    context_dim = C.shape[1]
    encoder = load_encoder(cfg.paths.encoder_ckpt, X.shape[1], cfg.ssl.lstm_hidden, device)
    gate = ContextGateFusion(encoder.output_dim, context_dim).to(device)
    classifier = MLPClassifier(encoder.output_dim, len(classes)).to(device)
    model = StressModel(encoder, classifier, gate).to(device)

    with torch.no_grad():
        ref_x = torch.tensor(X[:50], dtype=torch.float32)
        ref_c = torch.tensor(C[:50], dtype=torch.float32)
        h = model.embed(ref_x)
        fused = model.gate(h, ref_c)
        preds = model.classifier(fused).argmax(dim=1).numpy()

    detector = DriftDetector(num_features=encoder.output_dim, num_classes=len(classes), num_bins=10)
    importance = feature_importance_from_classifier(classifier)
    detector.fit_reference(fused.numpy(), preds, importance)

    # buffer is empty at this point; just confirm fit_reference produced usable state
    assert detector.importance is not None
    assert detector.bin_edges is not None

    def label_oracle(batch):
        idx = np.random.choice(len(y), size=batch.shape[0], replace=True)
        return y[idx]

    loop_cfg = AdaptationLoopConfig(
        check_every_n_windows=5,
        min_buffer_for_check=10,
        adapt_epochs=1,
        fisher_path="checkpoints/_test_fisher.pt",
        theta_star_path="checkpoints/_test_theta_star.pt",
    )
    loop = AdaptationLoop(model, detector, device, loop_cfg, label_oracle, has_context=True)

    n = min(30, X.shape[0])
    for i in range(n):
        x = torch.tensor(X[i : i + 1], dtype=torch.float32)
        c = torch.tensor(C[i : i + 1], dtype=torch.float32)
        loop.process_window(x, c)

    # cleanup test-only checkpoint files
    for p in [loop_cfg.fisher_path, loop_cfg.theta_star_path]:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    test_encoder_loads_and_runs()
    test_classifier_trains_one_epoch()
    test_drift_and_adaptation_cycle_runs()
    print("All smoke tests passed.")
