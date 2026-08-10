#!/usr/bin/env python3
"""
Loads the trained model and runs a single prediction on the first window
found in the sample data, purely to demonstrate the inference path end to
end. For a real deployment, replace `_load_demo_window` with however your
edge device is actually receiving live sensor data.

Usage:
    python scripts/run_inference.py --config configs/config.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adastress.config import load_config
from adastress.inference import StressPredictor


def _load_demo_window(cfg):
    df = pd.read_csv(cfg.data.labelled_csv)
    window_size = cfg.window.window_sec * cfg.window.fs
    seg = df[cfg.data.signal_columns].iloc[:window_size]
    return seg.values.T.astype(np.float32)  # [C, T]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    predictor = StressPredictor(
        encoder_ckpt=cfg.paths.encoder_ckpt,
        scaler_ckpt=cfg.paths.scaler_ckpt,
        meta_json=cfg.paths.meta_json,
        classifier_ckpt=cfg.paths.classifier_ckpt,
        device_preference=cfg.device,
    )

    window = _load_demo_window(cfg)
    result = predictor.predict(window)

    print("Predicted stress class:", result.predicted_class)
    print(f"Confidence: {result.confidence:.3f}")
    print("Class probabilities:")
    for name, p in zip(result.class_names, result.probabilities):
        print(f"  {name}: {p:.3f}")


if __name__ == "__main__":
    main()
