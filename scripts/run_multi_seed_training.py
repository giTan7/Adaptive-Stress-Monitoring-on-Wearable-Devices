#!/usr/bin/env python3
"""
Trains the classifier under several random seeds, with and without the
context gate, and writes a results CSV that `adastress.stats_tests` can
consume directly.

Usage:
    python scripts/run_multi_seed_training.py --config configs/config.yaml
"""
import argparse
import copy
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adastress.config import load_config
from adastress.train_classifier import run
from adastress.utils import ensure_dir

SEEDS = [42, 123, 456, 789, 2026]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="results/classifier_runs.csv")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    rows = []

    for use_gate, method_name in [(True, "with_context_gate"), (False, "no_context_gate")]:
        for seed in SEEDS:
            cfg = copy.deepcopy(base_cfg)
            cfg.classifier.use_context_gate = use_gate
            print(f"\n=== method={method_name} seed={seed} ===")
            result = run(cfg, seed=seed, save=False)
            rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "dataset": os.path.basename(cfg.data.labelled_csv),
                    "accuracy": result["val_accuracy"],
                    "macro_f1": result["val_macro_f1"],
                }
            )

    ensure_dir(os.path.dirname(args.out) or ".")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {len(rows)} rows to {args.out}")
    print("Run `python -m adastress.stats_tests --results", args.out, "` to test significance.")


if __name__ == "__main__":
    main()
