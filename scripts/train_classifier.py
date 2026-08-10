#!/usr/bin/env python3
"""Thin wrapper so the pipeline can be run as `python scripts/train_classifier.py`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adastress.train_classifier import main

if __name__ == "__main__":
    main()
