#!/usr/bin/env python3
"""Thin wrapper so stats tests can be run as `python scripts/run_stats_tests.py`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adastress.stats_tests import main

if __name__ == "__main__":
    main()
