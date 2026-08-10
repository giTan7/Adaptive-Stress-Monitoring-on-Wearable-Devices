"""
Statistical tests used to back up every comparison made during training and
evaluation: frozen encoder vs. joint fine-tuning, EWC vs. plain fine-tuning
vs. L2, hybrid vs. full personalization, and so on.

Nothing here is specific to one experiment. Every function takes plain
arrays of per-seed (or per-subject) metric values and returns a small,
serializable result. `run_from_results_csv` is the convenience entry point:
point it at a CSV produced by running `train_classifier.run(...)` under a
few different seeds/methods and it reports every pairwise and omnibus test
that applies.

Usage as a script:
    python -m adastress.stats_tests --results results/classifier_runs.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    test: str
    comparison: str
    statistic: float
    p_value: float
    significant_at_0_05: bool
    notes: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


def paired_ttest(a: Sequence[float], b: Sequence[float], label_a: str, label_b: str) -> TestResult:
    t, p = stats.ttest_rel(a, b)
    return TestResult(
        test="paired_t_test",
        comparison=f"{label_a} vs {label_b}",
        statistic=float(t),
        p_value=float(p),
        significant_at_0_05=bool(p < 0.05),
    )


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], label_a: str, label_b: str) -> TestResult:
    stat, p = stats.wilcoxon(a, b)
    return TestResult(
        test="wilcoxon_signed_rank",
        comparison=f"{label_a} vs {label_b}",
        statistic=float(stat),
        p_value=float(p),
        significant_at_0_05=bool(p < 0.05),
        notes="Nonparametric alternative to the paired t-test; use when normality is doubtful.",
    )


def levene_variance_test(a: Sequence[float], b: Sequence[float], label_a: str, label_b: str) -> TestResult:
    stat, p = stats.levene(a, b)
    return TestResult(
        test="levene",
        comparison=f"variance({label_a}) vs variance({label_b})",
        statistic=float(stat),
        p_value=float(p),
        significant_at_0_05=bool(p < 0.05),
        notes="Significant result means the two methods differ in how consistent they are across seeds/subjects, not just in mean.",
    )


def friedman_test(*groups: Sequence[float], labels: List[str]) -> TestResult:
    stat, p = stats.friedmanchisquare(*groups)
    return TestResult(
        test="friedman",
        comparison=" vs ".join(labels),
        statistic=float(stat),
        p_value=float(p),
        significant_at_0_05=bool(p < 0.05),
        notes="Omnibus test across 3+ related groups (e.g. EWC vs fine-tune vs L2 across the same seeds).",
    )


def bonferroni_correction(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def pearson_correlation(a: Sequence[float], b: Sequence[float], label: str) -> TestResult:
    r, p = stats.pearsonr(a, b)
    return TestResult(
        test="pearson_correlation",
        comparison=label,
        statistic=float(r),
        p_value=float(p),
        significant_at_0_05=bool(p < 0.05),
    )


def confidence_interval_diff(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """95% CI on the paired difference (a - b), the same style of interval
    used to check whether an apparent accuracy gain is actually distinguishable
    from zero rather than only reporting the point estimate.
    """
    diff = np.asarray(a) - np.asarray(b)
    mean = diff.mean()
    sem = stats.sem(diff)
    ci = stats.t.interval(confidence, len(diff) - 1, loc=mean, scale=sem) if len(diff) > 1 else (mean, mean)
    return {
        "mean_diff": float(mean),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "confidence": float(confidence),
    }



# Convenience entry point for a results CSV with columns:
#   seed, method, dataset, accuracy, macro_f1


def run_from_results_csv(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    required = {"seed", "method", "accuracy", "macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results CSV is missing required columns: {missing}")

    methods = sorted(df["method"].unique())
    results: List[Dict] = []

    # pairwise paired t-tests + Levene's test on accuracy and macro-F1
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            sub1 = df[df["method"] == m1].sort_values("seed")
            sub2 = df[df["method"] == m2].sort_values("seed")
            if len(sub1) != len(sub2) or len(sub1) < 2:
                continue

            for metric in ["accuracy", "macro_f1"]:
                a, b = sub1[metric].values, sub2[metric].values
                results.append(paired_ttest(a, b, f"{m1}:{metric}", f"{m2}:{metric}").as_dict())
                results.append(levene_variance_test(a, b, f"{m1}:{metric}", f"{m2}:{metric}").as_dict())
                results.append(confidence_interval_diff(a, b) | {"comparison": f"{m1} - {m2} ({metric})"})

    # omnibus Friedman test if there are 3+ methods with matching seeds
    if len(methods) >= 3:
        pivot = df.pivot_table(index="seed", columns="method", values="accuracy")
        pivot = pivot.dropna()
        if len(pivot) >= 2:
            groups = [pivot[m].values for m in methods if m in pivot.columns]
            results.append(friedman_test(*groups, labels=methods).as_dict())

    return results


def main():
    parser = argparse.ArgumentParser(description="Run statistical significance tests on classifier/CL results.")
    parser.add_argument("--results", required=True, help="CSV with columns: seed, method, dataset, accuracy, macro_f1")
    parser.add_argument("--out", default=None, help="Optional path to write results as JSON.")
    args = parser.parse_args()

    results = run_from_results_csv(args.results)
    for r in results:
        print(json.dumps(r, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} test results to {args.out}")


if __name__ == "__main__":
    main()
