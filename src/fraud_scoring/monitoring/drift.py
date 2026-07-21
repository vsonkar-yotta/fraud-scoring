"""Data quality + drift check: null/range violations and PSI between the
training feature distribution and a trailing/recent batch.

Logs a WARNING listing offending features and exits nonzero on breach, so
this can gate a pipeline (e.g. block promotion, or feed retrain_trigger.py).

Usage:
    python -m fraud_scoring.monitoring.drift \
        --reference data/features/features.parquet \
        --recent data/features/recent_batch.parquet \
        --psi-threshold 0.2
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_scoring.features.build import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("drift")

RANGE_CHECKS = {
    "amt": (0, None),
    "lat": (-90, 90),
    "long": (-180, 180),
    "merch_lat": (-90, 90),
    "merch_long": (-180, 180),
}


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = np.maximum(expected_counts / max(len(expected), 1), 1e-6)
    actual_pct = np.maximum(actual_counts / max(len(actual), 1), 1e-6)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def null_rate_check(df: pd.DataFrame, columns: list[str]) -> dict:
    return {c: float(df[c].isnull().mean()) for c in columns if c in df.columns}


def range_violation_check(df: pd.DataFrame) -> dict:
    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        rate = float(mask.mean())
        if rate > 0:
            violations[col] = rate
    return violations


def unseen_category_rate(reference: pd.DataFrame, recent: pd.DataFrame) -> float:
    known = set(reference["category"].unique())
    return float((~recent["category"].isin(known)).mean())


def run_drift_check(
    reference: pd.DataFrame, recent: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS, psi_threshold: float = 0.2,
    null_rate_threshold: float = 0.01,
) -> dict:
    psi_by_feature = {
        col: population_stability_index(reference[col].dropna().values, recent[col].dropna().values)
        for col in feature_columns if col in reference.columns and col in recent.columns
    }
    psi_breaches = {c: v for c, v in psi_by_feature.items() if v > psi_threshold}

    null_rates = null_rate_check(recent, feature_columns)
    null_breaches = {c: v for c, v in null_rates.items() if v > null_rate_threshold}

    range_violations = range_violation_check(recent)
    unseen_cat_rate = unseen_category_rate(reference, recent) if "category" in recent.columns else 0.0

    breach = bool(psi_breaches or null_breaches or range_violations)
    result = {
        "breach": breach,
        "psi_by_feature": psi_by_feature,
        "psi_breaches": psi_breaches,
        "null_rates": null_rates,
        "null_breaches": null_breaches,
        "range_violations": range_violations,
        "unseen_category_rate": unseen_cat_rate,
        "max_psi": max(psi_by_feature.values()) if psi_by_feature else 0.0,
    }

    if breach:
        logger.warning(
            "DRIFT BREACH: psi_breaches=%s null_breaches=%s range_violations=%s",
            psi_breaches, null_breaches, range_violations,
        )
    else:
        logger.info("no drift breach; max_psi=%.4f", result["max_psi"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/features/features.parquet")
    parser.add_argument("--recent", required=True)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--out", default="artifacts/eval/drift_report.json")
    args = parser.parse_args()

    reference = pd.read_parquet(args.reference)
    recent = pd.read_parquet(args.recent) if args.recent.endswith(".parquet") else pd.read_csv(args.recent)

    result = run_drift_check(reference, recent, psi_threshold=args.psi_threshold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    if result["breach"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
