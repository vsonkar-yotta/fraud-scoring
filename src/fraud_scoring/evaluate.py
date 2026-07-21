"""Evaluation harness: metrics, cost-optimal threshold, and the promotion gate.

Used both as a library (train.py calls compute_metrics/promotion_gate to
decide whether a freshly trained candidate replaces the current model) and
as a CLI that reloads saved predictions and (re)renders the eval report.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def cost_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, amounts: np.ndarray,
    missed_fraud_multiplier: float, false_decline_penalty: float,
) -> dict:
    """Sweep thresholds, pick the one minimizing expected cost.

    cost = sum(missed fraud amount * multiplier) + sum(false declines) * penalty
    """
    thresholds = np.unique(np.concatenate([[0.0], y_prob, [1.0]]))
    thresholds = np.sort(thresholds)[:: max(1, len(thresholds) // 500)]  # cap sweep size

    best = {"threshold": 0.5, "cost": float("inf")}
    for t in thresholds:
        decline = y_prob >= t
        false_declines = int(np.sum(decline & (y_true == 0)))
        missed_fraud_mask = (~decline) & (y_true == 1)
        missed_fraud_cost = float(np.sum(amounts[missed_fraud_mask])) * missed_fraud_multiplier
        cost = missed_fraud_cost + false_declines * false_decline_penalty
        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": cost}
    return best


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, amounts: np.ndarray, eval_config: dict
) -> dict:
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    target_fpr = eval_config["fpr_target"]
    recall_target = recall_at_fpr(y_true, y_prob, target_fpr)
    cost_result = cost_optimal_threshold(
        y_true, y_prob, amounts,
        eval_config["cost"]["missed_fraud_multiplier"],
        eval_config["cost"]["false_decline_penalty"],
    )
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        f"recall_at_{target_fpr}_fpr": recall_target,
        "cost_optimal_threshold": cost_result["threshold"],
        "cost_at_optimal_threshold": cost_result["cost"],
        "n_rows": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }


def promotion_gate(baseline_metrics: dict, candidate_metrics: dict, p95_inference_ms: float, gate_config: dict) -> dict:
    recall_key = next(k for k in baseline_metrics if k.startswith("recall_at_"))
    pr_auc_gain = candidate_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    recall_delta = candidate_metrics[recall_key] - baseline_metrics[recall_key]

    checks = {
        "pr_auc_gain_ok": pr_auc_gain >= gate_config["min_pr_auc_gain"],
        "recall_delta_ok": recall_delta >= gate_config["min_recall_at_target_fpr_delta"],
        "latency_ok": p95_inference_ms <= gate_config["max_p95_inference_ms"],
    }
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "pr_auc_gain": pr_auc_gain,
        "recall_delta": recall_delta,
        "p95_inference_ms": p95_inference_ms,
    }


def write_report(report: dict, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))

    lines = [f"# Evaluation report: {name}", ""]
    for model_name, metrics in report.get("models", {}).items():
        lines.append(f"## {model_name}")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if "gate" in report:
        lines.append("## Promotion gate")
        for k, v in report["gate"].items():
            lines.append(f"- {k}: {v}")
    (out_dir / f"{name}.md").write_text("\n".join(lines))
