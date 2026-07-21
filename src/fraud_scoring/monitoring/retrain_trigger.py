"""Retraining trigger: decides whether a retrain should fire, based on
staleness + new label volume, sustained feature drift, or model decay on
recent labeled feedback.

    retrain if any of:
      days_since_last_train >= 14 and new_labeled_rows >= 50_000
      psi_max_top_features > 0.2 for 3 consecutive daily checks
      pr_auc_on_recent_labels < promoted_pr_auc - 0.03

Keeps a small state file (`data/ingested/_retrain_state.json`) recording the
trailing PSI history so "3 consecutive daily checks" can be evaluated
without re-deriving it from scratch each run.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fraud_scoring import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain_trigger")

DAYS_SINCE_THRESHOLD = 14
NEW_ROWS_THRESHOLD = 50_000
PSI_THRESHOLD = 0.2
PSI_CONSECUTIVE_DAYS = 3
PR_AUC_DROP_THRESHOLD = 0.03


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"psi_history": []}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2))


def evaluate_retrain(
    days_since_last_train: float,
    new_labeled_rows: int,
    psi_max_history: list[float],
    pr_auc_on_recent_labels: float | None,
    promoted_pr_auc: float,
) -> dict:
    staleness_trigger = days_since_last_train >= DAYS_SINCE_THRESHOLD and new_labeled_rows >= NEW_ROWS_THRESHOLD

    recent_psi = psi_max_history[-PSI_CONSECUTIVE_DAYS:]
    drift_trigger = len(recent_psi) >= PSI_CONSECUTIVE_DAYS and all(p > PSI_THRESHOLD for p in recent_psi)

    decay_trigger = (
        pr_auc_on_recent_labels is not None
        and pr_auc_on_recent_labels < promoted_pr_auc - PR_AUC_DROP_THRESHOLD
    )

    reasons = []
    if staleness_trigger:
        reasons.append(f"stale: {days_since_last_train:.1f}d since last train with {new_labeled_rows} new labeled rows")
    if drift_trigger:
        reasons.append(f"drift: PSI > {PSI_THRESHOLD} for {PSI_CONSECUTIVE_DAYS} consecutive checks ({recent_psi})")
    if decay_trigger:
        reasons.append(f"decay: recent PR AUC {pr_auc_on_recent_labels:.4f} < promoted {promoted_pr_auc:.4f} - {PR_AUC_DROP_THRESHOLD}")

    return {
        "retrain": bool(reasons),
        "reasons": reasons,
        "staleness_trigger": staleness_trigger,
        "drift_trigger": drift_trigger,
        "decay_trigger": decay_trigger,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models/")
    parser.add_argument("--state", default="data/ingested/_retrain_state.json")
    parser.add_argument("--new-labeled-rows", type=int, required=True)
    parser.add_argument("--psi-max", type=float, required=True, help="today's max PSI across top features")
    parser.add_argument("--recent-pr-auc", type=float, default=None, help="PR AUC on recently labeled feedback, if available")
    args = parser.parse_args()

    state_path = Path(args.state)
    state = _load_state(state_path)
    state["psi_history"] = (state.get("psi_history", []) + [args.psi_max])[-PSI_CONSECUTIVE_DAYS:]

    _, metadata = registry.load_current(Path(args.models_dir))
    last_train = datetime.fromisoformat(metadata["saved_at"])
    days_since_last_train = (datetime.now(timezone.utc) - last_train).total_seconds() / 86400.0
    promoted_pr_auc = metadata["metrics_test"]["pr_auc"]

    result = evaluate_retrain(
        days_since_last_train, args.new_labeled_rows, state["psi_history"], args.recent_pr_auc, promoted_pr_auc
    )
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    state["last_result"] = result
    _save_state(state_path, state)

    if result["retrain"]:
        logger.warning("RETRAIN TRIGGERED: %s", result["reasons"])
    else:
        logger.info("no retrain trigger; psi_history=%s days_since_last_train=%.1f", state["psi_history"], days_since_last_train)

    print(json.dumps(result, indent=2))
    if result["retrain"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
