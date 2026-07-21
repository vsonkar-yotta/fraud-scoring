"""Full training pipeline: build features, temporal split, fit baseline +
LightGBM candidate, evaluate both, run the promotion gate, and register
whichever model wins (or keep the current one if nothing clears the gate).
"""

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_scoring import registry
from fraud_scoring.evaluate import compute_metrics, promotion_gate, write_report
from fraud_scoring.features.build import FEATURE_COLUMNS, build_offline_features, compute_category_rates


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_combined_data(cfg: dict) -> pd.DataFrame:
    ingested = pd.read_parquet(cfg["data"]["ingested_table"])
    holdout = pd.read_csv(cfg["data"]["raw_test_csv"])
    combined = pd.concat([ingested, holdout], ignore_index=True)
    combined["trans_date_trans_time"] = pd.to_datetime(combined["trans_date_trans_time"])
    combined["cc_num"] = combined["cc_num"].astype(str)
    return combined.sort_values("trans_date_trans_time").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = df["trans_date_trans_time"]
    train = df[t <= train_end]
    val = df[(t > train_end) & (t <= val_end)]
    test = df[t > val_end]
    return train, val, test


def measure_p95_latency_ms(model, X_row: pd.DataFrame, n: int = 300) -> float:
    times = []
    for i in range(min(n, len(X_row))):
        row = X_row.iloc[[i]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])

    print("loading combined data (ingested train + fraudTest holdout)...")
    combined = load_combined_data(cfg)

    train_end = cfg["split"]["train_end_date"]
    val_end = cfg["split"]["val_end_date"]
    train_raw, _, _ = temporal_split(combined, train_end, val_end)

    print("fitting category fraud-rate lookup on train fold only...")
    category_rates = compute_category_rates(train_raw)

    print("building features over combined data (rolling windows carry across split)...")
    features = build_offline_features(combined, category_rates)
    Path(cfg["data"]["features_table"]).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cfg["data"]["features_table"], index=False)

    train, val, test = temporal_split(features, train_end, val_end)
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"fraud_rate train/val/test = {train.is_fraud.mean():.4f}/{val.is_fraud.mean():.4f}/{test.is_fraud.mean():.4f}")

    X_train, y_train = train[FEATURE_COLUMNS], train["is_fraud"].values
    X_val, y_val = val[FEATURE_COLUMNS], val["is_fraud"].values
    X_test, y_test = test[FEATURE_COLUMNS], test["is_fraud"].values

    # --- baseline: logistic regression ---
    print("training baseline logistic regression...")
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=cfg["baseline"]["class_weight"],
            max_iter=cfg["baseline"]["max_iter"],
        )),
    ])
    baseline.fit(X_train, y_train)

    # --- candidate: LightGBM ---
    print("training LightGBM candidate...")
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidate = lgb.LGBMClassifier(
        n_estimators=cfg["candidate_gbdt"]["n_estimators"],
        learning_rate=cfg["candidate_gbdt"]["learning_rate"],
        num_leaves=cfg["candidate_gbdt"]["num_leaves"],
        max_depth=cfg["candidate_gbdt"]["max_depth"],
        scale_pos_weight=scale_pos_weight,
        random_state=cfg["seed"],
        verbose=-1,
    )
    candidate.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(cfg["candidate_gbdt"]["early_stopping_rounds"], verbose=False)],
    )

    # --- evaluate both on val (for the gate) ---
    eval_cfg = cfg["evaluation"]
    baseline_val_prob = baseline.predict_proba(X_val)[:, 1]
    candidate_val_prob = candidate.predict_proba(X_val)[:, 1]

    baseline_val_metrics = compute_metrics(y_val, baseline_val_prob, val["amt"].values, eval_cfg)
    candidate_val_metrics = compute_metrics(y_val, candidate_val_prob, val["amt"].values, eval_cfg)

    print("measuring single-row inference latency (p95, ms)...")
    baseline_p95 = measure_p95_latency_ms(baseline, X_val)
    candidate_p95 = measure_p95_latency_ms(candidate, X_val)

    gate_result = promotion_gate(baseline_val_metrics, candidate_val_metrics, candidate_p95, cfg["promotion_gate"])
    print(f"gate result: {gate_result}")

    # --- final report on held-out test ---
    baseline_test_prob = baseline.predict_proba(X_test)[:, 1]
    candidate_test_prob = candidate.predict_proba(X_test)[:, 1]
    baseline_test_metrics = compute_metrics(y_test, baseline_test_prob, test["amt"].values, eval_cfg)
    candidate_test_metrics = compute_metrics(y_test, candidate_test_prob, test["amt"].values, eval_cfg)
    baseline_test_metrics["p95_inference_ms"] = baseline_p95
    candidate_test_metrics["p95_inference_ms"] = candidate_p95

    report = {
        "split": {"train_end": train_end, "val_end": val_end, "n_train": len(train), "n_val": len(val), "n_test": len(test)},
        "models": {
            "baseline_logreg_val": baseline_val_metrics,
            "candidate_lightgbm_val": candidate_val_metrics,
            "baseline_logreg_test": baseline_test_metrics,
            "candidate_lightgbm_test": candidate_test_metrics,
        },
        "gate": gate_result,
    }
    write_report(report, Path(cfg["artifacts"]["eval_dir"]), "baseline_vs_candidate")
    print(f"eval report written to {cfg['artifacts']['eval_dir']}")

    # --- register both; promote whichever wins the gate ---
    models_dir = Path(cfg["artifacts"]["models_dir"])
    baseline_version = registry.save_model(
        baseline, "baseline_logreg",
        {"metrics_test": baseline_test_metrics, "metrics_val": baseline_val_metrics,
         "threshold": baseline_val_metrics["cost_optimal_threshold"],
         "data_range": [str(combined.trans_date_trans_time.min()), str(combined.trans_date_trans_time.max())],
         "feature_columns": FEATURE_COLUMNS, "category_rates": category_rates},
        models_dir,
    )
    candidate_version = registry.save_model(
        candidate, "candidate_lightgbm",
        {"metrics_test": candidate_test_metrics, "metrics_val": candidate_val_metrics,
         "threshold": candidate_val_metrics["cost_optimal_threshold"],
         "data_range": [str(combined.trans_date_trans_time.min()), str(combined.trans_date_trans_time.max())],
         "feature_columns": FEATURE_COLUMNS, "category_rates": category_rates},
        models_dir,
    )

    winner_version = candidate_version if gate_result["promote"] else baseline_version
    registry.promote(winner_version, models_dir)
    print(f"promoted: {winner_version} (candidate cleared gate: {gate_result['promote']})")


if __name__ == "__main__":
    main()
