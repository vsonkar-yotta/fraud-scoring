"""Deep candidate: MLP with a category embedding layer.

Reuses the feature table train.py already built (data/features/features.parquet)
and the same temporal split. Wrapped in DeepModelWrapper so it exposes the
same predict_proba(df) interface as the sklearn/LightGBM models, and can go
through the identical evaluate.compute_metrics / promotion_gate path.

Expectation set in the design doc: this probably won't beat LightGBM on
tabular fraud data at this size -- the interesting result is *how much*
it costs in latency and training time for whatever it gains (if anything).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_scoring import registry
from fraud_scoring.evaluate import compute_metrics, promotion_gate, write_report
from fraud_scoring.features.build import FEATURE_COLUMNS
from fraud_scoring.models_deep import FraudMLP
from fraud_scoring.train import load_config, temporal_split


class DeepModelWrapper:
    """sklearn-shaped wrapper: predict_proba(df[FEATURE_COLUMNS + 'category']) -> (n, 2)."""

    def __init__(self, torch_model: nn.Module, scaler: StandardScaler, category_vocab: dict, device: str = "cpu"):
        self.torch_model = torch_model.to(device).eval()
        self.scaler = scaler
        self.category_vocab = category_vocab
        self.device = device

    def _prep(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        continuous = self.scaler.transform(X[FEATURE_COLUMNS].values)
        cat_idx = X["category"].map(lambda c: self.category_vocab.get(c, len(self.category_vocab))).values
        return (
            torch.tensor(continuous, dtype=torch.float32, device=self.device),
            torch.tensor(cat_idx, dtype=torch.long, device=self.device),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        x_cont, x_cat = self._prep(X)
        with torch.no_grad():
            logits = self.torch_model(x_cont, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.stack([1 - probs, probs], axis=1)


def train_deep_model(train_df, val_df, cfg: dict, device: str) -> DeepModelWrapper:
    dcfg = cfg["candidate_deep"]
    scaler = StandardScaler().fit(train_df[FEATURE_COLUMNS].values)
    categories = sorted(train_df["category"].unique())
    category_vocab = {c: i for i, c in enumerate(categories)}

    def to_tensors(df):
        x_cont = torch.tensor(scaler.transform(df[FEATURE_COLUMNS].values), dtype=torch.float32)
        x_cat = torch.tensor(
            df["category"].map(lambda c: category_vocab.get(c, len(category_vocab))).values, dtype=torch.long
        )
        y = torch.tensor(df["is_fraud"].values, dtype=torch.float32)
        return x_cont, x_cat, y

    x_cont_tr, x_cat_tr, y_tr = to_tensors(train_df)
    x_cont_val, x_cat_val, y_val = to_tensors(val_df)

    train_loader = DataLoader(
        TensorDataset(x_cont_tr, x_cat_tr, y_tr), batch_size=dcfg["batch_size"], shuffle=True
    )

    model = FraudMLP(
        n_continuous=len(FEATURE_COLUMNS),
        n_categories=len(category_vocab),
        embedding_dim=dcfg["embedding_dim"],
        hidden_dims=dcfg["hidden_dims"],
        dropout=dcfg["dropout"],
    ).to(device)

    pos_weight = torch.tensor(
        float((y_tr == 0).sum() / max((y_tr == 1).sum().item(), 1)), device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=dcfg["lr"])

    best_val_loss = float("inf")
    patience_left = dcfg["early_stopping_patience"]
    best_state = None

    for epoch in range(dcfg["epochs"]):
        model.train()
        for xb_cont, xb_cat, yb in train_loader:
            xb_cont, xb_cat, yb = xb_cont.to(device), xb_cat.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb_cont, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_cont_val.to(device), x_cat_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
        print(f"epoch {epoch + 1}/{dcfg['epochs']} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = dcfg["early_stopping_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    return DeepModelWrapper(model, scaler, category_vocab, device=device)


def measure_p95_latency_ms(wrapper: DeepModelWrapper, X_row: pd.DataFrame, n: int = 300) -> float:
    times = []
    for i in range(min(n, len(X_row))):
        row = X_row.iloc[[i]]
        t0 = time.perf_counter()
        wrapper.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"training deep candidate on device={device}")

    features = pd.read_parquet(cfg["data"]["features_table"])
    train, val, test = temporal_split(features, cfg["split"]["train_end_date"], cfg["split"]["val_end_date"])

    t0 = time.time()
    wrapper = train_deep_model(train, val, cfg, device)
    train_seconds = time.time() - t0
    print(f"training took {train_seconds:.1f}s")

    eval_cfg = cfg["evaluation"]
    val_prob = wrapper.predict_proba(val)[:, 1]
    test_prob = wrapper.predict_proba(test)[:, 1]
    val_metrics = compute_metrics(val["is_fraud"].values, val_prob, val["amt"].values, eval_cfg)
    test_metrics = compute_metrics(test["is_fraud"].values, test_prob, test["amt"].values, eval_cfg)

    p95 = measure_p95_latency_ms(wrapper, val)
    val_metrics["p95_inference_ms"] = p95
    test_metrics["p95_inference_ms"] = p95
    test_metrics["train_seconds"] = train_seconds

    models_dir = Path(cfg["artifacts"]["models_dir"])
    baseline_model, baseline_meta = registry.load_version(
        next(v["version"] for v in registry._load_registry(models_dir)["versions"] if v["name"] == "baseline_logreg"),
        models_dir,
    )
    gate_result = promotion_gate(baseline_meta["metrics_val"], val_metrics, p95, cfg["promotion_gate"])

    lgbm_version = next(
        v for v in reversed(registry._load_registry(models_dir)["versions"]) if v["name"] == "candidate_lightgbm"
    )
    print(f"deep vs LightGBM (test PR AUC): deep={test_metrics['pr_auc']:.4f} "
          f"lightgbm={lgbm_version['metrics_test']['pr_auc']:.4f}")

    report = {
        "models": {
            "candidate_deep_mlp_val": val_metrics,
            "candidate_deep_mlp_test": test_metrics,
            "candidate_lightgbm_test_for_comparison": lgbm_version["metrics_test"],
        },
        "gate_vs_baseline": gate_result,
        "device": device,
    }
    write_report(report, Path(cfg["artifacts"]["eval_dir"]), "deep_candidate")

    registry.save_model(
        wrapper, "candidate_deep_mlp",
        {"metrics_test": test_metrics, "metrics_val": val_metrics,
         "threshold": val_metrics["cost_optimal_threshold"], "device": device,
         "feature_columns": FEATURE_COLUMNS},
        models_dir,
    )
    print("deep candidate registered (not auto-promoted; see design.md for the promotion decision)")


if __name__ == "__main__":
    main()
