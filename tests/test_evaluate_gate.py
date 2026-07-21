from fraud_scoring.evaluate import promotion_gate

GATE_CONFIG = {
    "min_pr_auc_gain": 0.02,
    "min_recall_at_target_fpr_delta": 0.0,
    "max_p95_inference_ms": 20,
}


def _metrics(pr_auc: float, recall: float) -> dict:
    return {"pr_auc": pr_auc, "recall_at_0.01_fpr": recall}


def test_gate_promotes_when_all_checks_pass():
    result = promotion_gate(_metrics(0.5, 0.6), _metrics(0.6, 0.7), p95_inference_ms=10, gate_config=GATE_CONFIG)
    assert result["promote"] is True
    assert all(result["checks"].values())


def test_gate_blocks_on_insufficient_pr_auc_gain():
    result = promotion_gate(_metrics(0.5, 0.6), _metrics(0.505, 0.7), p95_inference_ms=10, gate_config=GATE_CONFIG)
    assert result["promote"] is False
    assert result["checks"]["pr_auc_gain_ok"] is False


def test_gate_blocks_on_recall_regression():
    result = promotion_gate(_metrics(0.5, 0.6), _metrics(0.6, 0.5), p95_inference_ms=10, gate_config=GATE_CONFIG)
    assert result["promote"] is False
    assert result["checks"]["recall_delta_ok"] is False


def test_gate_blocks_on_latency_budget_even_if_metrics_win():
    result = promotion_gate(_metrics(0.5, 0.6), _metrics(0.9, 0.9), p95_inference_ms=25, gate_config=GATE_CONFIG)
    assert result["promote"] is False
    assert result["checks"]["latency_ok"] is False
