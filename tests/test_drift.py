import numpy as np
import pandas as pd

from fraud_scoring.monitoring.drift import range_violation_check, run_drift_check


def _make_reference(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amt_log": rng.normal(3, 1, n),
        "category": rng.choice(["grocery", "gas", "shopping"], n),
        "amt": rng.uniform(1, 200, n),
        "lat": rng.uniform(25, 49, n),
        "long": rng.uniform(-124, -67, n),
        "merch_lat": rng.uniform(25, 49, n),
        "merch_long": rng.uniform(-124, -67, n),
    })


def test_no_breach_on_identical_distribution():
    ref = _make_reference()
    result = run_drift_check(ref, ref.copy(), feature_columns=["amt_log"])
    assert result["breach"] is False


def test_breach_fires_on_shifted_distribution():
    ref = _make_reference()
    shifted = ref.copy()
    shifted["amt_log"] = shifted["amt_log"] + 5  # big mean shift
    result = run_drift_check(ref, shifted, feature_columns=["amt_log"])
    assert result["breach"] is True
    assert "amt_log" in result["psi_breaches"]


def test_range_violation_flags_negative_amount_and_bad_coordinates():
    df = pd.DataFrame({
        "amt": [-5.0, 10.0],
        "lat": [95.0, 40.0],  # invalid latitude
        "long": [-70.0, -70.0],
        "merch_lat": [40.0, 40.0],
        "merch_long": [-70.0, -70.0],
    })
    violations = range_violation_check(df)
    assert "amt" in violations
    assert "lat" in violations
