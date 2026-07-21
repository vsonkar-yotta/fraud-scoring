import math

from fraud_scoring.features import pure
from fraud_scoring.features.build import compute_category_rates, lookup_category_rate
import pandas as pd


def test_amt_log_matches_log1p():
    assert pure.amt_log(0.0) == 0.0
    assert abs(pure.amt_log(99.0) - math.log1p(99.0)) < 1e-9


def test_haversine_zero_distance_same_point():
    assert pure.haversine_km(40.0, -74.0, 40.0, -74.0) == 0.0


def test_haversine_known_distance():
    # NYC to LA, roughly 3936 km
    d = pure.haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3800 < d < 4100


def test_hour_cyclical_wraps():
    sin0, cos0 = pure.hour_cyclical(0)
    sin24, cos24 = pure.hour_cyclical(24 % 24)
    assert abs(sin0 - sin24) < 1e-9
    assert abs(cos0 - cos24) < 1e-9


def test_is_night_boundaries():
    assert pure.is_night(23) == 1
    assert pure.is_night(3) == 1
    assert pure.is_night(12) == 0
    assert pure.is_night(6) == 0
    assert pure.is_night(21) == 0


def test_category_rate_smoothing_pulls_toward_global_for_rare_categories():
    df = pd.DataFrame({
        "category": ["a"] * 100 + ["b"] * 2,
        "is_fraud": [1] * 5 + [0] * 95 + [1, 0],
    })
    rates = compute_category_rates(df, alpha=10.0)
    # category 'b' has only 2 rows -> smoothed rate should sit closer to global than raw 0.5
    raw_b_rate = 0.5
    assert abs(lookup_category_rate("b", rates) - raw_b_rate) > abs(lookup_category_rate("a", rates) - rates["global_rate"])


def test_unseen_category_falls_back_to_global_rate():
    df = pd.DataFrame({"category": ["a", "a", "b"], "is_fraud": [1, 0, 0]})
    rates = compute_category_rates(df)
    assert lookup_category_rate("never_seen", rates) == rates["global_rate"]
