"""Offline feature builder.

Computes the full feature set for a batch of historical transactions.
Pure per-row transforms come straight from `pure.py` (np.vectorize'd calls
into the exact same functions the online path uses, so there is only one
implementation of that math). Stateful/rolling features (velocity, rolling
averages, time-since-last, card age) are computed here with pandas
group-rolling for speed; `online.py` recomputes the same quantities from an
incremental per-card state store, and `tests/test_feature_parity.py`
asserts the two paths agree row for row.
"""

import numpy as np
import pandas as pd

from fraud_scoring.features import pure

FEATURE_COLUMNS = [
    "amt_log",
    "amt_over_card_avg_30d",
    "txn_count_card_1h",
    "txn_count_card_24h",
    "time_since_last_txn_card",
    "haversine_km_customer_merchant",
    "is_night",
    "hour_sin",
    "hour_cos",
    "category_fraud_rate_smoothed",
    "card_age_days_in_data",
]

TIME_SINCE_SENTINEL_SECONDS = 999_999.0
CARD_AGE_ALPHA = 10.0


def compute_category_rates(train_df: pd.DataFrame, alpha: float = CARD_AGE_ALPHA) -> dict:
    """Laplace-smoothed fraud rate per category, fit on the train fold only."""
    global_rate = float(train_df["is_fraud"].mean())
    agg = train_df.groupby("category")["is_fraud"].agg(["sum", "count"])
    rates = (agg["sum"] + alpha * global_rate) / (agg["count"] + alpha)
    return {"rates": rates.to_dict(), "global_rate": global_rate, "alpha": alpha}


def lookup_category_rate(category: str, category_rates: dict) -> float:
    return category_rates["rates"].get(category, category_rates["global_rate"])


def build_offline_features(df: pd.DataFrame, category_rates: dict) -> pd.DataFrame:
    """Vectorized feature build over a full historical dataframe.

    `df` must contain at least: cc_num, trans_date_trans_time, amt, category,
    lat, long, merch_lat, merch_long. Returns a copy with FEATURE_COLUMNS added,
    sorted by (cc_num, trans_date_trans_time).
    """
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # --- pure, stateless transforms: literally the same functions used online ---
    df["amt_log"] = np.vectorize(pure.amt_log)(df["amt"].values)
    df["haversine_km_customer_merchant"] = np.vectorize(pure.haversine_km)(
        df["lat"].values, df["long"].values, df["merch_lat"].values, df["merch_long"].values
    )
    hours = df["trans_date_trans_time"].dt.hour.values
    sin_cos = np.vectorize(pure.hour_cyclical)(hours)
    df["hour_sin"], df["hour_cos"] = sin_cos[0], sin_cos[1]
    df["is_night"] = np.vectorize(pure.is_night)(hours)

    # --- category rate lookup (fit on train fold, applied everywhere) ---
    df["category_fraud_rate_smoothed"] = df["category"].apply(
        lambda c: lookup_category_rate(c, category_rates)
    )

    # --- stateful / rolling features, per card ---
    ts_indexed = df.set_index("trans_date_trans_time")
    grp = ts_indexed.groupby("cc_num")

    count_1h = grp["amt"].rolling("1h").count().reset_index(level=0, drop=True)
    count_24h = grp["amt"].rolling("24h").count().reset_index(level=0, drop=True)
    sum_30d = grp["amt"].rolling("30d").sum().reset_index(level=0, drop=True)
    count_30d = grp["amt"].rolling("30d").count().reset_index(level=0, drop=True)

    df["txn_count_card_1h"] = (count_1h.values - 1).astype(float)
    df["txn_count_card_24h"] = (count_24h.values - 1).astype(float)

    prior_sum_30d = sum_30d.values - df["amt"].values
    prior_count_30d = count_30d.values - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_prior_30d = np.where(prior_count_30d > 0, prior_sum_30d / np.maximum(prior_count_30d, 1), np.nan)
        ratio = df["amt"].values / avg_prior_30d
    df["amt_over_card_avg_30d"] = np.where(np.isnan(ratio), 1.0, ratio)

    time_since = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    df["time_since_last_txn_card"] = time_since.fillna(TIME_SINCE_SENTINEL_SECONDS)

    first_seen = df.groupby("cc_num")["trans_date_trans_time"].transform("min")
    df["card_age_days_in_data"] = (df["trans_date_trans_time"] - first_seen).dt.total_seconds() / 86400.0

    return df
