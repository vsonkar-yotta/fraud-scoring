"""Skew guardrail: offline batch features must equal online request-time features.

Takes a slice of real ingested transactions, builds features with the
offline (vectorized, rolling-window) path, then replays the same rows in
time order through the online (per-row, incremental-state) path, and
asserts every feature column matches within a tight tolerance.
"""

from pathlib import Path

import pandas as pd
import pytest

from fraud_scoring.features.build import FEATURE_COLUMNS, build_offline_features, compute_category_rates
from fraud_scoring.features.online import OnlineFeatureStore

TABLE_PATH = Path("data/ingested/transactions.parquet")


@pytest.fixture(scope="module")
def sample_df():
    if not TABLE_PATH.exists():
        pytest.skip("ingested table not present; run ingestion first")
    df = pd.read_parquet(TABLE_PATH)
    # a handful of cards with enough history to exercise every rolling window
    cards = df["cc_num"].value_counts().head(5).index
    sample = df[df["cc_num"].isin(cards)].copy()
    sample["trans_date_trans_time"] = pd.to_datetime(sample["trans_date_trans_time"])
    sample = sample.sort_values(["cc_num", "trans_date_trans_time"]).head(2000)
    return sample.reset_index(drop=True)


def test_offline_online_feature_parity(sample_df):
    category_rates = compute_category_rates(sample_df)
    offline = build_offline_features(sample_df, category_rates)
    offline_by_id = offline.set_index("trans_num")

    store = OnlineFeatureStore()
    replay_order = sample_df.sort_values("trans_date_trans_time").to_dict("records")

    mismatches = []
    for txn in replay_order:
        online_feats = store.compute_features(txn, category_rates)
        store.observe(txn)

        offline_row = offline_by_id.loc[txn["trans_num"]]
        for col in FEATURE_COLUMNS:
            off_val = float(offline_row[col])
            on_val = float(online_feats[col])
            if abs(off_val - on_val) > 1e-6 * max(1.0, abs(off_val)):
                mismatches.append((txn["trans_num"], col, off_val, on_val))

    assert not mismatches, f"offline/online mismatches (first 10): {mismatches[:10]}"
