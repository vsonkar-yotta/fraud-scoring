"""Slice fraudTrain.csv into daily CSVs to simulate an incoming transaction feed.

Usage:
    python scripts/make_daily_batches.py \
        --source data/raw/fraudTrain.csv \
        --out-dir data/raw/daily_feed
"""

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category",
    "amt", "first", "last", "gender", "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
    "merch_lat", "merch_long", "is_fraud",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/fraudTrain.csv")
    parser.add_argument("--out-dir", default="data/raw/daily_feed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.source)
    assert list(df.columns) == EXPECTED_COLUMNS, "unexpected source schema"

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["_date"] = df["trans_date_trans_time"].dt.date

    n_days = 0
    for date, day_df in df.groupby("_date", sort=True):
        out_path = out_dir / f"{date.isoformat()}.csv"
        day_df.drop(columns="_date").to_csv(out_path, index=False)
        n_days += 1

    print(f"wrote {n_days} daily files to {out_dir} "
          f"({df['_date'].min()} to {df['_date'].max()}, {len(df)} rows total)")


if __name__ == "__main__":
    main()
