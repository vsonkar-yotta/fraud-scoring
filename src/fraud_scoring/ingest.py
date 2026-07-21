"""Micro-batch ingestion job.

Scans a directory of daily transaction CSVs (the simulated incoming feed),
picks up any files not yet processed, validates their schema, dedupes
against the existing training table on `trans_num`, and appends the result
to the parquet training table.

Every run appends one line to `data/ingested/ingest_log.jsonl` recording
rows ingested, date range covered, and null counts per column -- this log
is the evidence artifact for the ingestion screenshot.

Usage:
    python -m fraud_scoring.ingest \
        --feed-dir data/raw/daily_feed \
        --table data/ingested/transactions.parquet
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")

EXPECTED_COLUMNS = [
    "Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category",
    "amt", "first", "last", "gender", "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
    "merch_lat", "merch_long", "is_fraud",
]


class SchemaValidationError(Exception):
    pass


def validate_schema(df: pd.DataFrame, path: Path) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing or extra:
        raise SchemaValidationError(
            f"{path.name}: schema mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
        )


def load_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    return set(json.loads(state_path.read_text())["processed_files"])


def save_state(state_path: Path, processed_files: set[str]) -> None:
    state_path.write_text(json.dumps({"processed_files": sorted(processed_files)}, indent=2))


def run_ingest(feed_dir: Path, table_path: Path, state_path: Path, log_path: Path) -> dict:
    processed = load_state(state_path)
    all_files = sorted(feed_dir.glob("*.csv"))
    new_files = [f for f in all_files if f.name not in processed]

    run_record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "feed_dir": str(feed_dir),
        "files_seen": len(all_files),
        "new_files": len(new_files),
    }

    if not new_files:
        logger.info("no new files to ingest (%d already processed)", len(processed))
        run_record.update(rows_ingested=0, rejected_files=[], date_min=None, date_max=None, null_counts={})
        _append_log(log_path, run_record)
        return run_record

    rejected = []
    good_frames = []
    for f in new_files:
        try:
            df = pd.read_csv(f)
            validate_schema(df, f)
            good_frames.append(df)
        except SchemaValidationError as e:
            logger.error("rejecting %s: %s", f.name, e)
            rejected.append(f.name)

    accepted_names = {f.name for f in new_files if f.name not in rejected}

    if not good_frames:
        logger.warning("all %d new files rejected on schema validation", len(new_files))
        run_record.update(rows_ingested=0, rejected_files=rejected, date_min=None, date_max=None, null_counts={})
        _append_log(log_path, run_record)
        save_state(state_path, processed | accepted_names)
        return run_record

    new_data = pd.concat(good_frames, ignore_index=True)
    new_data["trans_date_trans_time"] = pd.to_datetime(new_data["trans_date_trans_time"])

    if table_path.exists():
        existing = pd.read_parquet(table_path)
        before = len(new_data)
        new_data = new_data[~new_data["trans_num"].isin(set(existing["trans_num"]))]
        deduped = before - len(new_data)
        if deduped:
            logger.info("dropped %d duplicate trans_num already in table", deduped)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    table_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(table_path, index=False)

    null_counts = {k: int(v) for k, v in new_data.isnull().sum().items() if v > 0}

    run_record.update(
        rows_ingested=len(new_data),
        rejected_files=rejected,
        date_min=str(new_data["trans_date_trans_time"].min()) if len(new_data) else None,
        date_max=str(new_data["trans_date_trans_time"].max()) if len(new_data) else None,
        null_counts=null_counts,
        table_row_count=len(combined),
    )
    logger.info(
        "ingested %d rows from %d files (%s to %s); table now %d rows; nulls=%s",
        run_record["rows_ingested"], len(good_frames),
        run_record["date_min"], run_record["date_max"],
        run_record["table_row_count"], null_counts,
    )

    _append_log(log_path, run_record)
    save_state(state_path, processed | accepted_names)
    return run_record


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-dir", default="data/raw/daily_feed")
    parser.add_argument("--table", default="data/ingested/transactions.parquet")
    parser.add_argument("--state", default="data/ingested/_ingest_state.json")
    parser.add_argument("--log", default="data/ingested/ingest_log.jsonl")
    args = parser.parse_args()

    record = run_ingest(Path(args.feed_dir), Path(args.table), Path(args.state), Path(args.log))
    if record["rows_ingested"] == 0 and record["new_files"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
