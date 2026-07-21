# Fraud Scoring — Real-Time Credit Card Fraud Detection

Vedant Rajesh Sonkar, Roll No. 2025em1100504. BITS Pilani Digital, ML Model Engineering.
Track A: online (request-response) scoring. See [`docs/design.md`](docs/design.md) for the full
design writeup and [`PLAN.md`](PLAN.md) for the original execution plan.

## What this is

A FastAPI service that scores a credit-card transaction for fraud in under a few milliseconds,
backed by a LightGBM model that beat a logistic-regression baseline and a PyTorch MLP candidate
on a strict temporal split of the Sparkov fraud dataset. Full pipeline: micro-batch ingestion →
shared offline/online feature builder → training + promotion gate → model registry → FastAPI
serving → drift monitoring → retrain trigger, orchestrated with DVC and tested in CI.

## Setup

Requires Python 3.11+ and (for LightGBM on macOS) `libomp`:

```bash
brew install libomp        # macOS only; Linux CI installs libgomp1 instead
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Get the dataset (Kaggle `kartik2112/fraud-detection`) — either via the Kaggle CLI:

```bash
export KAGGLE_API_TOKEN=<your token>   # from kaggle.com/settings/api
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

or download the zip manually and unzip `fraudTrain.csv` / `fraudTest.csv` into `data/raw/`.

## Run everything

```bash
dvc repro
```

This runs the full pipeline: slices `fraudTrain.csv` into a simulated daily feed, ingests it
into the training table, builds features, trains baseline + LightGBM + evaluates the promotion
gate, and trains the deep MLP candidate for comparison. Reports land in `artifacts/eval/`,
models in `models/` (file-based registry, `models/registry.json` tracks the current promoted
version).

Equivalent step-by-step (what `dvc.yaml` runs under the hood):

```bash
python scripts/make_daily_batches.py --source data/raw/fraudTrain.csv --out-dir data/raw/daily_feed
python -m fraud_scoring.ingest
python -m fraud_scoring.train
python -m fraud_scoring.train_deep
```

## Serve it

```bash
uvicorn fraud_scoring.serve.app:app --port 8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "trans_num": "abc123", "trans_date_trans_time": "2020-06-21 12:14:25",
  "cc_num": "2291163933867244", "merchant": "fraud_Kirlin and Sons",
  "category": "personal_care", "amt": 2.86, "city": "Columbia", "state": "SC",
  "zip": "29209", "lat": 33.9659, "long": -80.9355, "city_pop": 333497,
  "job": "Mechanical engineer", "dob": "1968-03-19",
  "merch_lat": 33.986391, "merch_long": -81.200714
}'
curl -s localhost:8000/metrics
```

Or with Docker:

```bash
docker compose up --build
```

## Load test

```bash
python scripts/load_test.py --url http://localhost:8000 --n 500 --concurrency 1 10 50 100
```

Results at each concurrency level (raw uvicorn vs Docker) are in
[`artifacts/load_test/`](artifacts/load_test/).

## Drift check / retrain trigger

```bash
python -m fraud_scoring.monitoring.drift --reference data/features/features.parquet --recent <path>
python -m fraud_scoring.monitoring.retrain_trigger --new-labeled-rows 60000 --psi-max 3.8
```

Both exit nonzero on a breach/trigger so they can gate a pipeline.

## Tests

```bash
pytest tests/ -q
```

Covers: feature-transform correctness, offline/online feature parity (the skew guardrail — see
`tests/test_feature_parity.py`), promotion-gate logic, drift-check triggers, and the API
contract. Tests that need the real ingested dataset or a promoted model skip gracefully if
those artifacts aren't present, so `pytest` runs clean on a bare checkout too.

## Repository layout

See [`PLAN.md`](PLAN.md) for the full annotated structure and rationale. Key entry points:

- `src/fraud_scoring/ingest.py` — micro-batch ingestion
- `src/fraud_scoring/features/{build,online,pure}.py` — the shared offline/online feature logic
- `src/fraud_scoring/train.py`, `train_deep.py` — training pipelines
- `src/fraud_scoring/evaluate.py` — metrics + promotion gate
- `src/fraud_scoring/registry.py` — file-based model registry
- `src/fraud_scoring/serve/app.py` — FastAPI service
- `src/fraud_scoring/monitoring/{drift,retrain_trigger}.py` — drift check + retrain trigger
- `docs/design.md` — full design writeup
- `docs/architecture.png` — architecture diagram

## Known environment notes

- LightGBM needs `libomp` (macOS) or `libgomp1` (Linux/CI) — not a Python dependency, installed
  separately (see Setup above and `.github/workflows/ci.yml`).
- The Dockerfile trusts a local corporate CA bundle (`ca-bundle.pem`, gitignored) to work around
  a network SSL intercept during `pip install` in this build environment; drop that `COPY` +
  `update-ca-certificates` block in an environment without that constraint.
