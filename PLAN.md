# Execution Plan: Real-Time Fraud Detection Scoring System

Vedant Rajesh Sonkar, Roll No. 2025em1100504
Course: Machine Learning Model Engineering, BITS Pilani Digital
Assignment: Design and Build a Mini Production ML System (Track A: online scoring)

## What I'm building and why

I'm building a real-time credit card fraud scoring system. A payment gateway sends a transaction to my API, and the API returns a fraud probability plus an approve/review/decline decision in under 200ms, because a customer is standing at checkout waiting for the answer.

I picked fraud over churn or demand forecasting because it forces me to deal with the hard parts of production ML in one project: extreme class imbalance (fraud is well under 1% of transactions), features that drift as fraud patterns change, latency budgets that rule out heavy preprocessing at request time, and a training-serving skew problem that is real rather than theoretical (velocity features computed over a warehouse offline behave differently than the same features computed live).

Track A fits this use case. Fraud scoring is a request-response problem where a human is waiting, so the plan puts extra weight on latency measurement, load testing, and monitoring of the live service. I'll still build the batch ingestion piece the assignment requires, and I'll use a hybrid feature pattern (batch precompute for heavy aggregates, online compute for light velocity features) to show I understand where each belongs.

The assignment says it doesn't grade on beating state of the art. I'm going to aim high on model quality anyway, because a strong candidate model makes the baseline-vs-candidate comparison and the promotion gate story much more convincing. Target: PR AUC and recall-at-low-FPR that clearly beat a logistic regression baseline, with an honest writeup of what each model costs in latency.

## Dataset

Sparkov Credit Card Transaction Fraud dataset (Kaggle: kartik2112/fraud-detection). Roughly 1.85M simulated transactions from Jan 2019 to Dec 2020, split into fraudTrain.csv (~1.3M rows) and fraudTest.csv (~555k rows), with a ~0.5% fraud rate.

Why this one and not the two more famous fraud datasets:

- ULB Credit Card Fraud: features are PCA-anonymized (V1 to V28). I can't build interpretable aggregation or velocity features on it, and Part A of the rubric is exactly about that.
- IEEE-CIS: realistic but hundreds of masked columns. Feature engineering turns into guessing what C13 means, and I can't explain my features in the design doc.

Sparkov gives me readable columns: timestamp, card number (as a stable card ID), merchant, category, amount, customer demographics, customer lat/long, merchant lat/long, and an is_fraud label. Every feature I build is explainable. It's synthetic, and I'll say so openly in the doc under limitations. The assignment grades production ML thinking, not dataset prestige.

Cleaning and assumptions I'll document: parse timestamps to UTC, treat cc_num as a card identifier (never a model feature directly), drop PII-like columns (name, street) from the model, note the simulator's known quirks (fraud is injected in bursts per card, which actually makes the drift story interesting).

## Repository structure

```
fraud-scoring/
├── README.md                  # setup, how to run everything, screenshots
├── PLAN.md                    # this file
├── pyproject.toml             # deps, pinned
├── Dockerfile                 # serves the API
├── docker-compose.yml         # api + optional monitoring view
├── dvc.yaml                   # data + pipeline versioning
├── .github/workflows/ci.yml   # lint, tests, tiny smoke-train on sample data
├── configs/
│   ├── train.yaml             # paths, model params, split dates
│   └── serve.yaml             # model path, thresholds, feature config
├── data/                      # DVC-tracked, not in git
│   ├── raw/
│   ├── ingested/              # the growing "training table"
│   └── features/
├── src/fraud_scoring/
│   ├── ingest.py              # micro-batch ingestion script
│   ├── features/
│   │   ├── build.py           # offline feature build (shared logic)
│   │   └── online.py          # online feature lookup + light compute
│   ├── train.py               # full training pipeline
│   ├── evaluate.py            # eval harness, baseline vs candidate, gate
│   ├── registry.py            # tiny file-based model registry
│   ├── serve/
│   │   ├── app.py             # FastAPI, /predict, /health, /metrics
│   │   └── schemas.py         # pydantic request/response
│   └── monitoring/
│       ├── drift.py           # PSI + null/range checks
│       └── retrain_trigger.py # trigger logic
├── tests/                     # pytest: features, api, gate, drift
├── scripts/
│   ├── load_test.py           # async latency/throughput measurement
│   └── make_daily_batches.py  # slices raw data into daily CSVs to simulate a feed
├── models/                    # saved artifacts + metadata json
├── artifacts/eval/            # eval reports (json + md)
└── docs/
    ├── design.md              # the 1500-2000 word design doc
    └── architecture.png       # the diagram
```

One thing I care about: `features/build.py` is imported by both `train.py` and the serving path. Same code computes features offline and online wherever possible. That's my main answer to training-serving skew, and I'll write a test that proves offline and online feature values match for the same input.

## Part A: Data and features (25%)

Ingestion: `make_daily_batches.py` slices fraudTrain.csv into daily CSVs so I have a realistic incoming feed. `ingest.py` runs as a micro-batch job: reads any new daily files, validates schema, dedupes on transaction ID, appends to the parquet training table in `data/ingested/`, and logs rows ingested, date range, and null counts per run. This log becomes a monitoring screenshot later.

Features, minimum eight, all interpretable:

1. `amt_log`: log-transformed amount.
2. `amt_over_card_avg_30d`: this transaction's amount divided by the card's 30-day average. Fraudsters spend differently than the cardholder.
3. `txn_count_card_1h` and `txn_count_card_24h`: velocity. Card testing attacks fire many transactions fast.
4. `time_since_last_txn_card`: seconds since this card's previous transaction.
5. `haversine_km_customer_merchant`: distance between customer home lat/long and merchant lat/long. Large jumps are suspicious.
6. `is_night` and `hour_sin/hour_cos`: cyclical time encoding. The Sparkov simulator concentrates fraud at night, and real fraud has strong time patterns too.
7. `category_fraud_rate_smoothed`: target-encoded merchant category with Laplace smoothing, computed only on training folds to avoid leakage.
8. `card_age_days_in_data`: how long we've seen this card. New cards score differently.

Offline vs online split, which I'll table in the design doc: 2, 7 and 8 are precomputed in batch and looked up from a small feature table keyed by card/category (the "hybrid" part). 3, 4 need to be online or near-online in real production; in my demo the API computes them from a rolling in-memory store seeded from recent history, and I'll discuss what this becomes at scale (Redis with TTL'd counters). 1, 5, 6 are pure request-time transforms, same function both sides.

Skew guardrail: a pytest case feeds identical transactions through the offline builder and the online path and asserts feature equality within tolerance.

## Part B: Training and offline evaluation (25%)

Split: strictly temporal. Train on the earlier months, validate on the next slice, hold out the final months as test. Random splits leak future velocity information into the past, and I'll say so in the doc since it's a classic fraud-ML mistake.

Models:

- Baseline: logistic regression on scaled features with class weights. Cheap, calibrated-ish, the thing you'd actually ship first.
- Candidate 1: LightGBM/XGBoost with tuned depth, learning rate, and scale_pos_weight. This is the expected winner on tabular fraud.
- Candidate 2 (the deep one, on the 4080): a PyTorch MLP with embedding layers for categoricals, or FT-Transformer if time allows. Honest expectation: it probably won't beat gradient boosting on this data, and that negative result is itself good design-doc material about when deep models earn their serving cost.

Metrics, chosen for the imbalance, each justified in the doc:

- PR AUC as the headline. ROC AUC looks flattering at 0.5% positive rate; PR AUC doesn't lie.
- ROC AUC as the familiar secondary.
- Recall at 1% FPR: the business question. How much fraud do I catch if I'm only allowed to annoy 1 in 100 legitimate customers?
- A simple cost metric: assume a missed fraud costs the transaction amount, a false decline costs a fixed friction penalty. Pick the decision threshold that minimizes expected cost on validation, and ship that threshold in serve.yaml.

Promotion gate, implemented in `evaluate.py`, not just described:

```
promote candidate if:
  PR_AUC(candidate) >= PR_AUC(baseline) + 0.02
  and recall_at_1pct_FPR(candidate) >= recall_at_1pct_FPR(baseline)
  and p95_single_row_inference_ms <= 20
```

The latency clause is the Track A touch: a model that wins offline but blows the latency budget doesn't get promoted. Eval outputs go to `artifacts/eval/` as JSON plus a rendered markdown report, and the winning model plus a metadata file (version, git commit, data range, metrics, threshold) goes to `models/` via `registry.py`.

## Part C: Serving and inference (25%)

FastAPI service with three endpoints:

- `POST /predict`: JSON transaction in; out comes `{fraud_probability, decision, model_version, latency_ms, feature_flags}` where feature_flags notes anything odd (e.g., unseen category, cold-start card).
- `GET /health`: liveness plus loaded model version.
- `GET /metrics`: rolling request count, error rate, latency percentiles, and score distribution stats, which feeds the monitoring story.

Inference pattern justification (M2 framing, goes in the doc): a human is waiting, the decision blocks money movement, so online request-response is mandatory. Acceptable end-to-end budget ~200ms; my model slice of that should be well under 50ms. Batch would mean detecting fraud after the money left, and streaming without a response channel doesn't answer the gateway. Hybrid enters through the feature layer, not the scoring layer.

Load test: `load_test.py` uses asyncio + httpx to fire configurable concurrent request streams sampled from real test rows. Reports avg, p50, p95, p99 latency and requests/sec at several concurrency levels, dumped to a table I screenshot. I'll run it against both the raw uvicorn process and the Docker container and report both.

Docker: multi-stage build, model baked in via build arg pointing at the registry, one `docker compose up` to run. The README shows the exact curl that produces the screenshot.

## Part D: Monitoring, data quality, retraining (25%)

Monitoring plan (design doc, with a table of metric, threshold, alert, audience):

- Infra: p95/p99 latency, error rate, throughput. Alert to the on-call engineer if p95 > 150ms for 5 min or error rate > 1%.
- Data/feature: null rate per feature, out-of-range rate (negative amounts, impossible coordinates), unseen-category rate, and PSI per feature between training distribution and the trailing window. Alert to the ML engineer at PSI > 0.2 on any top feature.
- Model/business: daily score distribution drift, decline rate, and once labels arrive (chargebacks lag weeks, which I'll discuss), rolling PR AUC on labeled feedback. Alert to the fraud team if decline rate doubles day over day.

Implemented check (the working one the rubric asks for): `drift.py` computes null/range violations and PSI between the training feature distribution and a "recent batch," logs a WARNING with the offending features, and exits nonzero on breach so it can gate a pipeline. I'll manufacture drift for the screenshot by scaling amounts in a synthetic recent batch.

Retraining trigger, real function in `retrain_trigger.py`:

```
retrain if any of:
  days_since_last_train >= 14 and new_labeled_rows >= 50_000
  psi_max_top_features > 0.2 for 3 consecutive daily checks
  pr_auc_on_recent_labels < promoted_pr_auc - 0.03
```

Incident scenario for the doc: the upstream gateway renames `amt` to `amount_usd` in the daily feed. Ingestion's schema validation rejects the files and alerts, so the training table stays clean, but the online API starts seeing nulls for amount-derived features, the null-rate monitor fires within minutes, and feature_flags in responses spike. Response: freeze retraining, serve falls back to imputation + a conservative threshold, fix the mapping in ingestion, backfill the gap, verify PSI back to normal, unfreeze. This one scenario touches monitoring, alerting, rollback thinking, and pipeline repair, which is why I chose it.

## Code quality, reproducibility, CI

- Pinned deps in pyproject.toml, one `make setup && make all` path in the README.
- DVC tracks raw data and the pipeline stages (ingest -> features -> train -> evaluate), so `dvc repro` rebuilds everything and the data version is pinned to the git commit.
- GitHub Actions: ruff, pytest, and a smoke train on a 5k-row sample so CI proves the pipeline runs end to end without needing the full dataset.
- Tests: feature correctness, offline/online parity, API contract (schema, model_version present), promotion gate logic, drift check triggers.
- Seeds fixed everywhere; every model artifact records git commit + data hash.

## Design doc and diagram

`docs/design.md`, 1500 to 2000 words, sections mapped one-to-one to the rubric: problem and metrics, data and features (with the offline/online table), model choice and evaluation (baseline vs both candidates, gate decision, the deep-model verdict), serving pattern and measured latency numbers, pipeline and retraining, monitoring plan, incident scenario, trade-offs and limitations (synthetic data, in-memory velocity store, label lag), future work (Redis feature store, shadow deployment, calibration).

Architecture diagram: daily CSVs -> ingestion (schema check, log) -> training table -> feature build -> train/eval -> promotion gate -> model registry -> FastAPI serving (with online feature path) -> monitoring -> retraining trigger looping back to training. One image, boxes and arrows, exported as PNG.

## Demo artifact

Screenshots, 5 of them: curl + JSON response from /predict, load test latency table, ingestion run log, drift check firing on the manufactured batch, CI green run. No frontend needed per the brief. If time is left over, a one-page Streamlit view of /metrics as a bonus sixth screenshot.

## Build order

1. Repo scaffold, deps, DVC init, download and profile the data, ingestion script. (Everything downstream depends on the ingested table.)
2. Feature builder + parity test, temporal split, baseline model + eval harness.
3. LightGBM candidate, tuning, promotion gate, registry, eval reports.
4. FastAPI service, online feature path, Docker, load test.
5. Deep candidate on the 4080, final model comparison.
6. Drift check, retraining trigger, monitoring writeup, incident scenario.
7. Design doc, architecture diagram, README polish, screenshots, CI green.
8. Final pass: run everything from a clean clone, fix anything that breaks, zip.

Steps 1 to 4 produce a submittable system on their own. 5 onward is where the aiming-high part lives, so if time gets tight the deep candidate is the first thing I cut, not the monitoring or the doc, since those carry 25% + rubric weight for documentation.

## Rubric mapping, so nothing gets dropped

- Problem understanding: opening section of design.md defines users (payment gateway, fraud ops team), inputs, outputs, latency requirement.
- Data prep and model development: temporal split rationale, imbalance handling, three-model comparison with justification.
- Production system: end-to-end DVC pipeline, working API, Docker, registry, the parity test.
- Evaluation and production considerations: PR AUC + recall@FPR + cost threshold, measured latency under load, monitoring and cost discussion.
- Documentation: design.md, README with exact commands, diagram, eval reports, screenshots.
