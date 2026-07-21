# Design Doc: Real-Time Fraud Detection Scoring System

Vedant Rajesh Sonkar, Roll No. 2025em1100504
Course: Machine Learning Model Engineering, BITS Pilani Digital
Assignment: Mini Production ML System — Track A (online scoring)

## Problem and metrics

A payment gateway sends a transaction to this API at authorization time and needs a fraud
probability plus an approve/review/decline decision back before the customer's card clears —
end-to-end budget ~200ms, of which the model's share should be well under 50ms. There are two
users: the gateway (needs a fast, reliable decision) and the fraud operations team (needs
calibrated scores and a threshold that reflects real cost, not just a classifier's default 0.5
cutoff). This is inherently request-response: a human is standing at checkout, the decision
blocks money movement, and there's no channel to correct a batch score after the fact. Batch
scoring would mean flagging fraud after the money already left; the hybrid pattern only enters
at the feature layer (heavy aggregates precomputed, light velocity features computed online),
never at the scoring layer.

Because fraud is under 1% of transactions, ROC AUC is misleadingly high even for a mediocre
model, so PR AUC is the headline offline metric. Recall at 1% FPR translates directly to a
business question — "how much fraud do we catch if we're only allowed to annoy 1 in 100
legitimate customers" — and is the number a fraud-ops stakeholder would actually ask for. A
cost metric (missed fraud costs the transaction amount, a false decline costs a flat friction
penalty) picks the deployed threshold rather than leaving it at an arbitrary 0.5.

## Data and features

Dataset: Sparkov Credit Card Transaction Fraud (Kaggle `kartik2112/fraud-detection`), ~1.85M
simulated transactions, Jan 2019–Dec 2020, ~0.5% fraud rate, split by the source into
`fraudTrain.csv` (1,296,675 rows, Jan 2019–Jun 2020) and `fraudTest.csv` (555,719 rows, Jun–Dec
2020). I chose it over ULB (PCA-anonymized, can't build interpretable features) and IEEE-CIS
(hundreds of masked columns) specifically because every feature below is explainable — that's
what Part A grades. It's synthetic, which is a real limitation discussed below.

`make_daily_batches.py` slices `fraudTrain.csv` into 537 daily CSVs to simulate an incoming
feed. `ingest.py` runs as an idempotent micro-batch job: picks up files not yet processed
(tracked in a state file), validates schema, dedupes on `trans_num`, appends to the parquet
training table, and logs rows ingested / date range / null counts per run — one ingestion run
over the full backlog produced 1,296,675 rows, 0 nulls, in one line of log output.

Eleven features, computed by one shared function set (`features/build.py` for offline,
`features/online.py` for the request-time path) so training and serving can never silently
diverge:

| Feature | Offline / Online | Notes |
|---|---|---|
| `amt_log` | pure (both, identical code) | log1p of amount |
| `haversine_km_customer_merchant` | pure (both, identical code) | distance, home vs merchant |
| `hour_sin` / `hour_cos` | pure (both, identical code) | cyclical time encoding |
| `is_night` | pure (both, identical code) | Sparkov concentrates fraud at night |
| `category_fraud_rate_smoothed` | batch precompute, online lookup | Laplace-smoothed, fit on train fold only |
| `txn_count_card_1h` / `_24h` | stateful, computed both sides | velocity; card-testing attacks fire fast |
| `amt_over_card_avg_30d` | stateful, computed both sides | ratio vs the card's own spending pattern |
| `time_since_last_txn_card` | stateful, computed both sides | seconds since the card's prior transaction |
| `card_age_days_in_data` | stateful, computed both sides | new cards behave differently |

The four "pure" features are literally the same Python functions (`features/pure.py`) called
from both paths via `np.vectorize` offline and directly online — there is only one
implementation, so they cannot drift. The stateful features are harder: offline they're
computed with pandas group-rolling windows over the full sorted history; online they're
computed from an incremental per-card `OnlineFeatureStore` (a deque of recent transactions,
trimmed to a 30-day window) that only sees transactions strictly before the current one. This
is the actual skew risk in the system, so `tests/test_feature_parity.py` replays 2,000 real
transactions through both paths in time order and asserts every feature matches to 1e-6 — the
guardrail, not just a description of intent. In production the in-memory store becomes Redis
with TTL'd sorted sets keyed by card id; the math doesn't change, only where the history lives.

## Model choice and evaluation

Split is strictly temporal — I used Kaggle's own file boundary (`fraudTrain.csv` ends
2020-06-21, `fraudTest.csv` starts there) since it's already chronological, and carved a
validation slice out of the tail of `fraudTrain` (train ≤ 2019-12-31, val through 2020-06-21,
test = all of `fraudTest.csv`). A random split would leak future velocity information
backward — a classic fraud-ML mistake I explicitly avoided.

Three models, same feature set, same split:

| Model | Test PR AUC | Test ROC AUC | Recall @1% FPR | p95 latency |
|---|---|---|---|---|
| Baseline: logistic regression (class-weighted) | 0.162 | 0.923 | 0.576 | 0.28 ms |
| Candidate: LightGBM (tuned, `scale_pos_weight`) | 0.839 | 0.994 | 0.935 | 0.26 ms |
| Deep candidate: MLP + category embedding | 0.797 | 0.996 | 0.908 | 1.42 ms |

The promotion gate (`evaluate.py`, not just described — it runs and decided this) requires
PR AUC gain ≥ 0.02, recall-at-1%-FPR delta ≥ 0, and p95 single-row inference ≤ 20ms, evaluated
on the validation fold. LightGBM cleared all three (PR AUC gain +0.65, recall delta +0.34,
0.26ms latency) and was auto-promoted by `train.py` via `registry.py`. The deep MLP is the
honest negative result I expected going in: despite an embedding layer for merchant category
and two hidden layers, it lands ~4 points of PR AUC behind LightGBM on test and costs 5–6x more
per-row latency, for no offsetting gain. Gradient-boosted trees on structured/tabular data with
engineered aggregate features are still the right tool here; a deep model would earn its
serving cost only with much larger data, richer sequence/graph structure (e.g. transaction
graphs across merchants), or representation learning that trees can't express — none of which
apply to this feature set. That negative result is itself the point of building the second
candidate rather than stopping at LightGBM.

## Serving pattern and measured latency

FastAPI service, three endpoints: `POST /predict` (fraud probability, approve/review/decline
decision, model version, latency, feature flags for unseen category / cold-start card),
`GET /health` (liveness + loaded model version), `GET /metrics` (rolling request count, error
rate, latency percentiles, score distribution — the input to the monitoring story below).
Decision bands come from the model's own cost-optimal threshold (0.0399 for LightGBM):
decline ≥ threshold, review ≥ 0.4×threshold, else approve.

Load-tested with `scripts/load_test.py` (asyncio + httpx) against real `fraudTest.csv` rows, at
four concurrency levels, both as a raw `uvicorn` process and as the Docker container:

| Concurrency | Raw p95 (ms) | Docker p95 (ms) |
|---|---|---|
| 1 | 1.11 | 2.95 |
| 10 | 11.41 | 15.97 |
| 50 | 44.27 | 68.58 |
| 100 | 80.42 | 124.70 |

Even at 100 concurrent requests inside Docker, p95 stays under the 200ms end-to-end budget with
headroom, and the model's own inference slice (0.26ms) is a rounding error next to feature
computation and HTTP overhead — the actual latency cost lives in request handling, not the
LightGBM call.

## Pipeline, monitoring, and retraining

DVC (`dvc.yaml`) chains four stages — `make_daily_batches → ingest → train → train_deep` — so
`dvc repro` rebuilds the whole thing deterministically; raw CSVs are tracked with
`dvc add` rather than committed to git. CI (`.github/workflows/ci.yml`) runs ruff, the pytest
suite, and a full smoke train on a 5,000-row synthetic fixture (not the real dataset, so CI
never needs Kaggle credentials) proving ingest → features → train → gate → registry runs clean
on every push.

Monitoring plan:

| Layer | Metric | Threshold | Alert to |
|---|---|---|---|
| Infra | p95/p99 latency, error rate, throughput | p95 > 150ms for 5 min, or error rate > 1% | on-call engineer |
| Data/feature | null rate, out-of-range rate, unseen-category rate, PSI vs training distribution | PSI > 0.2 on any top feature | ML engineer |
| Model/business | score distribution drift, decline rate, rolling PR AUC on labeled feedback (chargebacks lag weeks) | decline rate doubles day/day | fraud team |

`monitoring/drift.py` is the implemented check: null/range violations plus PSI per feature
between the training distribution and a recent batch, logging a WARNING and exiting nonzero on
breach. I manufactured drift by scaling a recent batch's amounts 8x (simulating a currency/unit
bug upstream) and it fired correctly — `amt_log` PSI 3.80, `amt_over_card_avg_30d` PSI 2.38,
both far past the 0.2 threshold. `monitoring/retrain_trigger.py` implements the real decision
function (staleness + label volume, 3 consecutive drift breaches, or PR AUC decay vs the
promoted model) and is exercised by feeding it three consecutive high-PSI days, at which point
it correctly fires.

**Incident scenario.** The upstream gateway renames `amt` to `amount_usd` in the daily feed.
`ingest.py`'s schema validation rejects the malformed files and logs the rejection — the
training table stays clean — but the *online* API keeps receiving live traffic with the new
field name, so every amount-derived feature (`amt_log`, `amt_over_card_avg_30d`, the cost
threshold itself) silently degrades to whatever default/null handling is in place, and
`feature_flags` in responses spikes. The null-rate monitor fires within minutes of the first
drifted batch. Response: freeze the retrain trigger so a bad batch doesn't get trained on,
fall back to a conservative fixed threshold in serving, fix the field mapping in ingestion,
backfill the missed daily files, verify PSI returns to baseline, then unfreeze retraining. This
one scenario exercises schema validation, monitoring, alerting, a rollback decision, and
pipeline repair together, which is why I picked it over a purely offline-metric incident.

## Trade-offs and limitations

Sparkov is synthetic — its fraud is injected in bursts per card, which makes the velocity
features unusually predictive and probably overstates real-world recall; I'd expect a live
system's PR AUC to be meaningfully lower. The online velocity store is in-process memory, which
doesn't survive a restart or scale past one process — a real deployment needs Redis with TTL'd
per-card state, which changes nothing about the feature math but does add an operational
dependency. Label lag (chargebacks arrive weeks later) means the "rolling PR AUC on labeled
feedback" monitor is necessarily stale; PSI on features is the leading indicator that actually
catches drift in near-real-time. The deep candidate was trained on CPU/MPS here rather than a
dedicated GPU, but at this data size that wasn't the bottleneck — 79 seconds total.

## Future work

Move the online feature store to Redis with TTL'd sorted sets so it survives restarts and scales
horizontally. Add calibration (e.g. isotonic regression) on top of LightGBM's raw scores since
the cost-optimal threshold assumes probabilities are meaningful, not just rank-ordered. Shadow-
deploy any future candidate against live traffic before promotion, rather than relying solely on
the offline gate. Extend the category-rate approach to a proper target-encoded merchant-level
feature with more aggressive smoothing for long-tail merchants.
