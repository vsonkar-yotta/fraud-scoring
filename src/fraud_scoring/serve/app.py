"""FastAPI serving app: /predict, /health, /metrics.

Loads the current promoted model from the registry at startup, keeps a
small in-memory OnlineFeatureStore for velocity features, and scores each
incoming transaction with the same feature code path proven equivalent to
training in tests/test_feature_parity.py.
"""

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI

from fraud_scoring import registry
from fraud_scoring.features.build import FEATURE_COLUMNS
from fraud_scoring.features.online import OnlineFeatureStore
from fraud_scoring.serve.schemas import HealthResponse, MetricsResponse, PredictResponse, TransactionRequest

CONFIG_PATH = os.environ.get("FRAUD_SERVE_CONFIG", "configs/serve.yaml")


class AppState:
    model = None
    metadata: dict = {}
    threshold: float = 0.5
    review_threshold: float = 0.2
    feature_store: OnlineFeatureStore = None
    latencies: deque = deque(maxlen=2000)
    scores: deque = deque(maxlen=2000)
    request_count: int = 0
    error_count: int = 0
    decline_count: int = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())
    model, metadata = registry.load_current(Path(cfg["models_dir"]))
    state.model = model
    state.metadata = metadata
    state.threshold = metadata["threshold"]
    state.review_threshold = metadata["threshold"] * cfg["review_band_fraction"]
    state.feature_store = OnlineFeatureStore()
    state.latencies = deque(maxlen=cfg["metrics_window_size"])
    state.scores = deque(maxlen=cfg["metrics_window_size"])
    yield


app = FastAPI(title="fraud-scoring", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(txn: TransactionRequest) -> PredictResponse:
    t0 = time.perf_counter()
    state.request_count += 1
    try:
        txn_dict = txn.model_dump()
        category_rates = state.metadata["category_rates"]

        flags = []
        if txn.category not in category_rates["rates"]:
            flags.append("unseen_category")
        card_state = state.feature_store._cards.get(txn.cc_num)
        if card_state is None:
            flags.append("cold_start_card")

        feats = state.feature_store.compute_features(txn_dict, category_rates)
        state.feature_store.observe(txn_dict)

        X = pd.DataFrame([feats])[FEATURE_COLUMNS]
        prob = float(state.model.predict_proba(X)[0, 1])

        if prob >= state.threshold:
            decision = "decline"
            state.decline_count += 1
        elif prob >= state.review_threshold:
            decision = "review"
        else:
            decision = "approve"

        latency_ms = (time.perf_counter() - t0) * 1000
        state.latencies.append(latency_ms)
        state.scores.append(prob)

        return PredictResponse(
            fraud_probability=prob,
            decision=decision,
            model_version=state.metadata["version"],
            latency_ms=latency_ms,
            feature_flags=flags,
        )
    except Exception:
        state.error_count += 1
        raise


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=state.metadata.get("version", "unknown"))


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    lat = list(state.latencies)
    scr = list(state.scores)
    return MetricsResponse(
        request_count=state.request_count,
        error_count=state.error_count,
        error_rate=state.error_count / state.request_count if state.request_count else 0.0,
        latency_p50_ms=float(np.percentile(lat, 50)) if lat else None,
        latency_p95_ms=float(np.percentile(lat, 95)) if lat else None,
        latency_p99_ms=float(np.percentile(lat, 99)) if lat else None,
        score_mean=float(np.mean(scr)) if scr else None,
        score_p95=float(np.percentile(scr, 95)) if scr else None,
        decline_rate=state.decline_count / state.request_count if state.request_count else None,
    )
