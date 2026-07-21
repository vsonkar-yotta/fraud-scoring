from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

MODELS_DIR = Path("models")

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "registry.json").exists(), reason="no promoted model; run train.py first"
)


@pytest.fixture(scope="module")
def client():
    from fraud_scoring.serve.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_payload():
    row = pd.read_csv("data/raw/fraudTest.csv", nrows=1).iloc[0]
    return {
        "trans_num": row["trans_num"],
        "trans_date_trans_time": str(row["trans_date_trans_time"]),
        "cc_num": str(row["cc_num"]),
        "merchant": row["merchant"],
        "category": row["category"],
        "amt": float(row["amt"]),
        "city": row["city"],
        "state": row["state"],
        "zip": str(row["zip"]),
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "city_pop": int(row["city_pop"]),
        "job": row["job"],
        "dob": str(row["dob"]),
        "merch_lat": float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),
    }


def test_health_reports_ok_and_model_version(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_predict_contract(client, sample_payload):
    resp = client.post("/predict", json=sample_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["model_version"]
    assert body["latency_ms"] >= 0
    assert isinstance(body["feature_flags"], list)


def test_predict_rejects_invalid_amount(client, sample_payload):
    bad = {**sample_payload, "amt": -5.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_metrics_reflects_request_count(client, sample_payload):
    before = client.get("/metrics").json()["request_count"]
    client.post("/predict", json=sample_payload)
    after = client.get("/metrics").json()["request_count"]
    assert after == before + 1
