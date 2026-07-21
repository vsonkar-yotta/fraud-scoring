from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    trans_num: str | None = None
    trans_date_trans_time: datetime
    cc_num: str
    merchant: str
    category: str
    amt: float = Field(gt=0)
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    merch_lat: float
    merch_long: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float
    decision: str  # approve | review | decline
    model_version: str
    latency_ms: float
    feature_flags: list[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    error_rate: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    score_mean: float | None
    score_p95: float | None
    decline_rate: float | None
