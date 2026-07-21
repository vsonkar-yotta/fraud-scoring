"""Online feature computation: request-time path used by the serving API.

Holds a small in-memory per-card state store (recent transaction
timestamps/amounts) that lets us compute the same velocity and rolling-
average features `build.py` computes offline, one transaction at a time,
using only transactions that happened strictly before the current one --
exactly what `build_offline_features` does with its rolling windows.

At production scale this store becomes Redis with TTL'd sorted sets keyed
by card id; the computation logic itself doesn't change, only where the
history lives.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fraud_scoring.features import pure
from fraud_scoring.features.build import TIME_SINCE_SENTINEL_SECONDS, lookup_category_rate

MAX_WINDOW = timedelta(days=30)


@dataclass
class CardState:
    history: deque = field(default_factory=deque)  # (timestamp, amt), oldest first
    first_seen: datetime | None = None

    def trim(self, now: datetime) -> None:
        cutoff = now - MAX_WINDOW
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def append(self, ts: datetime, amt: float) -> None:
        if self.first_seen is None:
            self.first_seen = ts
        self.history.append((ts, amt))
        self.trim(ts)


class OnlineFeatureStore:
    """Card-keyed rolling history. One process's worth of in-memory state."""

    def __init__(self) -> None:
        self._cards: dict[str, CardState] = {}

    def seed(self, ts: datetime, cc_num: str, amt: float) -> None:
        """Feed a historical transaction into state without scoring it."""
        state = self._cards.setdefault(cc_num, CardState())
        state.append(ts, amt)

    def compute_features(self, txn: dict, category_rates: dict) -> dict:
        """Compute the full feature vector for one incoming transaction.

        `txn` must have: cc_num, trans_date_trans_time (datetime), amt,
        category, lat, long, merch_lat, merch_long.
        Does not mutate state -- call `observe()` after scoring to record it.
        """
        ts = txn["trans_date_trans_time"]
        cc_num = txn["cc_num"]
        amt = txn["amt"]
        state = self._cards.get(cc_num)

        if state is not None:
            state.trim(ts)
            prior = list(state.history)
        else:
            prior = []

        count_1h = sum(1 for pts, _ in prior if ts - pts <= timedelta(hours=1))
        count_24h = sum(1 for pts, _ in prior if ts - pts <= timedelta(hours=24))
        amounts_30d = [pamt for pts, pamt in prior if ts - pts <= timedelta(days=30)]

        if amounts_30d:
            avg_prior_30d = sum(amounts_30d) / len(amounts_30d)
            amt_over_card_avg_30d = amt / avg_prior_30d if avg_prior_30d else 1.0
        else:
            amt_over_card_avg_30d = 1.0

        if prior:
            time_since_last = (ts - prior[-1][0]).total_seconds()
        else:
            time_since_last = TIME_SINCE_SENTINEL_SECONDS

        first_seen = state.first_seen if state is not None and state.first_seen is not None else ts
        card_age_days = (ts - first_seen).total_seconds() / 86400.0

        hour = ts.hour
        hour_sin, hour_cos = pure.hour_cyclical(hour)

        return {
            "amt_log": pure.amt_log(amt),
            "amt_over_card_avg_30d": amt_over_card_avg_30d,
            "txn_count_card_1h": float(count_1h),
            "txn_count_card_24h": float(count_24h),
            "time_since_last_txn_card": time_since_last,
            "haversine_km_customer_merchant": pure.haversine_km(
                txn["lat"], txn["long"], txn["merch_lat"], txn["merch_long"]
            ),
            "is_night": pure.is_night(hour),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "category_fraud_rate_smoothed": lookup_category_rate(txn["category"], category_rates),
            "card_age_days_in_data": card_age_days,
        }

    def observe(self, txn: dict) -> None:
        """Record a transaction into state after it has been scored."""
        self.seed(txn["trans_date_trans_time"], txn["cc_num"], txn["amt"])
