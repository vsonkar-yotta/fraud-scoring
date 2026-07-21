"""Pure, stateless per-transaction transforms.

These take a single transaction's fields and return a value with no
reference to history or other rows. Because they're pure functions used
verbatim by both the offline builder and the online path, they can never
drift between training and serving -- there's only one implementation.
"""

import math

import numpy as np


def amt_log(amt: float) -> float:
    return float(np.log1p(max(amt, 0.0)))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return float(2 * r * math.asin(math.sqrt(a)))


def hour_cyclical(hour: int) -> tuple[float, float]:
    angle = 2 * math.pi * hour / 24.0
    return float(math.sin(angle)), float(math.cos(angle))


def is_night(hour: int) -> int:
    return int(hour >= 22 or hour < 6)
