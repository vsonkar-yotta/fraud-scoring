"""Async load test: fires concurrent /predict requests sampled from real
test rows, reports avg/p50/p95/p99 latency and requests/sec at several
concurrency levels.

Usage:
    python scripts/load_test.py --url http://localhost:8000 \
        --source data/raw/fraudTest.csv --n 1000 --concurrency 1 10 50 100
"""

import argparse
import asyncio
import time

import httpx
import numpy as np
import pandas as pd


def row_to_payload(row: pd.Series) -> dict:
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


async def fire(client: httpx.AsyncClient, url: str, payload: dict, sem: asyncio.Semaphore) -> float | None:
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/predict", json=payload, timeout=10)
            r.raise_for_status()
        except Exception:
            return None
        return (time.perf_counter() - t0) * 1000


async def run_level(url: str, payloads: list[dict], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[fire(client, url, p, sem) for p in payloads])
        elapsed = time.perf_counter() - t0

    latencies = [r for r in results if r is not None]
    errors = len(results) - len(latencies)
    return {
        "concurrency": concurrency,
        "n_requests": len(payloads),
        "errors": errors,
        "elapsed_s": elapsed,
        "requests_per_sec": len(payloads) / elapsed,
        "avg_ms": float(np.mean(latencies)) if latencies else None,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--source", default="data/raw/fraudTest.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 50, 100])
    args = parser.parse_args()

    df = pd.read_csv(args.source, nrows=args.n)
    payloads = [row_to_payload(row) for _, row in df.iterrows()]

    print(f"{'concurrency':>12} {'req/s':>10} {'avg_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10} {'errors':>8}")
    for c in args.concurrency:
        result = asyncio.run(run_level(args.url, payloads, c))
        print(f"{result['concurrency']:>12} {result['requests_per_sec']:>10.1f} "
              f"{result['avg_ms']:>10.2f} {result['p50_ms']:>10.2f} "
              f"{result['p95_ms']:>10.2f} {result['p99_ms']:>10.2f} {result['errors']:>8}")


if __name__ == "__main__":
    main()
