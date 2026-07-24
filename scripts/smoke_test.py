"""Post-deploy smoke test: verify the API is healthy and can answer.

Usage:
    python scripts/smoke_test.py [--base-url http://localhost:8000]

Exits non-zero if any check fails, making it suitable for CI/CD gates.
"""

from __future__ import annotations

import argparse
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the backend API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    try:
        response = httpx.get(f"{base}/api/health", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"FAIL: health check unreachable: {exc}", file=sys.stderr)
        return 1

    body = response.json()
    if body.get("status") != "ok":
        print(f"FAIL: unexpected health payload: {body}", file=sys.stderr)
        return 1

    print(f"OK: {body['version']} on {body['environment']} "
          f"(llm={body['llm']}, embeddings={body['embedding_model']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
