#!/usr/bin/env python3
"""Download the configured IBKR Activity Flex Query into the private data folder."""

from __future__ import annotations

import argparse
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
OUTPUT = ROOT / "Transaction Data" / "Interactive Brokers" / "Flex" / "ibkr-flex-ytd.xml"
BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
USER_AGENT = "portfolio-management-dashboard/0.1"
RETRYABLE_CODES = {"1001", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1019", "1021"}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def request(path: str, params: dict[str, str]) -> bytes:
    url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def response_fields(content: bytes) -> dict[str, str]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return {}
    return {child.tag: (child.text or "").strip() for child in root}


def fail_message(fields: dict[str, str]) -> str:
    code = fields.get("ErrorCode", "unknown")
    message = fields.get("ErrorMessage", "IBKR Flex request failed")
    return f"IBKR Flex error {code}: {message}"


def fetch(token: str, query_id: str, attempts: int = 8, delay: int = 8) -> Path:
    send = request("SendRequest", {"t": token, "q": query_id, "v": "3"})
    fields = response_fields(send)
    if fields.get("Status") != "Success" or not fields.get("ReferenceCode"):
        raise RuntimeError(fail_message(fields))

    reference = fields["ReferenceCode"]
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            time.sleep(delay)
        statement = request("GetStatement", {"t": token, "q": reference, "v": "3"})
        statement_fields = response_fields(statement)
        if statement_fields.get("Status") in {"Fail", "Warn"}:
            code = statement_fields.get("ErrorCode", "")
            if code in RETRYABLE_CODES and attempt < attempts:
                continue
            raise RuntimeError(fail_message(statement_fields))
        if b"<FlexStatements" not in statement:
            raise RuntimeError("IBKR returned an unexpected non-Flex response")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(statement)
        return OUTPUT

    raise RuntimeError("IBKR Flex report did not become available before retries were exhausted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optional", action="store_true", help="Skip cleanly when Flex credentials are not configured")
    args = parser.parse_args()

    load_env(ENV_FILE)
    token = os.getenv("IBKR_FLEX_TOKEN", "").strip()
    query_id = os.getenv("IBKR_FLEX_QUERY_ID", "").strip()
    if not token or not query_id:
        if args.optional:
            print("IBKR Flex credentials not configured; using existing private statement files.")
            return
        raise SystemExit("Set IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID in Investment Dashboard/.env")

    output = fetch(token, query_id)
    print(f"Downloaded IBKR Flex report to {output}")


if __name__ == "__main__":
    main()
