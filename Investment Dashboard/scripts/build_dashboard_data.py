#!/usr/bin/env python3
"""Build the private local JSON payload consumed by the investment dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IBKR_DIR = ROOT / "Transaction Data" / "Interactive Brokers"
DEGIRO_DIR = ROOT / "Transaction Data" / "Degiro"
OUTPUT = ROOT / "public" / "data" / "dashboard.json"
IBKR_FLEX = IBKR_DIR / "Flex" / "ibkr-flex-ytd.xml"
ENV_FILE = ROOT / ".env"
SHEET_RANGE = "Portfolio Rebalancing!A1:U52"

SYMBOL_ALIASES = {
    "AGGH": "EUNA",
    "EUN4": "IEAG",
}
ISIN_TICKERS = {
    "IE00B4L5Y983": "IWDA",
    "IE00BKM4GZ66": "EMIM",
    "IE00BP3QZB59": "IWVL",
    "IE00BF4RFH31": "IUSN",
    "NL0009690239": "TRET",
    "IE00BDBRDM35": "EUNA",
    "IE00B3DKXQ41": "IEAG",
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    cleaned = str(value).replace("€", "").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def percentage(value: Any) -> float:
    parsed = number(value)
    return parsed if abs(parsed) <= 1 else parsed / 100


def read_sheet() -> list[list[str]]:
    spreadsheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    if not spreadsheet_id:
        raise SystemExit("Set GOOGLE_SHEET_ID in Investment Dashboard/.env")
    params = json.dumps(
        {
            "spreadsheetId": spreadsheet_id,
            "range": SHEET_RANGE,
            "valueRenderOption": "UNFORMATTED_VALUE",
            "dateTimeRenderOption": "FORMATTED_STRING",
        }
    )
    command = [
        "gws",
        "sheets",
        "spreadsheets",
        "values",
        "get",
        "--params",
        params,
        "--format",
        "json",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload_start = result.stdout.find("{")
    return json.loads(result.stdout[payload_start:])["values"]


def cell(row: list[str], index: int) -> str:
    return row[index] if index < len(row) else ""


def parse_sheet(rows: list[list[str]]) -> dict[str, Any]:
    holding_rows = rows[3:10]
    holdings = []
    for row in holding_rows:
        holdings.append(
            {
                "ticker": cell(row, 3),
                "name": cell(row, 1),
                "category": cell(row, 4).strip(),
                "price": number(cell(row, 10)),
                "degiroQuantity": number(cell(row, 11)),
                "ibkrQuantity": number(cell(row, 12)),
                "quantity": number(cell(row, 13)),
                "value": number(cell(row, 8)),
                "weight": percentage(cell(row, 5)),
                "targetWeight": percentage(cell(row, 7)),
                "divergence": percentage(cell(row, 14)),
            }
        )

    contributions = []
    for row in rows[8:]:
        date_value = cell(row, 19)
        amount_value = cell(row, 20)
        if not date_value or not amount_value:
            continue
        parsed_date = None
        for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(date_value, fmt).date().isoformat()
                break
            except ValueError:
                continue
        if parsed_date and number(amount_value):
            contributions.append({"date": parsed_date, "amount": number(amount_value)})

    summary_row = holding_rows[0]
    return {
        "sourceUpdatedAt": cell(rows[0], 2),
        "summary": {
            "currentValue": number(cell(summary_row, 15)),
            "investedCapital": number(cell(summary_row, 18)),
            "profitLoss": number(cell(summary_row, 19)),
            "profitLossPercent": percentage(cell(summary_row, 20)),
        },
        "holdings": holdings,
        "contributions": sorted(contributions, key=lambda item: item["date"]),
    }


def statement_sections(path: Path) -> dict[str, list[dict[str, str]]]:
    headers: dict[str, list[str]] = {}
    sections: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            section, row_type = row[0], row[1]
            if row_type == "Header":
                headers[section] = row[2:]
            elif row_type == "Data" and section in headers:
                values = row[2:]
                sections[section].append(dict(zip(headers[section], values, strict=False)))
    return sections


def section_value(rows: list[dict[str, str]], field_name: str) -> float:
    for row in rows:
        if row.get("Field Name") == field_name:
            return number(row.get("Field Value"))
    return 0.0


def parse_ibkr() -> dict[str, Any]:
    annual = []
    trades_by_symbol: dict[str, dict[str, float]] = defaultdict(lambda: {"quantity": 0.0, "cost": 0.0, "fees": 0.0})
    for path in sorted(IBKR_DIR.glob("*.csv")):
        sections = statement_sections(path)
        statement = {row.get("Field Name"): row.get("Field Value") for row in sections.get("Statement", [])}
        period = statement.get("Period", "")
        year = int(period[:4]) if period and period[:4].isdigit() else int(path.stem.split("_")[-1])
        change_rows = sections.get("Change in NAV", [])
        deposits = sum(
            number(row.get("Amount"))
            for row in sections.get("Deposits & Withdrawals", [])
            if row.get("Currency") != "Total"
        )
        dividends = sum(
            number(row.get("Amount")) for row in sections.get("Dividends", []) if row.get("Currency") != "Total"
        )
        fees = sum(number(row.get("Amount")) for row in sections.get("Fees", []))
        fees += sum(number(row.get("Comm/Fee")) for row in sections.get("Trades", []))
        ending_value = section_value(change_rows, "Ending Value")
        starting_value = section_value(change_rows, "Starting Value")
        if not ending_value:
            nav_rows = sections.get("Net Asset Value", [])
            ending_value = sum(
                number(row.get("Current Total"))
                for row in nav_rows
                if row.get("Asset Class") in {"Cash ", "Stock", "Stocks"}
            )

        annual.append(
            {
                "year": year,
                "startingValue": starting_value,
                "endingValue": ending_value,
                "deposits": deposits,
                "dividends": dividends,
                "fees": abs(fees),
                "netChange": ending_value - starting_value - deposits,
            }
        )

        for trade in sections.get("Trades", []):
            symbol = SYMBOL_ALIASES.get(trade.get("Symbol", ""), trade.get("Symbol", ""))
            if not symbol:
                continue
            trades_by_symbol[symbol]["quantity"] += number(trade.get("Quantity"))
            trades_by_symbol[symbol]["cost"] += number(trade.get("Basis"))
            trades_by_symbol[symbol]["fees"] += abs(number(trade.get("Comm/Fee")))

    return {
        "annual": sorted(annual, key=lambda item: item["year"]),
        "tradeSummary": [{"ticker": ticker, **values} for ticker, values in sorted(trades_by_symbol.items())],
        "historyCompleteThrough": max((item["year"] for item in annual), default=None),
    }


def parse_ibkr_flex() -> dict[str, Any] | None:
    if not IBKR_FLEX.exists():
        return None

    root = ET.parse(IBKR_FLEX).getroot()
    statements = list(root.iter("FlexStatement"))
    if not statements:
        return None
    latest = max(statements, key=lambda element: element.attrib.get("toDate", ""))

    positions = []
    for element in latest.iter("OpenPosition"):
        if element.attrib.get("levelOfDetail") != "SUMMARY":
            continue
        symbol = SYMBOL_ALIASES.get(element.attrib.get("symbol", ""), element.attrib.get("symbol", ""))
        positions.append(
            {
                "ticker": symbol,
                "quantity": number(element.attrib.get("position")),
                "value": number(element.attrib.get("positionValue")),
                "markPrice": number(element.attrib.get("markPrice")),
            }
        )

    nav = next(latest.iter("ChangeInNAV"), None)
    return {
        "reportDate": latest.attrib.get("toDate"),
        "generatedAt": latest.attrib.get("whenGenerated"),
        "endingValue": number(nav.attrib.get("endingValue")) if nav is not None else 0,
        "positions": sorted(positions, key=lambda item: item["ticker"]),
    }


def reconcile_flex(data: dict[str, Any], flex: dict[str, Any] | None) -> dict[str, Any] | None:
    if not flex:
        return None
    sheet_positions = {holding["ticker"]: holding["ibkrQuantity"] for holding in data["holdings"]}
    mismatches = []
    for position in flex["positions"]:
        expected = sheet_positions.get(position["ticker"])
        if expected is None or abs(expected - position["quantity"]) > 0.0001:
            mismatches.append(
                {
                    "ticker": position["ticker"],
                    "sheetQuantity": expected,
                    "flexQuantity": position["quantity"],
                }
            )
    flex["positionsReconciled"] = not mismatches and len(flex["positions"]) == len(sheet_positions)
    flex["positionMismatches"] = mismatches
    return flex


def parse_degiro(data: dict[str, Any]) -> dict[str, Any] | None:
    paths = sorted(DEGIRO_DIR.glob("*.csv"))
    if not paths:
        return None

    annual: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {"year": 0, "purchases": 0.0, "fees": 0.0, "transactions": 0}
    )
    holdings: dict[str, dict[str, float | str | int]] = defaultdict(
        lambda: {"ticker": "", "quantity": 0.0, "costBasis": 0.0, "fees": 0.0, "transactions": 0}
    )
    dates = []

    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                isin = row.get("ISIN", "").strip()
                quantity_text = row.get("Quantity", "").strip()
                if isin not in ISIN_TICKERS or not quantity_text:
                    continue
                ticker = ISIN_TICKERS[isin]
                quantity = number(quantity_text)
                cost = -number(row.get("Value EUR"))
                fees = abs(number(row.get("Transaction and/or third party fees EUR")))
                date = datetime.strptime(row["Date"], "%d-%m-%Y").date()
                dates.append(date)

                position = holdings[ticker]
                position["ticker"] = ticker
                position["quantity"] += quantity
                position["costBasis"] += cost
                position["fees"] += fees
                position["transactions"] += 1

                year = annual[date.year]
                year["year"] = date.year
                year["purchases"] += cost
                year["fees"] += fees
                year["transactions"] += 1

    sheet_positions = {holding["ticker"]: holding["degiroQuantity"] for holding in data["holdings"]}
    mismatches = []
    for ticker, position in holdings.items():
        expected = sheet_positions.get(ticker)
        if expected is None or abs(expected - number(position["quantity"])) > 0.0001:
            mismatches.append(
                {
                    "ticker": ticker,
                    "sheetQuantity": expected,
                    "transactionQuantity": position["quantity"],
                }
            )

    for position in holdings.values():
        quantity = number(position["quantity"])
        position["averageCost"] = number(position["costBasis"]) / quantity if quantity else 0

    return {
        "firstTradeDate": min(dates).isoformat() if dates else None,
        "lastTradeDate": max(dates).isoformat() if dates else None,
        "transactionCount": sum(int(item["transactions"]) for item in annual.values()),
        "totalPurchases": sum(number(item["purchases"]) for item in annual.values()),
        "totalFees": sum(number(item["fees"]) for item in annual.values()),
        "annual": sorted(annual.values(), key=lambda item: int(item["year"])),
        "holdings": sorted(holdings.values(), key=lambda item: str(item["ticker"])),
        "positionsReconciled": not mismatches and len(holdings) == len(sheet_positions),
        "positionMismatches": mismatches,
    }


def broker_history(ibkr: dict[str, Any], degiro: dict[str, Any] | None) -> list[dict[str, Any]]:
    years: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"year": 0, "ibkrEndingValue": 0.0, "ibkrDeposits": 0.0, "degiroPurchases": 0.0}
    )
    for item in ibkr["annual"]:
        year = int(item["year"])
        years[year].update({"year": year, "ibkrEndingValue": item["endingValue"], "ibkrDeposits": item["deposits"]})
    if degiro:
        for item in degiro["annual"]:
            year = int(item["year"])
            years[year].update({"year": year, "degiroPurchases": item["purchases"]})
    return sorted(years.values(), key=lambda item: item["year"])


def enrich(data: dict[str, Any]) -> dict[str, Any]:
    summary = data["summary"]
    summary["ibkrValue"] = sum(item["price"] * item["ibkrQuantity"] for item in data["holdings"])
    summary["degiroValue"] = sum(item["price"] * item["degiroQuantity"] for item in data["holdings"])
    summary["equityValue"] = sum(item["value"] for item in data["holdings"] if "bond" not in item["category"].lower())
    summary["bondValue"] = summary["currentValue"] - summary["equityValue"]

    running = 0.0
    contribution_history = []
    for contribution in data["contributions"]:
        running += contribution["amount"]
        contribution_history.append({**contribution, "cumulative": running})
    data["contributionHistory"] = contribution_history
    return data


def main() -> None:
    load_env(ENV_FILE)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet-json", type=Path, help="Optional saved Sheets API JSON response")
    args = parser.parse_args()

    if args.sheet_json:
        rows = json.loads(args.sheet_json.read_text())["values"]
    else:
        rows = read_sheet()

    payload = enrich(parse_sheet(rows))
    payload["ibkr"] = parse_ibkr()
    payload["ibkr"]["flex"] = reconcile_flex(payload, parse_ibkr_flex())
    payload["degiro"] = parse_degiro(payload)
    payload["brokerHistory"] = broker_history(payload["ibkr"], payload["degiro"])
    payload["generatedAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
    payload["dataQuality"] = {
        "combinedCurrentState": "complete",
        "ibkrHistory": "available through annual statements",
        "ibkrFlex": "connected and reconciled"
        if payload["ibkr"]["flex"] and payload["ibkr"]["flex"]["positionsReconciled"]
        else "not reconciled",
        "degiroHistory": "complete and reconciled"
        if payload["degiro"] and payload["degiro"]["positionsReconciled"]
        else "not reconciled",
        "marketPrices": "currently sourced from Google Sheet",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
