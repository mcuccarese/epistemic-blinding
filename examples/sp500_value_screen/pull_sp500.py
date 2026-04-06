"""
Pull S&P 500 fundamentals from yfinance for epistemic blinding experiment.

Outputs ratios and percentages ONLY — no absolute values (revenue, market cap,
share price, employee count) that could fingerprint individual companies.

Usage:
    python pull_sp500.py              # writes sp500_fundamentals.csv
    python pull_sp500.py --out FILE   # custom output path
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import io

import pandas as pd
import requests
import yfinance as yf


# --------------- S&P 500 constituent list --------------- #

def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (epistemic-blinding research)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return sorted(tickers)


# --------------- metric extraction --------------- #

def extract_fundamentals(ticker: str) -> dict | None:
    """Pull ratio-only fundamentals for one ticker. Returns None on failure."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        print(f"  [skip] {ticker}: {e}", file=sys.stderr)
        return None

    if not info or info.get("quoteType") not in ("EQUITY", None):
        return None

    def _get(*keys):
        for k in keys:
            v = info.get(k)
            if v is not None:
                return v
        return None

    row = {
        "ticker": ticker,
        "sector": _get("sector"),
        # --- Layer 1: Valuation ---
        "pe_trailing": _get("trailingPE"),
        "pe_forward": _get("forwardPE"),
        "ps_ratio": _get("priceToSalesTrailing12Months"),
        "pb_ratio": _get("priceToBook"),
        "ev_ebitda": _get("enterpriseToEbitda"),
        "peg_ratio": _get("pegRatio", "trailingPegRatio"),
        "fcf_yield": None,  # derived below
        # --- Layer 2: Growth ---
        "revenue_growth": _get("revenueGrowth"),
        "earnings_growth": _get("earningsGrowth"),
        # --- Layer 3: Quality & Financial Health ---
        "operating_margin": _get("operatingMargins"),
        "profit_margin": _get("profitMargins"),
        "gross_margin": _get("grossMargins"),
        "roe": _get("returnOnEquity"),
        "roa": _get("returnOnAssets"),
        "debt_to_equity": None,  # scaled below
        "current_ratio": _get("currentRatio"),
        # --- Layer 4: Shareholder Returns ---
        "dividend_yield": _get("dividendYield"),
    }

    # FCF yield = free cash flow / market cap  (both absolute, but ratio is safe)
    fcf = _get("freeCashflow")
    mcap = _get("marketCap")
    if fcf is not None and mcap and mcap > 0:
        row["fcf_yield"] = fcf / mcap

    # debt_to_equity from yfinance is already a ratio but sometimes in percent
    dte = _get("debtToEquity")
    if dte is not None:
        row["debt_to_equity"] = dte / 100.0 if dte > 10 else dte  # normalize

    return row


# --------------- main --------------- #

def main():
    ap = argparse.ArgumentParser(description="Pull S&P 500 fundamentals")
    ap.add_argument("--out", default="sp500_fundamentals.csv", help="Output CSV path")
    args = ap.parse_args()

    print("Fetching S&P 500 constituent list...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers.")

    print("Pulling fundamentals (this takes a few minutes)...")
    rows = []
    for i, tk in enumerate(tickers, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(tickers)}] {tk}...")
        row = extract_fundamentals(tk)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    # Drop companies with no sector (usually indicates bad data)
    df = df.dropna(subset=["sector"])

    # Round floats for readability
    float_cols = [c for c in df.columns if c not in ("ticker", "sector")]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(4)

    # Add pull timestamp as metadata
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} companies to {out}")
    print(f"Pull timestamp: {ts}")

    # Write timestamp sidecar
    meta = out.with_suffix(".meta.txt")
    meta.write_text(f"pull_timestamp: {ts}\nticker_count: {len(df)}\n")

    # Summary stats
    print(f"\nSectors: {df['sector'].nunique()}")
    print(f"Missing-value rates:")
    for c in float_cols:
        pct = df[c].isna().mean() * 100
        if pct > 0:
            print(f"  {c}: {pct:.1f}%")


if __name__ == "__main__":
    main()
