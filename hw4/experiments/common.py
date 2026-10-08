"""Shared loaders for the A04 experiments: the course dataset and the raw UCI log."""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parent.parent
RAW_XLSX = APP / "data_raw" / "Online Retail.xlsx"
RAW_CACHE = APP / "data_raw" / "online_retail.pkl"

# Codes that are charges or bookkeeping rather than products. Together with the
# row filters below this reproduces transactions.js exactly (checked in
# run_experiment.py), which is what lets every basket get its InvoiceDate back.
NON_PRODUCT = {"POST", "DOT", "C2", "M", "BANK CHARGES", "PADS", "AMAZONFEE", "CRUK",
               "D", "S", "B",
               "23444",   # NEXT DAY CARRIAGE
               "23574"}   # PACKING CHARGE


def load_course_dataset() -> dict:
    text = (APP / "transactions.js").read_text(encoding="utf-8")
    data = json.loads(re.search(r"window\.HW4 = (\{.*\});\s*$", text, re.S).group(1))
    data["basket_sets"] = [frozenset(data["stocks"][i] for i in b) for b in data["baskets"]]
    data["description"] = dict(zip(data["stocks"], data["descriptions"]))
    return data


def load_raw() -> pd.DataFrame:
    if RAW_CACHE.exists():
        return pd.read_pickle(RAW_CACHE)
    df = pd.read_excel(RAW_XLSX, dtype={"InvoiceNo": str, "StockCode": str})
    df.to_pickle(RAW_CACHE)
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning described in week4/readme.md §3 to the raw log."""
    x = df.copy()
    x["InvoiceNo"] = x["InvoiceNo"].astype(str).str.strip()
    x["StockCode"] = x["StockCode"].astype(str).str.strip().str.upper()
    x = x[~x["InvoiceNo"].str.startswith(("C", "A"))]
    x = x[(x["Quantity"] > 0) & (x["UnitPrice"] > 0)]
    x = x[x["Description"].notna() & (x["Description"].astype(str).str.strip() != "")]
    x = x[x["CustomerID"].notna()]
    x = x[~x["StockCode"].isin({c.upper() for c in NON_PRODUCT})]
    x = x[~x["StockCode"].str.startswith("GIFT_")]
    sizes = x.groupby("InvoiceNo")["StockCode"].nunique()
    x = x[x["InvoiceNo"].isin(sizes[sizes >= 2].index)]
    return x


def dated_baskets(df_clean: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice: the item set, its date, customer and country."""
    g = df_clean.groupby("InvoiceNo")
    out = pd.DataFrame({
        "items": g["StockCode"].agg(frozenset),
        "date": g["InvoiceDate"].min(),
        "customer": g["CustomerID"].first().astype(int),
        "country": g["Country"].first(),
    }).reset_index()
    return out


def same_multiset(a, b) -> tuple[int, int]:
    ca, cb = collections.Counter(a), collections.Counter(b)
    return sum((ca - cb).values()), sum((cb - ca).values())
