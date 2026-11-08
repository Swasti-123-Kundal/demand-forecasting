"""Leakage-safe feature engineering.

Rule: the feature row for date d may only use SALES information up to d - MIN_LAG.
Price / promo / festival / calendar are *planned, known-in-advance* covariates, so
they may be used for day d itself.
"""
import numpy as np
import pandas as pd
from .config import MIN_LAG

FEATURES = [
    "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56",
    "roll_std_7", "roll_std_28", "roll_nz_28", "roll_max_28",
    "price", "price_ratio", "promo", "festival", "snap",
    "days_to_fest", "days_since_fest",
    "dow", "dom", "month", "weekofyear", "is_weekend", "sku_id",
]


def _days_to_from(fest, cap=14):
    n = len(fest)
    dtf = np.full(n, float(cap))
    dsf = np.full(n, float(cap))
    nxt = None
    for i in range(n - 1, -1, -1):
        if fest[i]:
            nxt = i
        if nxt is not None:
            dtf[i] = min(cap, nxt - i)
    prv = None
    for i in range(n):
        if fest[i]:
            prv = i
        if prv is not None:
            dsf[i] = min(cap, i - prv)
    return dtf, dsf


def build_features(df, min_lag=MIN_LAG):
    df = df.sort_values(["sku", "date"]).reset_index(drop=True).copy()
    wide_s = df.pivot(index="date", columns="sku", values="sales")
    wide_p = df.pivot(index="date", columns="sku", values="price")
    sh = wide_s.shift(min_lag)

    wide = {}
    for l in (7, 14, 21, 28):
        assert l >= min_lag
        wide[f"lag_{l}"] = wide_s.shift(l)
    for w in (7, 14, 28, 56):
        wide[f"roll_mean_{w}"] = sh.rolling(w, min_periods=w).mean()
    for w in (7, 28):
        wide[f"roll_std_{w}"] = sh.rolling(w, min_periods=w).std()
    wide["roll_nz_28"] = (sh > 0).astype(float).where(sh.notna()).rolling(28, min_periods=28).mean()
    wide["roll_max_28"] = sh.rolling(28, min_periods=28).max()
    wide["price_ratio"] = wide_p / wide_p.rolling(56, min_periods=7).median()

    di = wide_s.index.get_indexer(df["date"])
    si = wide_s.columns.get_indexer(df["sku"])
    for name, w in wide.items():
        df[name] = w.values[di, si]

    fest_by_date = df.groupby("date")["festival"].max().reindex(wide_s.index).fillna(0).values.astype(bool)
    dtf, dsf = _days_to_from(fest_by_date)
    df["days_to_fest"] = dtf[di]
    df["days_since_fest"] = dsf[di]

    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["dom"] = d.dt.day
    df["month"] = d.dt.month
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    skus = sorted(df["sku"].unique())
    df["sku_id"] = pd.Categorical(df["sku"], categories=skus)
    df["train_ok"] = df["roll_mean_56"].notna()
    return df, wide_s
