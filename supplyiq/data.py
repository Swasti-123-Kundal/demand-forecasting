"""Data loading: real M5 (Walmart) subset, or a synthetic FMCG generator for testing.

Common long schema: sku, date, sales, price, promo, festival, snap
"""
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = ["sku", "date", "sales", "price", "promo", "festival", "snap"]

# Approximate dates of major Indian festivals / events (used ONLY by the synthetic generator).
_FEST = {
    "2021": ["2021-01-26", "2021-03-29", "2021-05-14", "2021-08-22", "2021-10-15", "2021-11-04", "2021-12-25"],
    "2022": ["2022-01-26", "2022-03-18", "2022-05-03", "2022-08-11", "2022-10-05", "2022-10-24", "2022-12-25"],
    "2023": ["2023-01-26", "2023-03-08", "2023-04-22", "2023-08-30", "2023-10-24", "2023-11-12", "2023-12-25"],
}
FESTIVAL_DATES = pd.to_datetime([d for v in _FEST.values() for d in v])


# --------------------------------------------------------------------------- M5
def load_m5(data_dir, category="FOODS", store="CA_1", n_items=60, max_days=1095,
            seed=42, min_nonzero_frac=0.10):
    """Load an M5 subset: one store, one category, items sampled evenly across volume
    (so fast movers AND slow / intermittent movers are included)."""
    d = Path(data_dir)
    f = d / "sales_train_evaluation.csv"
    if not f.exists():
        f = d / "sales_train_validation.csv"
    sales = pd.read_csv(f)
    cal = pd.read_csv(d / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(d / "sell_prices.csv")

    sales = sales[(sales.cat_id == category) & (sales.store_id == store)]
    day_cols = [c for c in sales.columns if c.startswith("d_")][-max_days:]
    nz = (sales[day_cols] > 0).mean(axis=1)
    sales = sales[nz >= min_nonzero_frac]
    order = sales[day_cols].sum(axis=1).sort_values().index
    pick = order[np.linspace(0, len(order) - 1, min(n_items, len(order))).astype(int)]
    sales = sales.loc[pick]

    long = sales.melt(id_vars=["item_id"], value_vars=day_cols, var_name="d", value_name="sales")
    state = store.split("_")[0]
    cal = cal.rename(columns={f"snap_{state}": "snap"})
    long = long.merge(cal[["d", "date", "wm_yr_wk", "event_name_1", "snap"]], on="d", how="left")
    pr = prices[prices.store_id == store][["item_id", "wm_yr_wk", "sell_price"]]
    long = long.merge(pr, on=["item_id", "wm_yr_wk"], how="left")
    long = long.dropna(subset=["sell_price"])          # drop days before the item was on sale
    long = long.rename(columns={"item_id": "sku", "sell_price": "price"})
    long = long.sort_values(["sku", "date"])
    med = long.groupby("sku")["price"].transform(lambda s: s.rolling(56, min_periods=14).median())
    long["promo"] = (long["price"] < 0.95 * med).astype(int)   # DERIVED: price >5% below 8-week median
    long["festival"] = long["event_name_1"].notna().astype(int)
    long["snap"] = long["snap"].fillna(0).astype(int)
    out = long[SCHEMA].reset_index(drop=True)
    meta = dict(source=f"M5 (Walmart) - {category}, store {store}", currency="$",
                synthetic=False, n_skus=out.sku.nunique(),
                note="promo = derived from price drops; festival = M5 US calendar events")
    return out, meta


# ----------------------------------------------------------------------- synthetic
def make_synthetic(n_skus=40, start="2021-01-01", days=1095, seed=42):
    """Synthetic FMCG-like demand: trend, weekly + annual seasonality, festival ramps,
    promo price elasticity, AR(1) demand shocks, negative-binomial noise, intermittent SKUs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    fest = dates.isin(FESTIVAL_DATES)
    n = len(dates)
    dtf = np.full(n, 99.0)
    nxt = None
    for i in range(n - 1, -1, -1):
        if fest[i]:
            nxt = i
        if nxt is not None:
            dtf[i] = nxt - i
    t = np.arange(n)
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values
    cats = ["Soap", "Shampoo", "Detergent", "Tea", "Biscuit", "Toothpaste", "Noodles", "Skincare"]
    n_int = max(1, int(0.2 * n_skus))
    frames = []
    for k in range(n_skus):
        cat = cats[k % len(cats)]
        intermittent = k >= n_skus - n_int
        base = rng.uniform(0.3, 1.4) if intermittent else float(np.clip(rng.lognormal(np.log(25), 0.8), 4, 200))
        g = rng.uniform(-0.15, 0.25)
        weekly = 1 + rng.uniform(0.0, 0.35) * np.cos(2 * np.pi * (np.arange(7) - rng.integers(0, 7)) / 7)
        amp, ph = rng.uniform(0.0, 0.25), rng.uniform(0, 365)
        uplift = rng.uniform(0.2, 1.0)
        elast = rng.uniform(1.5, 3.0)
        p0 = float(np.clip(rng.lognormal(np.log(80), 0.7), 15, 600))
        price = np.full(n, p0)
        price[rng.integers(min(200, n // 3), max(min(200, n // 3) + 1, n - 100)):] *= rng.choice([0.95, 1.0, 1.05, 1.08])
        promo = np.zeros(n, dtype=int)
        s = int(rng.integers(5, 40))
        while s < n - 10:
            dur = int(rng.integers(5, 11))
            disc = rng.uniform(0.10, 0.30)
            price[s:s + dur] *= (1 - disc)
            promo[s:s + dur] = 1
            s += int(rng.integers(45, 90))
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.95 * e[i - 1] + rng.normal(0, 0.08 * np.sqrt(1 - 0.95 ** 2) * 3)
        ref_price = pd.Series(price).rolling(56, min_periods=1).median().values
        mu = (base * (1 + g * t / 365) * weekly[dow] * (1 + amp * np.sin(2 * np.pi * (doy - ph) / 365))
              * (1 + uplift * np.exp(-np.minimum(dtf, 30) / 3.0) * (dtf <= 7))
              * (price / ref_price) ** (-elast) * np.exp(e))
        mu = np.clip(mu, 0.01, None)
        r = 1.5 if intermittent else 6.0
        lam = rng.gamma(shape=r, scale=mu / r)
        sales = rng.poisson(lam)
        frames.append(pd.DataFrame(dict(sku=f"{cat[:3].upper()}_{k:03d}", date=dates, sales=sales,
                                        price=price.round(2), promo=promo, festival=fest.astype(int), snap=0)))
    out = pd.concat(frames, ignore_index=True)[SCHEMA]
    meta = dict(source="SYNTHETIC FMCG demand (generator in supplyiq/data.py)", currency="Rs",
                synthetic=True, n_skus=n_skus,
                note="synthetic data for pipeline testing - NOT real market data")
    return out, meta


def load_data(source="synthetic", m5_dir="data", n_skus=40, seed=42):
    if source == "m5":
        return load_m5(m5_dir, n_items=n_skus, seed=seed)
    return make_synthetic(n_skus=n_skus, seed=seed)
