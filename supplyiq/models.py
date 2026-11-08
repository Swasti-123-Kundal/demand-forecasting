"""Baselines (naive, moving average, ETS) and LightGBM point + quantile models."""
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_jobs=4, verbose=-1)
QUANTILES = (0.1, 0.5, 0.9)


# ------------------------------------------------------------------ LightGBM
def train_lgbm_suite(train, features, seed=0):
    """Tweedie point model (handles zeros / intermittency) + P10/P50/P90 quantile models."""
    X, y = train[features], train["sales"]
    models = {"point": LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2,
                                     random_state=seed, **LGB_PARAMS).fit(X, y)}
    for q in QUANTILES:
        models[f"p{int(q * 100)}"] = LGBMRegressor(objective="quantile", alpha=q,
                                                   random_state=seed, **LGB_PARAMS).fit(X, y)
    return models


def predict_suite(models, X):
    out = pd.DataFrame(index=X.index)
    out["ml_point"] = np.clip(models["point"].predict(X), 0, None)
    q = np.column_stack([models["p10"].predict(X), models["p50"].predict(X), models["p90"].predict(X)])
    q = np.clip(np.sort(q, axis=1), 0, None)          # fix quantile crossing
    out["p10"], out["p50"], out["p90"] = q[:, 0], q[:, 1], q[:, 2]
    return out


# ----------------------------------------------------------------------- ETS
def _ets_block(y, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = y.dropna()
    if len(y) < 28 or y.sum() <= 0:
        return np.full(horizon, float(y.tail(28).mean()) if len(y) else 0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y.values.astype(float), trend=None, seasonal="add",
                                       seasonal_periods=7, initialization_method="estimated").fit()
            f = fit.forecast(horizon)
        return np.clip(f, 0, None)
    except Exception:
        return np.full(horizon, float(y.tail(28).mean()))


def ets_predictions(wide_sales, start, end, block=7, hist=364, n_jobs=-1):
    """Rolling-origin ETS (Holt-Winters, additive weekly seasonality).
    Every `block` days: refit on data up to the day BEFORE the block, forecast the block.
    (Info is at most as stale as the ML model's 7-day lag -> comparison is not unfair to ETS.)"""
    dates = wide_sales.index
    starts = [d for d in dates[(dates >= start) & (dates <= end)][::block]]
    tasks = []
    for b in starts:
        h = min(block, int((dates[dates >= b] <= end).sum()))
        past = wide_sales.loc[:b - pd.Timedelta(days=1)].tail(hist)
        for sku in wide_sales.columns:
            tasks.append((b, sku, past[sku], h))
    res = Parallel(n_jobs=n_jobs)(delayed(_ets_block)(y, h) for _, _, y, h in tasks)
    rows = []
    for (b, sku, _, h), f in zip(tasks, res):
        idx = dates[dates >= b][:h]
        rows.append(pd.DataFrame({"date": idx, "sku": sku, "ets": f}))
    return pd.concat(rows, ignore_index=True)
