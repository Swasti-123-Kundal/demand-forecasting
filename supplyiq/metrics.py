import numpy as np
import pandas as pd


def wape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(y - f).sum() / max(y.sum(), 1e-9))


def bias(y, f):
    """Signed: >0 means over-forecast (excess stock), <0 under-forecast (stockouts)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f - y).sum() / max(y.sum(), 1e-9))


def rmse(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def pinball(y, q, tau):
    y, q = np.asarray(y, float), np.asarray(q, float)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def score_models(df, model_cols, y_col="sales"):
    rows = []
    for m in model_cols:
        sub = df[[y_col, m]].dropna()
        rows.append(dict(model=m, WAPE=wape(sub[y_col], sub[m]), Bias=bias(sub[y_col], sub[m]),
                         RMSE=rmse(sub[y_col], sub[m]), n=len(sub)))
    return pd.DataFrame(rows)


def interval_scores(df, y_col="sales"):
    sub = df[[y_col, "p10", "p50", "p90"]].dropna()
    cov = float(((sub[y_col] >= sub["p10"]) & (sub[y_col] <= sub["p90"])).mean())
    return dict(coverage_p10_p90=cov, pinball_p10=pinball(sub[y_col], sub["p10"], 0.1),
                pinball_p50=pinball(sub[y_col], sub["p50"], 0.5), pinball_p90=pinball(sub[y_col], sub["p90"], 0.9))
