"""Forecast -> inventory decision: safety stock, reorder point, EOQ, policy construction."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import MIN_LAG

POLICIES = ["A: MA + fixed SS", "A+: MA + sigma-demand SS", "B1: ML + residual-sigma SS", "B2: ML + quantile-sigma SS"]


def z_from_service(sl):
    return float(norm.ppf(sl))


def safety_stock(z, sigma_daily, lead_time):
    """SS = z * sigma_error * sqrt(L)   (sigma_error = std of daily forecast error)"""
    return z * sigma_daily * np.sqrt(lead_time)


def reorder_point(mean_daily, lead_time, ss):
    """ROP = demand during lead time + safety stock."""
    return mean_daily * lead_time + ss


def eoq(annual_demand, order_cost, holding_cost_per_unit_year):
    return np.maximum(1.0, np.ceil(np.sqrt(2 * np.asarray(annual_demand) * order_cost
                                           / np.maximum(holding_cost_per_unit_year, 1e-9))))


def make_econ(df_train, skus):
    """Per-SKU economics from the training period: mean price and mean daily demand."""
    g = df_train.groupby("sku")
    return pd.DataFrame({"price": g["price"].mean(), "mean_daily": g["sales"].mean()}).loc[list(skus)]


def econ_arrays(econ, p):
    price = econ["price"].values
    unit_cost = p.unit_cost_ratio * price
    return dict(
        h_year=p.holding_rate_per_year * unit_cost,
        h_day=p.holding_rate_per_year * unit_cost / 365.0,
        order_cost=p.order_cost_frac * price * np.maximum(econ["mean_daily"].values, 0.05),
        c_short=p.shortage_cost_mult * price,
        unit_cost=unit_cost,
    )


def build_policies(hist, pred, sig_q, sig_resid, econ, p, service_level=None, pad=28):
    """Return {policy: dict(mu, ss, Q)} where, for each day t of the test window,
       mu[t] = forecast demand over the next L days, ss[t] = safety stock, Q = EOQ.
    hist  : (pad + T, K) actual sales incl. `pad` days before the test window
    pred  : (T, K) ML point forecasts (made with >= 7 day information lag)
    sig_q : (T, K) calibrated daily sigma from the P10-P90 spread
    sig_resid: (K,) std of daily ML forecast error on the calibration window"""
    sl = p.service_level if service_level is None else service_level
    z = z_from_service(sl)
    T, K = pred.shape
    L = int(p.lead_time_days)
    if L > MIN_LAG:
        raise ValueError(f"lead_time_days={L} > MIN_LAG={MIN_LAG}: ML forecasts would use information not yet available at decision time.")
    Te = T - L
    ea = econ_arrays(econ, p)
    H = pd.DataFrame(hist)
    ma = H.rolling(28, min_periods=14).mean().values[pad:pad + T]
    sd = H.rolling(28, min_periods=14).std().fillna(0).values[pad:pad + T]
    zero = np.zeros((1, K))
    c1 = np.vstack([zero, np.cumsum(pred, axis=0)])
    c2 = np.vstack([zero, np.cumsum(sig_q ** 2, axis=0)])
    t = np.arange(Te)
    mu_ml = c1[t + L + 1] - c1[t + 1]
    sig_lt = np.sqrt(np.maximum(c2[t + L + 1] - c2[t + 1], 0))
    mu_ma = ma[:Te] * L

    ss = {
        POLICIES[0]: np.tile(p.fixed_cover_days * econ["mean_daily"].values, (Te, 1)),
        POLICIES[1]: safety_stock(z, sd[:Te], L),
        POLICIES[2]: np.tile(safety_stock(z, sig_resid, L), (Te, 1)),
        POLICIES[3]: z * sig_lt,
    }
    mu = {POLICIES[0]: mu_ma, POLICIES[1]: mu_ma, POLICIES[2]: mu_ml, POLICIES[3]: mu_ml}
    out = {}
    for name in POLICIES:
        daily = (mu[name] / L).mean(axis=0)
        Q = eoq(daily * 365, ea["order_cost"], ea["h_year"])
        out[name] = dict(mu=mu[name], ss=ss[name], Q=Q)
    return out
