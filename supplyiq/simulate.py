"""Monte Carlo simulation of a continuous-review (ROP, EOQ) policy with lost sales.

Scenarios = perturbed replays of the held-out demand period:
  * demand shocks : actual demand x correlated log-normal noise (sigma = demand_sigma)
  * lead-time disturbance : each order's real lead time ~ N(L, sd), integer, >= 1
Every policy sees the SAME random scenarios (common random numbers), so differences
between policies are due to the policy, not luck.
"""
from dataclasses import dataclass
import numpy as np
from .inventory import econ_arrays


@dataclass
class Scenarios:
    demand: np.ndarray   # (S, T, K)
    lt: np.ndarray       # (S, T, K) integer lead times


def make_scenarios(true_demand, n_scen, p, seed=0, demand_sigma=0.20, rho=0.7):
    rng = np.random.default_rng(seed)
    T, K = true_demand.shape
    eps = rng.standard_normal((n_scen, T, K)).astype(np.float32)
    e = np.empty_like(eps)
    e[:, 0] = eps[:, 0]
    a = np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        e[:, t] = rho * e[:, t - 1] + a * eps[:, t]
    m = np.exp(demand_sigma * e - demand_sigma ** 2 / 2)
    d = np.rint(true_demand[None].astype(np.float32) * m).astype(np.float32)
    L = p.lead_time_days
    lt = np.clip(np.rint(L + p.lead_time_sd_days * rng.standard_normal((n_scen, T, K))), 1, 3 * L).astype(np.int16)
    return Scenarios(d, lt)


def simulate(scen, mu, ss, Q, econ, p, init_onhand, warmup=21):
    S, T, K = scen.demand.shape
    assert mu.shape[0] >= T and ss.shape[0] >= T
    ea = econ_arrays(econ, p)
    onhand = np.tile(init_onhand.astype(float), (S, 1))
    onorder = np.zeros((S, K))
    maxlt = int(scen.lt.max())
    arrivals = np.zeros((S, K, T + maxlt + 1))
    s_idx, k_idx = np.meshgrid(np.arange(S), np.arange(K), indexing="ij")
    dem_tot = np.zeros((S, K)); lost_tot = np.zeros((S, K)); so_days = np.zeros((S, K))
    inv_sum = np.zeros((S, K)); n_orders = np.zeros((S, K))
    path = np.zeros((T, K))
    for t in range(T):
        rec = arrivals[:, :, t]
        onhand += rec
        onorder -= rec
        d = scen.demand[:, t, :]
        sold = np.minimum(onhand, d)
        lost = d - sold
        onhand -= sold
        if t >= warmup:
            dem_tot += d; lost_tot += lost; so_days += (lost > 0); inv_sum += onhand
        rop = mu[t] + ss[t]
        ip = onhand + onorder
        need = ip <= rop[None, :]
        if need.any():
            n = np.maximum(1.0, np.ceil((rop[None, :] - ip) / Q[None, :]))
            q = np.where(need, n * Q[None, :], 0.0)
            lt = scen.lt[:, t, :]
            arrivals[s_idx[need], k_idx[need], t + lt[need]] += q[need]
            onorder += q
            if t >= warmup:
                n_orders += need
        path[t] = onhand[0]
    nd = T - warmup
    hold = inv_sum * ea["h_day"][None, :]
    ordc = n_orders * ea["order_cost"][None, :]
    short = lost_tot * ea["c_short"][None, :]
    return dict(demand=dem_tot, lost=lost_tot, stockout_days=so_days,
                avg_onhand=inv_sum / nd, avg_inv_value=(inv_sum / nd) * ea["unit_cost"][None, :],
                n_orders=n_orders, holding_cost=hold, order_cost=ordc, shortage_cost=short,
                total_cost=hold + ordc + short, path0=path)


def summarize(res):
    """Aggregate over SKUs for each scenario, then mean / P5 / P95 over scenarios."""
    dem = res["demand"].sum(1)
    fill = 1 - res["lost"].sum(1) / np.maximum(dem, 1)
    per = dict(fill_rate=fill, stockout_days=res["stockout_days"].sum(1), avg_inv_value=res["avg_inv_value"].sum(1),
               orders=res["n_orders"].sum(1), holding=res["holding_cost"].sum(1), ordering=res["order_cost"].sum(1),
               shortage=res["shortage_cost"].sum(1), total_cost=res["total_cost"].sum(1))
    out = {}
    for k, v in per.items():
        out[k] = float(v.mean())
        out[k + "_p5"] = float(np.percentile(v, 5))
        out[k + "_p95"] = float(np.percentile(v, 95))
    return out


def run_all(policies, true_demand, econ, p, n_scen=1000, seed=0, warmup=21, demand_sigma=0.20):
    """Run every policy on identical scenarios. Returns (summary_dict, raw_results_dict)."""
    names = list(policies)
    L = p.lead_time_days
    Te = policies[names[0]]["mu"].shape[0]
    scen = make_scenarios(true_demand[:Te], n_scen, p, seed=seed, demand_sigma=demand_sigma)
    a = policies[names[0]]
    init = a["mu"][0] + a["ss"][0] + a["Q"]
    raw, summ = {}, {}
    for n in names:
        raw[n] = simulate(scen, policies[n]["mu"], policies[n]["ss"], policies[n]["Q"], econ, p, init, warmup)
        summ[n] = summarize(raw[n])
    return summ, raw
