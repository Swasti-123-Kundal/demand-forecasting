import numpy as np
import pandas as pd
import pytest
from supplyiq.config import PlantParams, MIN_LAG
from supplyiq.data import make_synthetic
from supplyiq.features import build_features, FEATURES
from supplyiq.inventory import eoq, safety_stock, reorder_point, build_policies, make_econ, POLICIES
from supplyiq.metrics import wape, bias
from supplyiq.simulate import make_scenarios, simulate


def test_formulas():
    assert eoq(1000, 50, 5) == np.ceil(np.sqrt(2 * 1000 * 50 / 5))
    assert safety_stock(1.645, 10, 4) == pytest.approx(1.645 * 10 * 2)
    assert reorder_point(20, 7, 30) == 170


def test_metrics():
    y = np.array([10, 10]); f = np.array([8, 14])
    assert wape(y, f) == pytest.approx(6 / 20)
    assert bias(y, f) == pytest.approx(2 / 20)


def test_features_have_no_leakage():
    """Changing sales at dates >= d-6 must NOT change the feature row of date d."""
    df, _ = make_synthetic(n_skus=3, days=300, seed=1)
    base, _ = build_features(df)
    sku = df.sku.iloc[0]
    d = pd.Timestamp("2021-09-01")
    df2 = df.copy()
    m = (df2.sku == sku) & (df2.date >= d - pd.Timedelta(days=MIN_LAG - 1))
    df2.loc[m, "sales"] += 999
    new, _ = build_features(df2)
    def row(x):
        r = x[(x.sku == sku) & (x.date == d)][FEATURES].copy()
        r["sku_id"] = r["sku_id"].cat.codes
        return r.astype(float).values
    a, b = row(base), row(new)
    assert a.shape == (1, len(FEATURES))
    np.testing.assert_allclose(a, b, equal_nan=True)


def _tiny(K=1, T=60, demand=10.0):
    p = PlantParams(lead_time_sd_days=0.0)
    econ = pd.DataFrame({"price": [100.0] * K, "mean_daily": [demand] * K})
    true = np.full((T, K), demand)
    scen = make_scenarios(true, 20, p, seed=0, demand_sigma=0.0)
    return p, econ, scen, true


def test_simulation_flows():
    p, econ, scen, true = _tiny()
    T = true.shape[0]
    big = simulate(scen, np.full((T, 1), 70.0), np.full((T, 1), 500.0), np.array([100.0]), econ, p, np.array([800.0]))
    assert big["lost"].sum() == 0                                # huge buffer -> no lost sales
    none = simulate(scen, np.zeros((T, 1)), np.zeros((T, 1)), np.array([5.0]), econ, p, np.array([0.0]))
    assert none["lost"].sum() > 0.5 * none["demand"].sum()         # no buffer -> mostly lost sales
    for r in (big, none):
        assert (r["lost"] <= r["demand"] + 1e-6).all() and (r["avg_onhand"] >= 0).all()
        assert r["total_cost"].min() >= 0


def test_long_lead_time_rejected():
    p = PlantParams(lead_time_days=MIN_LAG + 1)
    T, K = 40, 2
    with pytest.raises(ValueError):
        build_policies(np.ones((T + 28, K)), np.ones((T, K)), np.ones((T, K)), np.ones(K),
                       pd.DataFrame({"price": [1.0] * K, "mean_daily": [1.0] * K}), p)


def test_policies_shapes():
    p = PlantParams(); T, K = 50, 3
    econ = pd.DataFrame({"price": [50.0] * K, "mean_daily": [5.0] * K})
    pol = build_policies(np.random.rand(T + 28, K) * 10, np.random.rand(T, K) * 10,
                         np.random.rand(T, K) + 0.5, np.ones(K), econ, p)
    for n in POLICIES:
        assert pol[n]["mu"].shape == (T - p.lead_time_days, K) and (pol[n]["Q"] >= 1).all()
