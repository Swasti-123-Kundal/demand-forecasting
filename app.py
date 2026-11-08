"""SupplyIQ dashboard.   Run:  streamlit run app.py   (after  python run_pipeline.py)"""
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from supplyiq.config import PlantParams, MIN_LAG
from supplyiq.inventory import build_policies, POLICIES, econ_arrays, z_from_service
from supplyiq.simulate import make_scenarios, simulate, summarize

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
SHORT = {POLICIES[0]: "A  MA + fixed SS", POLICIES[1]: "A+ MA + z·σ(demand)",
         POLICIES[2]: "B1 ML + z·σ(error)", POLICIES[3]: "B2 ML + quantile σ"}
COL = {POLICIES[0]: "#9aa0a6", POLICIES[1]: "#5f6368", POLICIES[2]: "#1a73e8", POLICIES[3]: "#0b3d91"}

st.set_page_config(page_title="SupplyIQ", page_icon="📦", layout="wide")


@st.cache_resource
def load():
    f = OUT / "app_data.pkl"
    if not f.exists():
        st.error("outputs/app_data.pkl not found. Run `python run_pipeline.py` first.")
        st.stop()
    return pickle.load(open(f, "rb"))


D = load()
meta, skus, dates = D["meta"], D["skus"], pd.DatetimeIndex(D["test_dates"])
cur = meta["currency"]

# --------------------------------------------------------------------------- sidebar
st.sidebar.title("📦 SupplyIQ")
sku = st.sidebar.selectbox("SKU", skus, index=len(skus) // 2)
sl = st.sidebar.slider("Target service level (cycle SL)", 0.80, 0.999, 0.95, 0.005, format="%.3f",
                       help="Like a design margin: higher = fewer stockouts, more buffer inventory.")
L = st.sidebar.slider(f"Lead time L (days) - max {MIN_LAG} (ML forecast info lag)", 1, MIN_LAG, 7)
n_scen = st.sidebar.select_slider("Monte Carlo scenarios", [100, 200, 500, 1000], value=200)
with st.sidebar.expander("Assumed plant parameters (editable)"):
    base = D["params"]
    lt_sd = st.slider("Lead-time disturbance sd (days)", 0.0, 3.0, base.lead_time_sd_days, 0.25)
    hold = st.slider("Holding cost (% unit cost / yr)", 5, 50, int(base.holding_rate_per_year * 100)) / 100
    short = st.slider("Shortage cost (× price per lost unit)", 0.1, 3.0, base.shortage_cost_mult, 0.1)
    ordc = st.slider("Order cost (% of 1 day's sales value)", 1, 30, int(base.order_cost_frac * 100)) / 100
    cover = st.slider("Policy A fixed SS (days of demand)", 0.5, 8.0, base.fixed_cover_days, 0.5)
P = replace(base, lead_time_days=L, lead_time_sd_days=lt_sd, holding_rate_per_year=hold, shortage_cost_mult=short,
            order_cost_frac=ordc, fixed_cover_days=cover, service_level=sl)

k = skus.index(sku)
sl_ = lambda a: a[:, [k]]


@st.cache_data(show_spinner="Running Monte Carlo ...")
def run_sku(k, sl, n_scen, pkey):
    p = PlantParams(**dict(pkey))
    pol = build_policies(D["hist"][:, [k]], D["pred"][:, [k]], D["sig_q"][:, [k]], D["sig_resid"][[k]],
                         D["econ"].iloc[[k]], p, service_level=sl, pad=D["pad"])
    Te = pol[POLICIES[0]]["mu"].shape[0]
    scen = make_scenarios(D["demand"][:Te, [k]], n_scen, p, seed=7)
    a = pol[POLICIES[0]]
    init = a["mu"][0] + a["ss"][0] + a["Q"]
    res = {n: simulate(scen, v["mu"], v["ss"], v["Q"], D["econ"].iloc[[k]], p, init) for n, v in pol.items()}
    return pol, {n: summarize(r) for n, r in res.items()}, {n: r["path0"][:, 0] for n, r in res.items()}


pkey = tuple(sorted(P.to_dict().items()))
pol, summ, paths = run_sku(k, sl, n_scen, pkey)
Te = pol[POLICIES[0]]["mu"].shape[0]
z = z_from_service(sl)

# ---------------------------------------------------------------------------- header
st.title("SupplyIQ — ML demand forecasting → inventory decisions")
if meta["synthetic"]:
    st.warning("Running on **synthetic** demo data — numbers validate the pipeline only, not real demand. "
               "Re-run `run_pipeline.py --source m5` for M5 results.")
else:
    st.caption(f"Data: {meta['source']}")
st.caption("Plant-side parameters (lead time, holding / order / shortage cost) are **assumed**, not real plant data.")

ps = D["per_sku"].set_index("sku").loc[sku]
t = len(dates) - Te - 1
c = st.columns(5)
c[0].metric("WAPE: ML", f"{ps['wape_ml_point']:.1%}", f"{(ps['wape_ml_point'] - ps['wape_ma28']) * 100:+.1f} pts vs MA-28",
            delta_color="inverse")
c[1].metric("z (service target)", f"{z:.2f}", f"{sl:.1%}")
for col, n in zip(c[2:], [POLICIES[0], POLICIES[2], POLICIES[3]]):
    rop = pol[n]["mu"][-1, 0] + pol[n]["ss"][-1, 0]
    col.metric(f"ROP now — {SHORT[n].split()[0]}", f"{rop:,.0f} units", f"SS {pol[n]['ss'][-1, 0]:,.0f} · EOQ {pol[n]['Q'][0]:,.0f}",
               delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast & policy", "⚖️ Cost vs reliability", "🧪 Model accuracy", "🔍 SHAP & assumptions"])

# ------------------------------------------------------------------------- tab 1
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=D["p90"][:, k], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=D["p10"][:, k], fill="tonexty", fillcolor="rgba(26,115,232,0.18)",
                             line=dict(width=0), name="ML P10–P90"))
    fig.add_trace(go.Scatter(x=dates, y=D["demand"][:, k], mode="lines+markers", name="actual", line=dict(color="black", width=1),
                             marker=dict(size=3)))
    fig.add_trace(go.Scatter(x=dates, y=D["pred"][:, k], name="ML forecast", line=dict(color="#1a73e8", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=D["ma28"][:, k], name="MA-28 baseline", line=dict(color="#9aa0a6", dash="dash")))
    fig.update_layout(title="Held-out period: actual demand vs forecasts", height=350, margin=dict(t=40, b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for n in [POLICIES[0], POLICIES[2]]:
        fig.add_trace(go.Scatter(x=dates[:Te], y=paths[n], name=f"stock — {SHORT[n]}", line=dict(color=COL[n], width=2)))
        fig.add_trace(go.Scatter(x=dates[:Te], y=pol[n]["mu"][:, 0] + pol[n]["ss"][:, 0], name=f"ROP — {SHORT[n].split()[0]}",
                                 line=dict(color=COL[n], dash="dot")))
    fig.update_layout(title="Inventory 'tank level' (one scenario) and reorder point = low-level alarm", height=350,
                      margin=dict(t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    rows = []
    for n, s in summ.items():
        rows.append({"Policy": SHORT[n], "Fill rate": f"{s['fill_rate']:.2%}", "Stockout days": f"{s['stockout_days']:.1f}",
                     f"Avg inventory ({cur})": f"{s['avg_inv_value']:,.0f}", "Orders": f"{s['orders']:.1f}",
                     f"Holding": f"{s['holding']:,.0f}", "Ordering": f"{s['ordering']:,.0f}", "Shortage": f"{s['shortage']:,.0f}",
                     f"Total cost ({cur})": f"{s['total_cost']:,.0f}"})
    st.subheader(f"Monte Carlo result ({n_scen} scenarios, this SKU)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Scenarios = perturbed replays of the held-out demand + random lead-time disturbance. All policies see identical scenarios. "
               "SS = z·σ·√L; ROP = forecast demand over L days + SS.")

# ------------------------------------------------------------------------- tab 2
with tab2:
    levels = [0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]

    @st.cache_data(show_spinner="Sweeping service levels ...")
    def sweep(k, pkey):
        rows = []
        for lv in levels:
            _, s_, _ = run_sku(k, lv, 200, pkey)
            for n, v in s_.items():
                rows.append(dict(level=lv, policy=n, fill=v["fill_rate"], cost=v["total_cost"], inv=v["avg_inv_value"]))
        return pd.DataFrame(rows)

    sw = sweep(k, pkey)
    c1, c2 = st.columns(2)
    f1, f2 = go.Figure(), go.Figure()
    for n in POLICIES:
        d = sw[sw.policy == n]
        f1.add_trace(go.Scatter(x=d.fill, y=d.cost, mode="lines+markers", name=SHORT[n], line=dict(color=COL[n]),
                                text=[f"target {x:.1%}" for x in d.level]))
        f2.add_trace(go.Scatter(x=d.fill, y=d.inv, mode="lines+markers", name=SHORT[n], line=dict(color=COL[n]),
                                text=[f"target {x:.1%}" for x in d.level]))
    f1.update_layout(title="Total cost vs achieved fill rate", xaxis_title="fill rate", yaxis_title=f"cost ({cur})", height=380)
    f2.update_layout(title="Average inventory vs achieved fill rate", xaxis_title="fill rate", yaxis_title=f"inventory value ({cur})", height=380)
    c1.plotly_chart(f1, width="stretch")
    c2.plotly_chart(f2, width="stretch")
    st.info("Reading the chart: a curve further **left-and-down** is better (same reliability, less cost / stock). "
            "Like design margin in a plant: buying reliability costs buffer capacity, with diminishing returns.")
    st.subheader("Whole portfolio (precomputed, all SKUs, 1000 scenarios)")
    s = D["sim"].copy()
    s["policy"] = s["policy"].map(SHORT)
    st.dataframe(s[["policy", "fill_rate", "stockout_days", "avg_inv_value", "orders", "holding", "ordering", "shortage", "total_cost"]]
                 .round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Time-series cross-validation (expanding window, no look-ahead)")
    st.dataframe(D["cv_sum"].round(4), hide_index=True, width="stretch")
    st.subheader("Final holdout")
    st.dataframe(D["hold"].round(4), hide_index=True, width="stretch")
    i = D["interval"]
    st.write(f"P10–P90 coverage on holdout: **{i['coverage_p10_p90']:.1%}** (nominal 80%).")
    st.subheader("Per-SKU: where does ML NOT win?")
    ps_all = D["per_sku"]
    st.write(f"ML beats MA-28 on **{int(ps_all.ml_beats_ma28.sum())} / {len(ps_all)}** SKUs. "
             f"Intermittent SKUs (>50 % zero-days) are the weak spot.")
    st.dataframe(ps_all.sort_values("zero_frac", ascending=False).round(3), hide_index=True, width="stretch")

# ------------------------------------------------------------------------- tab 4
with tab4:
    png = OUT / "fig_shap_summary.png"
    if png.exists():
        st.image(str(png), caption="SHAP summary — what drives demand")
    st.dataframe(D["shap_imp"].rename("mean |SHAP|").round(4).to_frame(), width="stretch")
    st.subheader("Assumed parameters")
    st.json(P.to_dict())
    st.markdown("""
| Supply-chain / ML concept | ChemE analogy |
|---|---|
| Inventory | Tank hold-up |
| Safety stock | Buffer-tank sizing against feed fluctuation |
| Lead time | Residence time / transport delay |
| Reorder point | Low-level alarm set with the delay in mind |
| Baseline vs ML model | First-principles vs data-driven model |
| Time-series CV | Train on old batches, validate on new batches |
| Service-level slider | Reliability vs cost (design margin) |
| Monte Carlo | Disturbance sensitivity analysis |
""")
