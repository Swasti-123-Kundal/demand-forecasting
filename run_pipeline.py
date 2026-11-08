"""SupplyIQ end-to-end pipeline.

    python run_pipeline.py --source synthetic            # quick test, no downloads
    python run_pipeline.py --source m5 --m5-dir data     # real M5 data (put the 3 CSVs in data/)
"""
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supplyiq.config import PlantParams, MIN_LAG
from supplyiq.data import load_data
from supplyiq.features import build_features, FEATURES
from supplyiq.models import train_lgbm_suite, predict_suite, ets_predictions
from supplyiq.metrics import score_models, interval_scores, wape, bias
from supplyiq.inventory import make_econ, build_policies, POLICIES
from supplyiq.simulate import run_all
from supplyiq.explain import shap_analysis


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "m5"])
    ap.add_argument("--m5-dir", default="data")
    ap.add_argument("--n-skus", type=int, default=40)
    ap.add_argument("--test-days", type=int, default=120)
    ap.add_argument("--calib-days", type=int, default=56)
    ap.add_argument("--cv-folds", type=int, default=4)
    ap.add_argument("--fold-days", type=int, default=28)
    ap.add_argument("--n-scen", type=int, default=1000)
    ap.add_argument("--no-ets", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    P = PlantParams()

    # ------------------------------------------------------------ data + features
    print("[1/7] loading data + features")
    df, meta = load_data(a.source, a.m5_dir, a.n_skus, a.seed)
    feat, wide = build_features(df)
    dates = wide.index
    test_start = dates[-a.test_days]
    calib_start = test_start - pd.Timedelta(days=a.calib_days)
    test_dates = dates[dates >= test_start]
    bad = wide.loc[calib_start:].columns[wide.loc[calib_start:].isna().any()]
    if len(bad):
        print(f"   dropping {len(bad)} SKUs with gaps in calib/test window")
        feat = feat[~feat.sku.isin(bad)]; wide = wide.drop(columns=bad)
    skus = list(wide.columns)
    print(f"   {meta['source']} | {len(skus)} SKUs | {dates[0].date()} -> {dates[-1].date()} | test from {test_start.date()}")

    feat["naive"] = feat["lag_7"]; feat["ma7"] = feat["roll_mean_7"]; feat["ma28"] = feat["roll_mean_28"]
    base_cols = ["naive", "ma7", "ma28"]

    # ---------------------------------------------------------- ETS (rolling origin)
    cv_start = test_start - pd.Timedelta(days=a.cv_folds * a.fold_days)
    ets = None
    if not a.no_ets:
        print("[2/7] ETS rolling-origin baseline (parallel) ...")
        ets = ets_predictions(wide, cv_start, dates[-1])
    else:
        print("[2/7] ETS skipped")

    def attach_ets(d):
        return d.merge(ets, on=["date", "sku"], how="left") if ets is not None else d

    # ------------------------------------------------------- time-series CV (expanding)
    print("[3/7] time-series cross-validation")
    cv_rows, cv_int = [], []
    for i in range(a.cv_folds):
        vs = test_start - pd.Timedelta(days=a.fold_days * (a.cv_folds - i))
        ve = vs + pd.Timedelta(days=a.fold_days)
        tr = feat[(feat.date < vs) & feat.train_ok]
        va = feat[(feat.date >= vs) & (feat.date < ve)].copy()
        models = train_lgbm_suite(tr, FEATURES)
        va = pd.concat([va, predict_suite(models, va[FEATURES])], axis=1)
        va = attach_ets(va)
        cols = base_cols + (["ets"] if ets is not None else []) + ["ml_point"]
        sc = score_models(va, cols); sc["fold"] = i + 1; sc["val_start"] = vs.date()
        cv_rows.append(sc)
        cv_int.append(dict(fold=i + 1, **interval_scores(va.assign(p50=va["p50"]))))
    cv = pd.concat(cv_rows, ignore_index=True)
    cv_sum = cv.groupby("model").agg(WAPE=("WAPE", "mean"), WAPE_std=("WAPE", "std"),
                                     Bias=("Bias", "mean"), RMSE=("RMSE", "mean")).reset_index()
    cv_sum = cv_sum.sort_values("WAPE").reset_index(drop=True)
    cv_int = pd.DataFrame(cv_int)
    print(cv_sum.to_string(index=False))

    # -------------------------------------- stage 1: calibration of forecast uncertainty
    print("[4/7] calibrating forecast uncertainty + final model")
    tr1 = feat[(feat.date < calib_start) & feat.train_ok]
    cal = feat[(feat.date >= calib_start) & (feat.date < test_start)].copy()
    m1 = train_lgbm_suite(tr1, FEATURES)
    cal = pd.concat([cal, predict_suite(m1, cal[FEATURES])], axis=1)
    cal["resid"] = cal["sales"] - cal["ml_point"]
    cal["sigq_raw"] = (cal["p90"] - cal["p10"]) / 2.5631
    g = cal.groupby("sku")
    sig_resid = np.sqrt(g["resid"].apply(lambda r: np.mean(r ** 2))).reindex(skus)
    c_k = np.sqrt(sig_resid ** 2 / np.maximum(g["sigq_raw"].apply(lambda r: np.mean(r ** 2)).reindex(skus), 1e-9))
    c_k = c_k.clip(0.3, 5.0)

    # ------------------------------------------------------ stage 2: final model, holdout
    tr2 = feat[(feat.date < test_start) & feat.train_ok]
    te = feat[feat.date >= test_start].copy()
    m2 = train_lgbm_suite(tr2, FEATURES)
    te = pd.concat([te, predict_suite(m2, te[FEATURES])], axis=1)
    te = attach_ets(te)
    cols = base_cols + (["ets"] if ets is not None else []) + ["ml_point"]
    hold = score_models(te, cols).sort_values("WAPE").reset_index(drop=True)
    hold_int = interval_scores(te)
    print("holdout:\n", hold.to_string(index=False)); print(hold_int)

    # ------------------------------------------------------------- per-SKU analysis
    zero_frac = tr2.groupby("sku")["sales"].apply(lambda s: (s == 0).mean()).reindex(skus)
    rows = []
    for sku, d in te.groupby("sku"):
        r = dict(sku=sku, zero_frac=zero_frac[sku], mean_daily=d.sales.mean())
        for m in cols:
            r[f"wape_{m}"] = wape(d.sales, d[m])
        rows.append(r)
    per_sku = pd.DataFrame(rows)
    per_sku["ml_beats_ma28"] = per_sku["wape_ml_point"] < per_sku["wape_ma28"]
    per_sku["intermittent"] = per_sku["zero_frac"] > 0.5
    grp = per_sku.groupby("intermittent").agg(n=("sku", "count"), ml_wins=("ml_beats_ma28", "sum"),
                                              wape_ma28=("wape_ma28", "mean"), wape_ml=("wape_ml_point", "mean")).reset_index()
    print(grp.to_string(index=False))

    # --------------------------------------------------------------------- SHAP
    print("[5/7] SHAP")
    imp = shap_analysis(m2["point"], te[te.train_ok][FEATURES], out / "fig_shap_summary.png", out / "shap_importance.csv")
    print(imp.head(8).round(4).to_string())

    # --------------------------------------------- forecast -> inventory simulation
    print("[6/7] Monte Carlo inventory simulation")
    pv = lambda c: te.pivot(index="date", columns="sku", values=c).loc[test_dates, skus].values
    pred, p10, p90 = pv("ml_point"), pv("p10"), pv("p90")
    demand = wide.loc[test_dates, skus].values.astype(float)
    pad = 28
    hist = wide.loc[:, skus].iloc[len(dates) - a.test_days - pad:].values.astype(float)
    sigq_raw = (p90 - p10) / 2.5631
    sig_q = np.maximum(sigq_raw * c_k.values[None, :], 0.3 * sig_resid.values[None, :])
    econ = make_econ(feat[feat.date < test_start], skus)

    pol = build_policies(hist, pred, sig_q, sig_resid.values, econ, P, pad=pad)
    summ, raw = run_all(pol, demand, econ, P, n_scen=a.n_scen, seed=a.seed)
    sim = pd.DataFrame(summ).T.reset_index().rename(columns={"index": "policy"})
    print(sim[["policy", "fill_rate", "stockout_days", "avg_inv_value", "total_cost"]].to_string(index=False))

    levels = [0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995]
    sweep = []
    for sl in levels:
        pl = build_policies(hist, pred, sig_q, sig_resid.values, econ, P, service_level=sl, pad=pad)
        s_, _ = run_all(pl, demand, econ, P, n_scen=300, seed=a.seed + 1)
        for n, v in s_.items():
            sweep.append(dict(service_level=sl, policy=n, fill_rate=v["fill_rate"], total_cost=v["total_cost"],
                              avg_inv_value=v["avg_inv_value"], stockout_days=v["stockout_days"]))
    sweep = pd.DataFrame(sweep)

    # ---------------------------------------------------------------------- figures
    print("[7/7] figures + report")
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    ps = per_sku.sort_values("mean_daily")
    sel = [ps.iloc[-1]["sku"], ps.iloc[len(ps) // 2]["sku"], per_sku.sort_values("zero_frac").iloc[-1]["sku"]]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for axx, s in zip(ax, sel):
        d = te[te.sku == s].sort_values("date")
        axx.fill_between(d.date, d.p10, d.p90, alpha=0.25, label="ML P10-P90")
        axx.plot(d.date, d.sales, "k.-", lw=0.6, ms=3, label="actual")
        axx.plot(d.date, d.ml_point, lw=1.4, label="ML point")
        axx.plot(d.date, d.ma28, "--", lw=1.1, label="MA-28 baseline")
        axx.set_title(f"{s}  (zero-days {zero_frac[s]:.0%})", fontsize=9)
    ax[0].legend(ncol=4, fontsize=8); plt.tight_layout(); plt.savefig(out / "fig_forecast_examples.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(cv_sum.model, cv_sum.WAPE, yerr=cv_sum.WAPE_std.fillna(0), capsize=3, color="#4c78a8")
    ax.set_ylabel("WAPE (lower better)"); ax.set_title(f"Time-series CV ({a.cv_folds} expanding folds)")
    plt.tight_layout(); plt.savefig(out / "fig_cv_wape.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    short = ["A", "A+", "B1", "B2"]
    for axx, (k, t) in zip(ax, [("fill_rate", "Fill rate"), ("stockout_days", "Stockout days"),
                                ("avg_inv_value", "Avg inventory value"), ("total_cost", "Total cost")]):
        axx.bar(short, sim[k], color=["#bbb", "#999", "#4c78a8", "#2f5d8a"]); axx.set_title(t)
        if k == "fill_rate": axx.set_ylim(max(0, sim[k].min() - 0.05), 1)
    plt.tight_layout(); plt.savefig(out / "fig_policy_comparison.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in POLICIES:
        d = sweep[sweep.policy == n]
        ax[0].plot(d.fill_rate, d.total_cost, "o-", label=n); ax[1].plot(d.service_level, d.fill_rate, "o-", label=n)
    ax[0].set_xlabel("achieved fill rate"); ax[0].set_ylabel("total cost"); ax[0].set_title("Cost vs reliability trade-off")
    ax[1].set_xlabel("target service level (z)"); ax[1].set_ylabel("achieved fill rate"); ax[1].set_title("Target vs achieved")
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(out / "fig_tradeoff.png", dpi=140); plt.close()

    k0 = skus.index(sel[1]); Te = pol[POLICIES[0]]["mu"].shape[0]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for n, c in [(POLICIES[0], "#999"), (POLICIES[2], "#4c78a8")]:
        ax.plot(test_dates[:Te], raw[n]["path0"][:, k0], label=n, color=c)
        ax.plot(test_dates[:Te], (pol[n]["mu"] + pol[n]["ss"])[:, k0], ":", color=c, label=f"ROP - {n.split(':')[0]}")
    ax.set_title(f"Inventory 'tank level' - {sel[1]} (one scenario). Dotted = reorder point (low-level alarm)")
    ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(out / "fig_inventory_path.png", dpi=140); plt.close()

    # ---------------------------------------------------------------------- outputs
    cv.to_csv(out / "cv_metrics_by_fold.csv", index=False); cv_sum.to_csv(out / "cv_metrics_summary.csv", index=False)
    hold.to_csv(out / "holdout_metrics.csv", index=False); per_sku.to_csv(out / "per_sku_metrics.csv", index=False)
    sim.to_csv(out / "simulation_summary.csv", index=False); sweep.to_csv(out / "tradeoff_sweep.csv", index=False)
    te[["sku", "date", "sales", "naive", "ma7", "ma28"] + (["ets"] if ets is not None else []) +
       ["ml_point", "p10", "p50", "p90"]].to_csv(out / "holdout_predictions.csv.gz", index=False)

    h = hold.set_index("model")
    A, A2, B1, B2 = [sim.set_index("policy").loc[n] for n in POLICIES]
    pct = lambda new, old: 100 * (new - old) / old
    summary = dict(
        data=meta, n_skus=len(skus), test_days=a.test_days, plant_params_ASSUMED=P.to_dict(),
        holdout_wape=h["WAPE"].round(4).to_dict(), holdout_bias=h["Bias"].round(4).to_dict(),
        wape_reduction_vs_ma28_pct=round(100 * (h.loc["ma28", "WAPE"] - h.loc["ml_point", "WAPE"]) / h.loc["ma28", "WAPE"], 1),
        interval=hold_int, skus_ml_beats_ma28=f"{int(per_sku.ml_beats_ma28.sum())}/{len(per_sku)}",
        sim={n: {k: round(float(v), 4) for k, v in sim.set_index('policy').loc[n].items() if not k.endswith(("_p5", "_p95"))} for n in POLICIES},
        B1_vs_A=dict(stockout_days_pct=round(pct(B1.stockout_days, A.stockout_days), 1), cost_pct=round(pct(B1.total_cost, A.total_cost), 1),
                     inventory_pct=round(pct(B1.avg_inv_value, A.avg_inv_value), 1), fill_rate_pts=round(100 * (B1.fill_rate - A.fill_rate), 2)),
        B1_vs_Aplus=dict(stockout_days_pct=round(pct(B1.stockout_days, A2.stockout_days), 1), cost_pct=round(pct(B1.total_cost, A2.total_cost), 1),
                         inventory_pct=round(pct(B1.avg_inv_value, A2.avg_inv_value), 1), fill_rate_pts=round(100 * (B1.fill_rate - A2.fill_rate), 2)),
        runtime_sec=round(time.time() - t0, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    rep = [f"# SupplyIQ results\n\n**Data:** {meta['source']} - {len(skus)} SKUs, {a.test_days}-day untouched holdout.\n"]
    if meta["synthetic"]:
        rep.append("> WARNING: these numbers come from SYNTHETIC data. They validate the pipeline, they are NOT evidence about real demand. Re-run with `--source m5` before quoting results.\n")
    rep += ["\n## Forecast accuracy - time-series CV (expanding window, mean over folds)\n", md_table(cv_sum),
            "\n\n## Forecast accuracy - final holdout\n", md_table(hold),
            f"\n\nP10-P90 interval coverage on holdout: **{hold_int['coverage_p10_p90']:.1%}** (nominal 80%).\n",
            f"\nML beats MA-28 on **{summary['skus_ml_beats_ma28']}** SKUs. Split by intermittency (zero-days > 50%):\n\n", md_table(grp),
            "\n\n## Inventory policies - Monte Carlo (" + str(a.n_scen) + f" scenarios, target service level {P.service_level:.0%})\n",
            md_table(sim[["policy", "fill_rate", "stockout_days", "avg_inv_value", "orders", "holding", "ordering", "shortage", "total_cost"]], "{:,.3f}"),
            "\n\nA = moving average + fixed safety stock; A+ = moving average + z*sigma(demand)*sqrt(L); "
            "B1 = ML forecast + z*sigma(forecast error)*sqrt(L); B2 = ML forecast + quantile-spread uncertainty.\n",
            "\n\n## Assumed plant parameters (NOT real plant data)\n\n```\n" + json.dumps(P.to_dict(), indent=2) + "\n```\n",
            "\n![forecast](fig_forecast_examples.png)\n![cv](fig_cv_wape.png)\n![shap](fig_shap_summary.png)\n![policies](fig_policy_comparison.png)\n![tradeoff](fig_tradeoff.png)\n![path](fig_inventory_path.png)\n"]
    (out / "results.md").write_text("".join(rep))

    with open(out / "app_data.pkl", "wb") as f:
        pickle.dump(dict(meta=meta, skus=skus, test_dates=test_dates, hist=hist, pad=pad, demand=demand, pred=pred,
                         p10=p10, p50=pv("p50"), p90=p90, sig_q=sig_q, sig_resid=sig_resid.values, econ=econ, params=P,
                         cv_sum=cv_sum, hold=hold, per_sku=per_sku, sim=sim, sweep=sweep, shap_imp=imp, interval=hold_int,
                         ma28=pv("ma28")), f)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
