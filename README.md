# SupplyIQ — ML Demand Forecasting & Inventory Optimisation (FMCG)

SKU-level demand forecasting with LightGBM (point + P10/P50/P90 quantiles), converted into
safety stock / reorder point / EOQ decisions, and stress-tested with a 1,000-scenario Monte Carlo
simulation against classical moving-average inventory policies. Interactive Streamlit dashboard included.

> **Honesty notes**
> * Plant-side parameters (lead time, holding / ordering / shortage cost) are **assumed**, not real plant data. They live in `supplyiq/config.py` and are editable in the dashboard.
> * `--source synthetic` is a pipeline test only. **Quote results only from a run on real data (`--source m5`).**
> * ML does not win everywhere: intermittent-demand SKUs are the weak spot (see per-SKU table in `outputs/results.md`).

## Pipeline

```
data (M5 / synthetic) -> leakage-safe features -> baselines + LightGBM -> time-series CV
        -> calibrate forecast error -> SS = z·σ·√L, ROP, EOQ -> Monte Carlo (4 policies) -> Streamlit
```

| Phase | What | Where |
|---|---|---|
| 1 Data | M5 subset (one store, FOODS, items sampled across volume incl. slow movers) or synthetic generator | `supplyiq/data.py` |
| 2 Forecast | naive (lag-7), MA-7, MA-28, rolling-origin ETS vs LightGBM (Tweedie point + quantile models); WAPE / bias / RMSE / interval coverage / pinball; expanding-window CV; SHAP | `features.py`, `models.py`, `metrics.py`, `explain.py` |
| 3 Inventory | SS = z·σ_error·√L, ROP = lead-time demand + SS, EOQ; Monte Carlo with random demand shocks and lead-time disturbance | `inventory.py`, `simulate.py` |
| 4 Dashboard | SKU selector, service-level slider, live ROP / SS / EOQ, cost-vs-reliability curve | `app.py` |

### Policies compared (same 1,000 scenarios for every policy)
* **A** – moving-average forecast + *fixed* safety stock (3 days of demand)
* **A+** – moving-average forecast + z·σ(demand)·√L (a stronger, textbook baseline)
* **B1** – ML forecast + z·σ(forecast error)·√L (σ from a calibration window)
* **B2** – ML forecast + safety stock from the learned P10–P90 spread (time-varying uncertainty)

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 6 tests incl. a no-leakage test

# Quick test (no download):
python run_pipeline.py --source synthetic

# Real data: download M5 from Kaggle ("M5 Forecasting - Accuracy") and put
# sales_train_evaluation.csv, calendar.csv, sell_prices.csv into ./data
python run_pipeline.py --source m5 --m5-dir data --n-skus 60

streamlit run app.py
```
Outputs (`outputs/`): `results.md` (all tables + figures), `summary.json`, CSVs, PNG figures, `app_data.pkl` (dashboard cache).
Add `--no-ets` to skip the slower ETS baseline.

## Design decisions (good interview material)
* **No leakage.** Every sales-derived feature for date *d* uses information ≤ *d − 7*; price / promo / festival / calendar are planned, known-in-advance covariates. `tests/test_core.py` proves it by perturbing recent sales and checking the feature row does not move.
* **Forecast horizon = 7 days = max lead time.** The forecast used in the ROP at time *t* must exist at time *t*, so `build_policies` refuses `L > 7`.
* **Validation:** expanding-window time-series CV (4 × 28 days) for model comparison, plus an untouched 120-day holdout. Forecast-error σ is calibrated on a separate window *before* the holdout, then the final model is refit.
* **Fair baselines.** ETS is refit every 7 days (information at least as fresh as the ML model's), and there are two MA policies (fixed SS and σ-based SS).
* **Tweedie objective** for the point forecast: demand is non-negative, with many zeros.
* **Target service level is a *cycle* service level** (z = Φ⁻¹(SL)); the *fill rate* is the measured outcome. They differ, and the dashboard shows both.
* **Monte Carlo scenarios** are perturbed replays of the held-out demand (correlated log-normal shocks) plus random lead times, with common random numbers across policies.

## Limitations
* One realisation of held-out demand (perturbed), not many independent futures; a rolling-origin policy backtest would be stronger.
* SS = z·σ·√L assumes independent daily errors; errors are autocorrelated in practice, so it can understate lead-time-demand variance.
* Plant economics are assumed. M5 is retail store demand, a proxy for FMCG distributor demand rather than plant demand. M5 "promo" is derived from price drops; "festival" = US calendar events.
* Intermittent SKUs (many zero-demand days) need dedicated methods (Croston / TSB); not implemented.

## ChemE analogy

| Supply chain / ML | ChemE |
|---|---|
| Inventory | Tank hold-up |
| Safety stock | Buffer-tank sizing against feed fluctuation |
| Lead time | Residence time / transport delay |
| Reorder point | Low-level alarm set with the delay in mind |
| Baseline vs ML | First-principles vs data-driven model |
| Time-series CV | Train on old batches, validate on new batches |
| Service-level slider | Reliability vs cost (design margin) |
| Monte Carlo | Disturbance sensitivity analysis |
