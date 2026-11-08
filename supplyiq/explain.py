import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def shap_analysis(model, X, out_png, out_csv=None, n=2500, seed=0):
    Xs = X.sample(min(n, len(X)), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    if out_csv:
        imp.rename("mean_abs_shap").to_csv(out_csv)
    keep = [c for c in Xs.columns if c != "sku_id"]          # SKU identity only encodes volume level
    plt.figure()
    shap.summary_plot(sv[:, [Xs.columns.get_loc(c) for c in keep]], Xs[keep], show=False, max_display=15)
    plt.title("SHAP: drivers of demand (LightGBM point model, log-link; sku_id omitted)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close("all")
    return imp
