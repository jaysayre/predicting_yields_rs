"""
3_mun_cv_aef_hist_bins.py — municipality-level CV R2 for the AEF Hist *bins-only*
feature set (64 dims x 8 fixed-width bin shares = 512 features), computed under
BOTH the leave-one-year-out scheme and the random muni-year K-fold scheme, using
the SAME HGB grid and config-selection convention as mun_cv_aef_hist.py so the
bins and percentile muni-level numbers are produced by an identical protocol.

"AEF Hist" is the bins-only histogram model as of 2026-09; this script is the
muni-level CV analogue of the deployed bins model in 2_gb_aef_hist_ensemble.py.
Unlike that deployment it trains directly on the municipal histograms -- no
multinomial subsampling, since there is no ADC-level coarseness to mimic here.
Features: 512 bin shares (alpha_earth_mex_mun_binned_hist), each dimension's 8
counts normalised to sum to 1, NaN -> 0. Compare mun_cv_aef_hist.py, which runs
the same protocol on the older 448-feature percentile/stdDev/mean set.

Run: ~/miniforge3/envs/geo_env/bin/python 3_mun_cv_aef_hist_bins.py
"""
import os, sys
import numpy  as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from siap_yields import load_muni_yields

home_dir =  os.path.expanduser("~")
proj_dir =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir  =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir =  os.path.join(proj_dir, "Data", "predictions")
FOLDS    =  5

HGB_GRID =  [(0.05, 500, 5), (0.05, 1000, 7), (0.03, 1000, 6),
             (0.03, 1500, 8), (0.01, 2000, 6)]


def calc_r2(y, yh):
    ss_tot =  np.mean((y - y.mean()) ** 2)
    return 1 - np.mean((y - yh) ** 2) / ss_tot if ss_tot > 0 else 0.0


def fit_grid(X_tr, y_tr, X_va, y_va):
    best =  (None, -np.inf)
    for lr, n_est, depth in HGB_GRID:
        m =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                           learning_rate=lr, min_samples_leaf=5,
                                           random_state=42)
        m.fit(X_tr, y_tr)
        r2 =  calc_r2(y_va, m.predict(X_va))
        if r2 > best[1]:
            best =  ((lr, n_est, depth), r2)
    return best[0]


def build_features():
    """512 municipal bin shares: counts per dimension normalised to sum to 1."""
    bh =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet"))
    bcols =  sorted([c for c in bh.columns if c.startswith('A') and '_b' in c])

    n_dims, n_bins =  64, 8
    arr =  bh[bcols].to_numpy(np.float64).reshape(len(bh), n_dims, n_bins)
    arr =  np.clip(np.nan_to_num(arr, nan=0.0), 0, None)
    sums =  arr.sum(axis=2, keepdims=True)
    sums[sums < 1e-8] =  1.0
    arr =  arr / sums

    shares =  pd.DataFrame(arr.reshape(len(bh), n_dims * n_bins).astype(np.float32),
                           columns=bcols, index=bh.index)
    feats =  pd.concat([bh[['muncode', 'year']], shares], axis=1)
    return feats, bcols


def cv_predict_kfold(X, y):
    preds =  np.full(len(y), np.nan)
    splitter =  KFold(n_splits=FOLDS, shuffle=True, random_state=42).split(X)
    for tr_idx, te_idx in splitter:
        cfg =  fit_grid(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])
        lr, n_est, depth =  cfg
        m =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                           learning_rate=lr, min_samples_leaf=5,
                                           random_state=42)
        m.fit(X[tr_idx], y[tr_idx])
        preds[te_idx] =  m.predict(X[te_idx])
    return preds


def main():
    feats, fcols =  build_features()
    feats['muncode'] =  feats['muncode'].astype(str).str.zfill(5)
    yields =  load_muni_yields(crop='Maize', season='Spring-Summer')
    yields['muncode'] =  yields['muncode'].astype(str).str.zfill(5)
    yields =  yields[~yields['muncode'].str.endswith('000')]        # drop state aggregates

    merged =  feats.merge(yields[['muncode', 'year', 'yield']], on=['muncode', 'year'])
    print(f"AEF Hist bins muni-years: {len(merged):,}  features: {len(fcols)}")

    X =  merged[fcols].to_numpy(np.float32)
    y =  merged['yield'].to_numpy()
    years =  merged['year'].to_numpy()

    # LOYO
    preds_l =  np.full(len(y), np.nan)
    for h in sorted(set(years)):
        tr, te =  years != h, years == h
        cfg =  fit_grid(X[tr], y[tr], X[te], y[te])
        lr, n_est, depth =  cfg
        m =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                           learning_rate=lr, min_samples_leaf=5,
                                           random_state=42)
        m.fit(X[tr], y[tr]); preds_l[te] =  m.predict(X[te])
    r2_l =  calc_r2(y, preds_l)

    # random muni-year K-fold
    preds_k =  cv_predict_kfold(X, y)
    r2_k =  calc_r2(y, preds_k)

    print(f"\n{'='*56}")
    print(f"AEF Hist (bins) muni-level R2   LOYO = {r2_l:.4f}")
    print(f"AEF Hist (bins) muni-level R2   random muni-year {FOLDS}-fold = {r2_k:.4f}")
    print(f"{'='*56}")

    out_l =  merged[['muncode', 'year', 'yield']].copy()
    out_l['yield_pred'] =  preds_l
    out_l.to_parquet(os.path.join(pred_dir, "mun_aef_hist_bins_gb_loyo_preds.parquet"), index=False)
    print("saved -> mun_aef_hist_bins_gb_loyo_preds.parquet")

    out_k =  merged[['muncode', 'year', 'yield']].copy()
    out_k['yield_pred'] =  preds_k
    out_k.to_parquet(os.path.join(pred_dir, "mun_aef_hist_bins_gb_kfold_preds.parquet"), index=False)
    print("saved -> mun_aef_hist_bins_gb_kfold_preds.parquet")


if __name__ == '__main__':
    main()
