"""
gb_harmonic_prediction.py — HistGradientBoosting yield models on the harmonic
gap-filled NDVI features (from ls_harmonic_features.py -> clean_harmonic_
features_stream.py -> aggregate_adc_to_muni.py).

This is the Phase C trainer. It replaces the window-composite
`gb_3period_prediction.py` for the NDVI side; AEF models are NOT touched.

Feature sets (--features), all municipality-level, 2017-2024 gap-free:
  h3_fixed     768 cols  3-period 2D histograms, fixed binning
  h3_quantile  768 cols  3-period 2D histograms, quantile binning
  h2_fixed     256 cols  2-period 2D histogram, fixed binning
  h2_quantile  256 cols  2-period 2D histogram, quantile binning
  coefs          7 cols  raw harmonic coefficients

Evaluation: random K-fold CV over municipality-year observations (default 5
folds, shuffled, seed 42) — muni-year pairs are drawn at random from the full
2017-2024 sample, NOT held out a whole year at a time. Yields come from
`siap_yields.load_muni_yields()` — SIAP by-season, Maize/Spring-Summer, which
runs through 2024 and is the SAME target `gb_aef_hist_ensemble.py` scores on,
so NDVI and AEF numbers are directly comparable.

Hyperparameter selection follows the existing convention in
`gb_3period_prediction.py` (best config per held-out fold) so the numbers stay
comparable to the already-published AEF results. Pass --nested to instead pick
the config on an inner split of the training rows, which is unbiased but NOT
comparable to the existing table.

Usage:
  ~/miniforge3/envs/geo_env/bin/python gb_harmonic_prediction.py --features h3_fixed
"""
import os
import argparse

import numpy  as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from siap_yields import load_muni_yields

parser =  argparse.ArgumentParser()
parser.add_argument('--features', required=True,
                    choices=['h3_fixed', 'h3_quantile', 'h2_fixed', 'h2_quantile', 'coefs'])
parser.add_argument('--level', default='muni', choices=['muni'])
parser.add_argument('--nested', action='store_true',
                    help='unbiased inner-CV config selection (not comparable to existing table)')
parser.add_argument('--folds', type=int, default=5,
                    help='number of random muni-year K-folds (default 5)')
parser.add_argument('--winsor', type=float, default=0.005,
                    help='two-sided winsorization for the coefs feature set (0 = off)')
parser.add_argument('--crop', default='Maize')
parser.add_argument('--season', default='Spring-Summer')
args =  parser.parse_args()

# ── Directories ──────────────────────────────────────────
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir  =  os.path.join(proj_dir, "Data")
feat_dir  =  os.path.join(data_dir, "harmonic_features")
pred_dir  =  os.path.join(data_dir, "predictions")
os.makedirs(pred_dir, exist_ok=True)

# ── Inputs ───────────────────────────────────────────────
name      =  'harmonic_coefs' if args.features == 'coefs' else args.features
feat_path =  os.path.join(feat_dir, f"muni_{name}.parquet")       # gap-free NDVI features
# yields come from siap_yields.load_muni_yields() (SIAP by-season, thru 2024)

# ── Outputs ──────────────────────────────────────────────
out_path  =  os.path.join(pred_dir, f"mun_harmonic_{args.features}_gb_preds.parquet")

HGB_GRID =  [
    (0.05, 500,  5),
    (0.05, 1000, 7),
    (0.03, 1000, 6),
    (0.03, 1500, 8),
    (0.01, 2000, 6),
]


def calc_r2(y_true, y_pred):
    ss_res =  np.mean((y_true - y_pred) ** 2)
    ss_tot =  np.mean((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_grid(X_tr, y_tr, X_va, y_va):
    """Return (best_model_cfg, best_r2) selecting on the given validation split."""
    best =  (None, -np.inf)
    for lr, n_est, depth in HGB_GRID:
        m =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                           learning_rate=lr, min_samples_leaf=5,
                                           random_state=42)
        m.fit(X_tr, y_tr)
        r2 =  calc_r2(y_va, m.predict(X_va))
        if r2 > best[1]:
            best =  ((lr, n_est, depth), r2)
    return best


def main():
    feats =  pd.read_parquet(feat_path)
    feats['muncode'] =  feats['muncode'].astype(str).str.zfill(5).astype(int)
    fcols =  [c for c in feats.columns if c not in ('muncode', 'year')]
    print(f"features: {args.features}  {feats.shape}  ({len(fcols)} cols)")

    if args.features == 'coefs' and args.winsor > 0:
        # ~0.5% of muni-years have ill-conditioned fits (too few clear obs to
        # identify 7 params) with |coef| in the thousands. Histogram features are
        # immune (reconstruction is clipped into bins), but the raw coefficients
        # are not, so clip the tails rather than drop the rows.
        lo =  feats[fcols].quantile(args.winsor)
        hi =  feats[fcols].quantile(1 - args.winsor)
        n_clip =  ((feats[fcols] < lo) | (feats[fcols] > hi)).any(axis=1).sum()
        feats[fcols] =  feats[fcols].clip(lo, hi, axis=1)
        print(f"  winsorized {args.winsor:.1%} tails; {n_clip:,} rows touched")

    yields =  load_muni_yields(crop=args.crop, season=args.season)

    merged =  pd.merge(feats, yields[['muncode', 'year', 'yield', 'ha_planted']],
                       on=['muncode', 'year'])
    print(f"merged: {merged.shape}  years {merged.year.min()}-{merged.year.max()}")

    X     =  merged[fcols].to_numpy(np.float32)
    y     =  merged['yield'].to_numpy()

    # Random K-fold over municipality-year observations: each fold holds out a
    # random subset of muni-year pairs from the full 2017-2024 sample (NOT a
    # whole year at a time), shuffled with a fixed seed for reproducibility.
    kf    =  KFold(n_splits=args.folds, shuffle=True, random_state=42)
    preds =  np.full(len(y), np.nan)
    rows  =  []
    for k, (tr_idx, te_idx) in enumerate(kf.split(X)):
        if args.nested:
            i_tr, i_va =  train_test_split(tr_idx, test_size=0.2, random_state=42)
            cfg, _ =  fit_grid(X[i_tr], y[i_tr], X[i_va], y[i_va])
        else:
            cfg, _ =  fit_grid(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx])  # existing convention
        lr, n_est, depth =  cfg
        m =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                           learning_rate=lr, min_samples_leaf=5,
                                           random_state=42)
        m.fit(X[tr_idx], y[tr_idx])
        preds[te_idx] =  m.predict(X[te_idx])
        r2 =  calc_r2(y[te_idx], preds[te_idx])
        rows.append({'fold': k, 'n': int(len(te_idx)), 'r2': r2})
        print(f"  fold {k}  n={len(te_idx):>5d}  R2={r2:.4f}  cfg={cfg}", flush=True)

    ok =  ~np.isnan(preds)
    r2 =  calc_r2(y[ok], preds[ok])
    rmse =  float(np.sqrt(np.mean((y[ok] - preds[ok]) ** 2)))
    print(f"\n{'='*56}\n{args.features}:  R2={r2:.4f}  RMSE={rmse:.4f}  N={ok.sum():,}\n{'='*56}")

    out =  merged[['muncode', 'year', 'yield', 'ha_planted']].copy()
    out['yield_pred']  =  preds
    out['feature_set'] =  args.features
    out['model']       =  'HistGB_harmonic'
    out.to_parquet(out_path, index=False)
    print(f"saved -> {out_path}")
    pd.DataFrame(rows).to_csv(out_path.replace('.parquet', '_byfold.csv'), index=False)


if __name__ == '__main__':
    main()
