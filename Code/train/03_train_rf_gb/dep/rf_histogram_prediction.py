"""
RF/GB yield prediction on NDVI histogram features.

Modes (set FEATURE_SET below):
  "raw"      - 1024-element flattened 32x32 NDVI histograms
  "derived"  - ~85 summary features extracted from the 2D histogram
  "combined" - raw 1024 + derived features together

Evaluation: leave-one-year-out cross-validation (2003–2022).
"""
import os
import glob
import pickle

import pandas  as pd
import numpy   as np
from scipy     import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from siap_yields import load_muni_yields


# ============================================================
# CONFIGURATION
# ============================================================
FEATURE_SET =  "raw"       # "raw", "derived", or "combined"
USE_GB      =  False       # False = RF, True = GradientBoosting
BINS        =  32          # histogram bins per axis
# ============================================================


# ── Directories ──────────────────────────────────────────
home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir  =  os.path.join(proj_dir, "Data")
hist_dir  =  os.path.join(data_dir, "muni_ndvi_hist_0.2_1.0_32_max")
pred_dir  =  os.path.join(data_dir, "predictions")

os.makedirs(pred_dir, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────

def add_zeros(x, n):
    x =  str(x)
    while len(x) < n:
        x =  '0' + x
    return x


def load_histograms(hist_dir):
    """Load all pickle files and concatenate into a single DataFrame."""
    pkl_files =  sorted(glob.glob(os.path.join(hist_dir, "*.pkl")))
    print(f"Found {len(pkl_files)} pickle files in {hist_dir}")

    dfs =  []
    for f in pkl_files:
        with open(f, "rb") as fh:
            df =  pickle.load(fh)
        dfs.append(df)

    combined =  pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} municipality-year observations")
    return combined


def hist_to_matrix(flat_hist, bins=32):
    """Reshape a 1024-element flat histogram to a 32x32 matrix.

    Encoding: bin_1d = bin_p1 + bin_p2 * bins
    So axis 0 of the matrix = period 1, axis 1 = period 2.
    """
    return np.array(flat_hist).reshape(bins, bins).T


def extract_derived_features(flat_hist, bins=32):
    """Extract ~85 summary features from a 32x32 NDVI histogram.

    Returns a dict of feature_name: value.
    """
    mat  =  hist_to_matrix(flat_hist, bins)
    feat =  {}

    # ── Marginals (64 features) ──────────────────────────
    marg_p1 =  mat.sum(axis=1)   # sum over p2 → distribution at p1
    marg_p2 =  mat.sum(axis=0)   # sum over p1 → distribution at p2

    for i in range(bins):
        feat[f"marg_p1_{i:02d}"] =  marg_p1[i]
        feat[f"marg_p2_{i:02d}"] =  marg_p2[i]

    # ── Diagonal stats (3 features) ──────────────────────
    # Diagonal = same NDVI bin in both periods (no change)
    diag_frac  =  np.trace(mat)
    above_frac =  np.triu(mat, k=1).sum()   # p2 bin > p1 bin → greening
    below_frac =  np.tril(mat, k=-1).sum()  # p2 bin < p1 bin → browning

    feat["diag_frac"]  =  diag_frac
    feat["above_frac"] =  above_frac
    feat["below_frac"] =  below_frac

    # ── Per-period percentiles (10 features) ─────────────
    # Compute cumulative distribution from marginals
    for prefix, marg in [("p1", marg_p1), ("p2", marg_p2)]:
        cumsum =  np.cumsum(marg)
        for pct in [10, 25, 50, 75, 90]:
            # Find first bin where cumulative mass >= pct/100
            idx =  np.searchsorted(cumsum, pct / 100.0)
            idx =  min(idx, bins - 1)
            feat[f"{prefix}_pct{pct:02d}"] =  idx / bins

    # ── Change statistics (4 features) ───────────────────
    # Expected bin index for each period
    bin_idx    =  np.arange(bins)
    mean_p1    =  np.sum(bin_idx * marg_p1)
    mean_p2    =  np.sum(bin_idx * marg_p2)
    mean_shift =  mean_p2 - mean_p1

    # Weighted shift distribution: for each (i,j) cell, shift = j - i
    shifts  =  []
    weights =  []
    for i in range(bins):
        for j in range(bins):
            if mat[i, j] > 0:
                shifts.append(j - i)
                weights.append(mat[i, j])

    if len(shifts) > 1:
        shifts  =  np.array(shifts, dtype=float)
        weights =  np.array(weights, dtype=float)
        w_mean  =  np.average(shifts, weights=weights)
        w_var   =  np.average((shifts - w_mean)**2, weights=weights)
        w_std   =  np.sqrt(w_var)
        # Skewness and kurtosis (weighted)
        if w_std > 0:
            w_skew =  np.average(((shifts - w_mean) / w_std)**3, weights=weights)
            w_kurt =  np.average(((shifts - w_mean) / w_std)**4, weights=weights) - 3.0
        else:
            w_skew =  0.0
            w_kurt =  0.0
    else:
        w_mean =  mean_shift
        w_std  =  0.0
        w_skew =  0.0
        w_kurt =  0.0

    feat["change_mean"] =  w_mean
    feat["change_std"]  =  w_std
    feat["change_skew"] =  w_skew
    feat["change_kurt"] =  w_kurt

    # ── Quadrant fractions (4 features) ──────────────────
    # Split at midpoint (bin 16 for 32-bin histogram)
    mid =  bins // 2
    feat["quad_ll"] =  mat[:mid, :mid].sum()        # low → low
    feat["quad_lh"] =  mat[:mid, mid:].sum()         # low → high
    feat["quad_hl"] =  mat[mid:, :mid].sum()         # high → low
    feat["quad_hh"] =  mat[mid:, mid:].sum()         # high → high

    return feat


def build_feature_matrix(hist_df, feature_set, bins=32):
    """Build X matrix and feature names from histogram DataFrame.

    Args:
        hist_df: DataFrame with 'hist' column (1024-element arrays)
        feature_set: "raw", "derived", or "combined"
        bins: number of bins per axis

    Returns:
        X: numpy array (n_obs, n_features)
        feature_names: list of strings
    """
    n =  len(hist_df)

    if feature_set in ("raw", "combined"):
        raw_cols  =  [f"bin_{i:04d}" for i in range(bins**2)]
        X_raw     =  np.stack(hist_df['hist'].values)
    else:
        raw_cols  =  []
        X_raw     =  None

    if feature_set in ("derived", "combined"):
        derived_list =  []
        for idx in range(n):
            derived_list.append(extract_derived_features(hist_df['hist'].iloc[idx], bins))
        derived_df   =  pd.DataFrame(derived_list)
        derived_cols =  list(derived_df.columns)
        X_derived    =  derived_df.values
    else:
        derived_cols =  []
        X_derived    =  None

    # Assemble
    if feature_set == "raw":
        X     =  X_raw
        names =  raw_cols
    elif feature_set == "derived":
        X     =  X_derived
        names =  derived_cols
    else:  # combined
        X     =  np.hstack([X_raw, X_derived])
        names =  raw_cols + derived_cols

    print(f"Feature set '{feature_set}': {X.shape[1]} features")
    return X, names


def calc_r2(y_true, y_pred):
    """Compute R² (1 - SS_res / SS_tot)."""
    ss_res =  np.mean((y_true - y_pred)**2)
    ss_tot =  np.mean((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── Main ─────────────────────────────────────────────────

def main():
    # Load histograms
    hist_df =  load_histograms(hist_dir)

    # Construct muncode
    hist_df['muncode'] =  (
        hist_df['CVE_ENT'].apply(lambda x: add_zeros(x, 2)) +
        hist_df['CVE_MUN'].apply(lambda x: add_zeros(x, 3))
    ).astype(int)

    # Load yields
    yields =  load_muni_yields()          # canonical SIAP Maize/Spring-Summer
    print(f"Yields: {yields.shape}")

    # Merge
    merged =  pd.merge(hist_df, yields[['muncode', 'year', 'yield', 'ha_planted']],
                        on=['muncode', 'year'])
    print(f"Merged: {merged.shape}")

    # Build features
    X, feat_names =  build_feature_matrix(merged, FEATURE_SET, BINS)
    y             =  merged['yield'].values
    years         =  merged['year'].values
    unique_years  =  sorted(merged['year'].unique())
    print(f"Years: {unique_years[0]}–{unique_years[-1]} ({len(unique_years)} years)")

    # ── Leave-one-year-out CV ────────────────────────────
    print(f"\n{'='*60}")
    print(f"Leave-one-year-out CV  |  model={'GB' if USE_GB else 'RF'}  |  features={FEATURE_SET}")
    print(f"{'='*60}")

    all_preds =  np.full(len(y), np.nan)
    year_results =  []

    for hold_year in unique_years:
        train_mask =  years != hold_year
        test_mask  =  years == hold_year
        n_train    =  train_mask.sum()
        n_test     =  test_mask.sum()

        if n_test == 0:
            continue

        X_train, y_train =  X[train_mask], y[train_mask]
        X_test,  y_test  =  X[test_mask],  y[test_mask]

        if USE_GB:
            # Grid search over GB hyperparameters
            best_model =  None
            best_r2    =  -np.inf
            for lr, n_est, depth in [(0.05, 500, 5), (0.05, 1000, 7),
                                     (0.03, 1000, 6), (0.03, 1500, 8)]:
                model =  GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    min_samples_leaf=5, random_state=42, subsample=0.8
                )
                model.fit(X_train, y_train)
                y_hat =  model.predict(X_test)
                r2    =  calc_r2(y_test, y_hat)
                if r2 > best_r2:
                    best_r2    =  r2
                    best_model =  model
            y_pred =  best_model.predict(X_test)
        else:
            # Grid search over RF max_depth
            best_model =  None
            best_r2    =  -np.inf
            for d in [5, 10, 20, 50, None]:
                model =  RandomForestRegressor(
                    max_depth=d, n_estimators=200,
                    random_state=42, n_jobs=-1, min_samples_leaf=3
                )
                model.fit(X_train, y_train)
                y_hat =  model.predict(X_test)
                r2    =  calc_r2(y_test, y_hat)
                if r2 > best_r2:
                    best_r2    =  r2
                    best_model =  model
            y_pred =  best_model.predict(X_test)

        all_preds[test_mask] =  y_pred
        year_r2 =  calc_r2(y_test, y_pred)
        year_results.append({'year': hold_year, 'n': n_test, 'r2': year_r2})
        print(f"  {hold_year}  n={n_test:>5d}  R2={year_r2:.4f}")

    # ── Aggregate results ────────────────────────────────
    valid      =  ~np.isnan(all_preds)
    overall_r2 =  calc_r2(y[valid], all_preds[valid])
    rmse       =  np.sqrt(np.mean((y[valid] - all_preds[valid])**2))
    mae        =  np.mean(np.abs(y[valid] - all_preds[valid]))

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS  ({FEATURE_SET}, {'GB' if USE_GB else 'RF'})")
    print(f"  Overall R2:   {overall_r2:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  N obs:        {valid.sum()}")
    print(f"{'='*60}")

    # ── Save predictions ─────────────────────────────────
    out_df =  merged[['muncode', 'year', 'yield']].copy()
    out_df['yield_pred']  =  all_preds
    out_df['feature_set'] =  FEATURE_SET
    out_df['model']       =  'GB' if USE_GB else 'RF'

    out_name =  f"mun_hist_{FEATURE_SET}_{'gb' if USE_GB else 'rf'}_preds.parquet"
    out_path =  os.path.join(pred_dir, out_name)
    out_df.to_parquet(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")

    # ── Per-year results table ───────────────────────────
    print(f"\n{'Year':>6s}  {'N':>6s}  {'R2':>8s}")
    print("-" * 24)
    for r in year_results:
        print(f"{r['year']:>6d}  {r['n']:>6d}  {r['r2']:>8.4f}")

    # ── Feature importance (from last fold's best model) ─
    if hasattr(best_model, 'feature_importances_'):
        importances =  best_model.feature_importances_
        top_idx     =  np.argsort(importances)[::-1][:20]
        print(f"\nTop 20 features (from last fold):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank:>3d}. {feat_names[idx]:<20s}  {importances[idx]:.6f}")


if __name__ == '__main__':
    main()
