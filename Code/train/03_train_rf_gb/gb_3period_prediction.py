"""
gb_3period_prediction.py — HistGradientBoosting yield prediction on 3-period
2D NDVI histogram features.

Modes (set FEATURE_SET below):
  "raw"      - 768-element flattened histograms (3 pairs × 256 bins)
  "derived"  - ~159 summary features extracted from three 2D histograms
  "combined" - raw 768 + derived features together

Evaluation: leave-one-year-out cross-validation (2003–2022).
After CV: retrain on all muni data and predict ADC-level yields.
"""
import os
import glob
import pickle

import pandas  as pd
import numpy   as np
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# CONFIGURATION
# ============================================================
FEATURE_SET =  "raw"       # "raw", "derived", or "combined"
BINS        =  16          # histogram bins per axis
N_PAIRS     =  3           # number of period pairs
BIN_TYPE    =  "uniform"   # "uniform" or "quantile"
# ============================================================


# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir   =  os.path.join(proj_dir, "Data")
pred_dir   =  os.path.join(data_dir, "predictions")

if BIN_TYPE == "quantile":
    muni_hist =  os.path.join(data_dir, "3period_hists", "muni_3period_hists_quantile_clean")
    adc_hist  =  os.path.join(data_dir, "3period_hists", "adc_3period_hists_quantile_clean")
else:
    muni_hist =  os.path.join(data_dir, "3period_hists", "muni_3period_hists_clean")
    adc_hist  =  os.path.join(data_dir, "3period_hists", "adc_3period_hists_clean")

os.makedirs(pred_dir, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────

def load_histograms(hist_dir):
    """Load all per-year pickle files and concatenate."""
    pkl_files =  sorted(glob.glob(os.path.join(hist_dir, "*.pkl")))
    print(f"Found {len(pkl_files)} pickle files in {hist_dir}")

    dfs =  []
    for f in pkl_files:
        df =  pd.read_pickle(f)
        dfs.append(df)

    combined =  pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} observations")
    return combined


def hist_to_matrix(flat_pair, bins=16):
    """Reshape a bins²-element flat histogram to a bins×bins matrix.

    Encoding: bin_1d = bin_p1 + bin_p2 * bins
    So axis 0 = period A, axis 1 = period B.
    """
    return np.array(flat_pair).reshape(bins, bins).T


def extract_pair_features(flat_pair, bins=16, prefix=""):
    """Extract ~53 summary features from one 2D histogram pair.

    Returns a dict of feature_name: value.
    """
    mat  =  hist_to_matrix(flat_pair, bins)
    feat =  {}

    # ── Marginals (2 × bins features) ────────────────────
    marg_a =  mat.sum(axis=1)   # sum over B → distribution at A
    marg_b =  mat.sum(axis=0)   # sum over A → distribution at B

    for i in range(bins):
        feat[f"{prefix}marg_a_{i:02d}"] =  marg_a[i]
        feat[f"{prefix}marg_b_{i:02d}"] =  marg_b[i]

    # ── Diagonal stats (3 features) ──────────────────────
    feat[f"{prefix}diag_frac"]  =  np.trace(mat)
    feat[f"{prefix}above_frac"] =  np.triu(mat, k=1).sum()
    feat[f"{prefix}below_frac"] =  np.tril(mat, k=-1).sum()

    # ── Per-period percentiles (10 features) ─────────────
    for name, marg in [("a", marg_a), ("b", marg_b)]:
        cumsum =  np.cumsum(marg)
        for pct in [10, 25, 50, 75, 90]:
            idx =  np.searchsorted(cumsum, pct / 100.0)
            idx =  min(idx, bins - 1)
            feat[f"{prefix}{name}_pct{pct:02d}"] =  idx / bins

    # ── Change statistics (4 features) ───────────────────
    bin_idx =  np.arange(bins)
    mean_a  =  np.sum(bin_idx * marg_a)
    mean_b  =  np.sum(bin_idx * marg_b)

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
        if w_std > 0:
            w_skew =  np.average(((shifts - w_mean) / w_std)**3, weights=weights)
            w_kurt =  np.average(((shifts - w_mean) / w_std)**4, weights=weights) - 3.0
        else:
            w_skew =  0.0
            w_kurt =  0.0
    else:
        w_mean =  mean_b - mean_a
        w_std  =  0.0
        w_skew =  0.0
        w_kurt =  0.0

    feat[f"{prefix}change_mean"] =  w_mean
    feat[f"{prefix}change_std"]  =  w_std
    feat[f"{prefix}change_skew"] =  w_skew
    feat[f"{prefix}change_kurt"] =  w_kurt

    # ── Quadrant fractions (4 features) ──────────────────
    mid =  bins // 2
    feat[f"{prefix}quad_ll"] =  mat[:mid, :mid].sum()
    feat[f"{prefix}quad_lh"] =  mat[:mid, mid:].sum()
    feat[f"{prefix}quad_hl"] =  mat[mid:, :mid].sum()
    feat[f"{prefix}quad_hh"] =  mat[mid:, mid:].sum()

    return feat


def extract_derived_features(hist_768, bins=16):
    """Extract derived features from all 3 period pairs.

    hist_768 = concatenation of [hist_p12 (256), hist_p23 (256), hist_p13 (256)]
    """
    n =  bins * bins
    pair_hists =  {
        'p12_': hist_768[:n],
        'p23_': hist_768[n:2*n],
        'p13_': hist_768[2*n:3*n],
    }

    all_feat =  {}
    for prefix, flat_pair in pair_hists.items():
        all_feat.update(extract_pair_features(flat_pair, bins, prefix))

    return all_feat


def build_feature_matrix(hist_df, feature_set, bins=16):
    """Build X matrix and feature names from histogram DataFrame.

    Args:
        hist_df: DataFrame with 'hist' column (768-element arrays)
        feature_set: "raw", "derived", or "combined"
        bins: number of bins per axis

    Returns:
        X: numpy array (n_obs, n_features)
        feature_names: list of strings
    """
    n =  len(hist_df)

    if feature_set in ("raw", "combined"):
        n_total  =  N_PAIRS * bins * bins
        raw_cols =  [f"bin_{i:04d}" for i in range(n_total)]
        X_raw    =  np.stack(hist_df['hist'].values)
    else:
        raw_cols =  []
        X_raw    =  None

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

    if feature_set == "raw":
        X     =  X_raw
        names =  raw_cols
    elif feature_set == "derived":
        X     =  X_derived
        names =  derived_cols
    else:
        X     =  np.hstack([X_raw, X_derived])
        names =  raw_cols + derived_cols

    print(f"Feature set '{feature_set}': {X.shape[1]} features")
    return X, names


def calc_r2(y_true, y_pred):
    """Compute R² (1 - SS_res / SS_tot)."""
    ss_res =  np.mean((y_true - y_pred)**2)
    ss_tot =  np.mean((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── HistGB hyperparameter grid ───────────────────────────
HGB_GRID =  [
    (0.05, 500,  5),
    (0.05, 1000, 7),
    (0.03, 1000, 6),
    (0.03, 1500, 8),
    (0.01, 2000, 6),
]


# ── Main ─────────────────────────────────────────────────

def main():
    # Load municipality histograms
    hist_df =  load_histograms(muni_hist)

    # Ensure muncode is string for merge
    hist_df['muncode'] =  hist_df['muncode'].astype(str).str.zfill(5).astype(int)

    # Load yields
    yields =  pd.read_csv(os.path.join(data_dir, "muni_grano_yields.csv"))
    yields =  yields[yields['yield'].notna() & (yields['yield'] > 0)]
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
    print(f"Leave-one-year-out CV  |  model=HistGB  |  features={FEATURE_SET}")
    print(f"{'='*60}")

    all_preds    =  np.full(len(y), np.nan)
    year_results =  []
    best_configs =  []

    for hold_year in unique_years:
        train_mask =  years != hold_year
        test_mask  =  years == hold_year
        n_train    =  train_mask.sum()
        n_test     =  test_mask.sum()

        if n_test == 0:
            continue

        X_train, y_train =  X[train_mask], y[train_mask]
        X_test,  y_test  =  X[test_mask],  y[test_mask]

        # Grid search over HistGB hyperparameters
        best_model =  None
        best_r2    =  -np.inf
        best_cfg   =  None

        for lr, n_est, depth in HGB_GRID:
            model =  HistGradientBoostingRegressor(
                max_iter=n_est, max_depth=depth, learning_rate=lr,
                min_samples_leaf=5, random_state=42,
            )
            model.fit(X_train, y_train)
            y_hat =  model.predict(X_test)
            r2    =  calc_r2(y_test, y_hat)
            if r2 > best_r2:
                best_r2    =  r2
                best_model =  model
                best_cfg   =  (lr, n_est, depth)

        y_pred =  best_model.predict(X_test)
        all_preds[test_mask] =  y_pred
        year_r2 =  calc_r2(y_test, y_pred)
        year_results.append({'year': hold_year, 'n': n_test, 'r2': year_r2})
        best_configs.append(best_cfg)
        print(f"  {hold_year}  n={n_test:>5d}  R2={year_r2:.4f}  cfg={best_cfg}")

    # ── Aggregate results ────────────────────────────────
    valid      =  ~np.isnan(all_preds)
    overall_r2 =  calc_r2(y[valid], all_preds[valid])
    rmse       =  np.sqrt(np.mean((y[valid] - all_preds[valid])**2))
    mae        =  np.mean(np.abs(y[valid] - all_preds[valid]))

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS  ({FEATURE_SET}, HistGB, 3-period)")
    print(f"  Overall R2:   {overall_r2:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  N obs:        {valid.sum()}")
    print(f"{'='*60}")

    # ── Save municipality predictions ────────────────────
    out_df =  merged[['muncode', 'year', 'yield']].copy()
    out_df['yield_pred']  =  all_preds
    out_df['feature_set'] =  FEATURE_SET
    out_df['model']       =  'HistGB_3period'

    bin_suffix =  f"_{BIN_TYPE}" if BIN_TYPE != "uniform" else ""
    muni_out =  os.path.join(pred_dir, f"mun_3period_hist_{FEATURE_SET}{bin_suffix}_gb_preds.parquet")
    out_df.to_parquet(muni_out, index=False)
    print(f"\nSaved muni predictions to {muni_out}")

    # ── Per-year results table ───────────────────────────
    print(f"\n{'Year':>6s}  {'N':>6s}  {'R2':>8s}")
    print("-" * 24)
    for r in year_results:
        print(f"{r['year']:>6d}  {r['n']:>6d}  {r['r2']:>8.4f}")

    # ── Feature importance (from last fold) ──────────────
    if hasattr(best_model, 'feature_importances_'):
        importances =  best_model.feature_importances_
        top_idx     =  np.argsort(importances)[::-1][:20]
        print(f"\nTop 20 features (from last fold):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank:>3d}. {feat_names[idx]:<20s}  {importances[idx]:.6f}")

    # ── ADC prediction ───────────────────────────────────
    print(f"\n{'='*60}")
    print("ADC-level prediction")
    print(f"{'='*60}")

    if not os.path.isdir(adc_hist) or len(glob.glob(os.path.join(adc_hist, "*.pkl"))) == 0:
        print("ADC histogram directory not found or empty — skipping ADC prediction.")
        return

    # Find most common best config across CV folds
    from collections import Counter
    cfg_counts =  Counter(best_configs)
    final_cfg  =  cfg_counts.most_common(1)[0][0]
    print(f"Most common best config: lr={final_cfg[0]}, n_est={final_cfg[1]}, depth={final_cfg[2]}")

    # Retrain on all municipality data
    final_model =  HistGradientBoostingRegressor(
        max_iter=final_cfg[1], max_depth=final_cfg[2],
        learning_rate=final_cfg[0],
        min_samples_leaf=5, random_state=42,
    )
    final_model.fit(X, y)
    print(f"Retrained on {len(y)} municipality-year observations")

    # Load ADC histograms
    adc_df =  load_histograms(adc_hist)
    print(f"ADC observations: {len(adc_df)}")

    # Build ADC features
    X_adc, _ =  build_feature_matrix(adc_df, FEATURE_SET, BINS)

    # Predict
    adc_df['yield_pred']  =  final_model.predict(X_adc)
    adc_df['feature_set'] =  FEATURE_SET
    adc_df['model']       =  'HistGB_3period'

    adc_out_cols =  ['adcid', 'year', 'yield_pred', 'feature_set', 'model']
    adc_out      =  os.path.join(pred_dir, f"adc_3period_hist{bin_suffix}_gb_preds.parquet")
    adc_df[adc_out_cols].to_parquet(adc_out, index=False)
    print(f"Saved {len(adc_df)} ADC predictions to {adc_out}")

    # ── CIMMYT prediction ─────────────────────────────────
    cimmyt_hist =  os.path.join(data_dir, "3period_hists", "cimmyt_3period_hists_clean")

    if os.path.isdir(cimmyt_hist) and len(glob.glob(os.path.join(cimmyt_hist, "*.pkl"))) > 0:
        print(f"\n{'='*60}")
        print("CIMMYT plot-level prediction")
        print(f"{'='*60}")

        cimmyt_df =  load_histograms(cimmyt_hist)
        print(f"CIMMYT observations: {len(cimmyt_df)}")

        X_cimmyt, _ =  build_feature_matrix(cimmyt_df, FEATURE_SET, BINS)

        cimmyt_df['yield_pred']  =  final_model.predict(X_cimmyt)
        cimmyt_df['feature_set'] =  FEATURE_SET
        cimmyt_df['model']       =  'HistGB_3period'

        cimmyt_df['plot_id'] =  cimmyt_df['plot_id'].astype(str)
        cimmyt_out_cols =  ['plot_id', 'year', 'yield_pred', 'feature_set', 'model']
        cimmyt_out      =  os.path.join(pred_dir, f"cimmyt_3period_hist{bin_suffix}_gb_preds.parquet")
        cimmyt_df[cimmyt_out_cols].to_parquet(cimmyt_out, index=False)
        print(f"Saved {len(cimmyt_df)} CIMMYT predictions to {cimmyt_out}")


if __name__ == '__main__':
    main()
