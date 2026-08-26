"""
gb_aef_hist_ensemble_africa.py
==============================
AEF Histogram Ensemble for HarvestStat Africa maize yields.

Replicates the Mexico ensemble approach that achieved the best within-R2:
  - Train at admin_2 (lowest available boundary) level
  - Two models: subsampled bins (HistGB) + percentile features (HistGB)
  - Ensemble: pred = w * bins + (1-w) * percentile  (w = 0.4)
  - Predictions will later be made at CIMMYT plot level (data TBD)

Evaluated with leave-one-year-out (LOYO) cross-validation at the
boundary level. Within-R2 is computed grouping by admin_1.

Inputs (in Data/HarvestStat_Africa/):
  hvstat_africa_maize_adm2.parquet        -- admin_2 yields (fnid x year)
  hvstat_africa_fnid_mapping.parquet      -- fnid -> admin1_id
  aef_africa_adm2_mean.parquet            -- 64 mean features
  aef_africa_adm2_hist.parquet            -- 384 percentile features
  aef_africa_adm2_binned_hist.parquet     -- 512 binned features

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 gb_aef_hist_ensemble_africa.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# Configuration  (matches Mexico best: N=2, K=5, w=0.4)
# ============================================================
N_PIX    =  2       # pixels per subsample draw
K_SAMP   =  5       # subsamples per region-year
W_BIN    =  0.4     # ensemble weight on bins model
MIN_YEAR =  2017    # first AEF year
# ============================================================


# -- Directories -----------------------------------------------
home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
hvstat_dir  =  os.path.join(proj_dir, "Data", "HarvestStat_Africa")
pred_dir    =  os.path.join(hvstat_dir, "predictions")
os.makedirs(pred_dir, exist_ok=True)


# -- Feature column names --------------------------------------
mean_cols  =  [f"A{d:02d}" for d in range(64)]

pct_cols   =  []
for d in range(64):
    for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']:
        pct_cols.append(f"A{d:02d}{s}")
pct_combined  =  pct_cols + mean_cols   # 448 features

bin_cols  =  [f"A{d:02d}_b{b}" for d in range(64) for b in range(8)]


# -- HistGB config (same as Mexico) ----------------------------
cfg  =  {
    'max_iter':         1500,
    'max_depth':        8,
    'learning_rate':    0.03,
    'min_samples_leaf': 5,
    'random_state':     42,
    'early_stopping':   False,
}


# ============================================================
# Helpers
# ============================================================
def r2(y, yh):
    m   =  np.isfinite(y) & np.isfinite(yh)
    y   =  np.array(y)[m]
    yh  =  np.array(yh)[m]
    if len(y) < 2:
        return np.nan
    ss_tot  =  np.sum((y - np.mean(y)) ** 2)
    return 1 - np.sum((y - yh) ** 2) / ss_tot if ss_tot > 0 else np.nan


def within_r2(df, ycol, pcol, gc='admin1_id'):
    sub  =  df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c    =  sub.groupby(gc).size()
    sub  =  sub[sub[gc].isin(c[c >= 2].index)]
    if len(sub) == 0:
        return np.nan
    gm  =  sub.groupby(gc)[[ycol, pcol]].transform('mean')
    return r2(sub[ycol] - gm[ycol], sub[pcol] - gm[pcol])


def between_r2(df, ycol, pcol, gc='admin1_id'):
    sub  =  df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g    =  sub.groupby(gc)[[ycol, pcol]].mean()
    return r2(g[ycol], g[pcol])


def eval_row(df, ycol, pcol, label, gc='admin1_id'):
    sub   =  df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    n     =  len(sub)
    ov    =  r2(sub[ycol], sub[pcol])
    b     =  between_r2(sub, ycol, pcol, gc)
    w     =  within_r2(sub, ycol, pcol, gc)
    rmse  =  np.sqrt(np.mean((sub[ycol].values - sub[pcol].values) ** 2))
    print(f"  {label:<40s} {n:>7,} {ov:>7.3f} {b:>7.3f} {w:>7.3f} {rmse:>7.3f}")
    return {'n': n, 'r2': ov, 'btw': b, 'wtn': w, 'rmse': rmse}


def subsample_bins(train_df, b_cols, K, N, seed=42):
    """Multinomial subsampling of region bin distributions.

    Creates K synthetic plot-like observations per region-year,
    mimicking the coarseness of field-level AEF features (fewer pixels).
    """
    rng      =  np.random.default_rng(seed)
    n_dims   =  64
    n_bins   =  8
    n_obs    =  len(train_df)

    # (n_obs, 64, 8) bin proportions
    arr  =  train_df[b_cols].values.reshape(n_obs, n_dims, n_bins).astype(np.float64)
    arr  =  np.clip(np.nan_to_num(arr, nan=0.0), 0, None)
    sums =  arr.sum(axis=2, keepdims=True)
    sums[sums < 1e-8]  =  1.0
    arr  =  arr / sums

    yields    =  train_df['yield'].values
    all_bin   =  []
    all_y     =  []

    for k in range(K):
        out  =  np.zeros((n_obs, n_dims, n_bins), dtype=np.float32)
        for d in range(n_dims):
            for i in range(n_obs):
                out[i, d, :]  =  rng.multinomial(N, arr[i, d, :]) / N
        all_bin.append(out.reshape(n_obs, n_dims * n_bins))
        all_y.append(yields)
        print(f"    subsample {k + 1}/{K}", flush=True)

    return np.vstack(all_bin), np.concatenate(all_y)


# ============================================================
# Main
# ============================================================
t0  =  time.time()


# -- 1. Load data -----------------------------------------------
print("Loading data...")

adm2_yields  =  pd.read_parquet(os.path.join(hvstat_dir, "hvstat_africa_maize_adm2.parquet"))
fnid_map     =  pd.read_parquet(os.path.join(hvstat_dir, "hvstat_africa_fnid_mapping.parquet"))

adm2_bh   =  pd.read_parquet(os.path.join(hvstat_dir, "aef_africa_adm2_binned_hist.parquet"))
adm2_pct  =  pd.read_parquet(os.path.join(hvstat_dir, "aef_africa_adm2_hist.parquet"))
adm2_mean =  pd.read_parquet(os.path.join(hvstat_dir, "aef_africa_adm2_mean.parquet"))

# Attach admin1_id to features
adm2_bh   =  adm2_bh.merge(fnid_map[['fnid', 'admin1_id']], on='fnid', how='inner')
adm2_pct  =  adm2_pct.merge(fnid_map[['fnid', 'admin1_id']], on='fnid', how='inner')
adm2_mean =  adm2_mean.merge(fnid_map[['fnid', 'admin1_id']], on='fnid', how='inner')

# Combine percentile + mean features
adm2_pct_full  =  adm2_pct.merge(
    adm2_mean[['fnid', 'year'] + mean_cols],
    on=['fnid', 'year'], how='inner'
)

# Full feature set per admin_2 region-year (bins + pct + mean)
adm2_combo  =  adm2_bh.merge(
    adm2_pct_full[['fnid', 'year', 'admin1_id'] + pct_combined],
    on=['fnid', 'year', 'admin1_id'], how='inner'
)

print(f"  Admin_2 AEF features:  {len(adm2_combo):,} region-years")
print(f"  Admin_2 yield obs:     {len(adm2_yields):,}")


# -- 2. Merge features with yields for training -----------------
print("\nBuilding training data at admin_2 level...")

train_bin  =  adm2_bh.merge(
    adm2_yields[['fnid', 'year', 'yield', 'admin1_id']],
    on=['fnid', 'year', 'admin1_id'], how='inner'
)
train_pct  =  adm2_pct_full.merge(
    adm2_yields[['fnid', 'year', 'yield', 'admin1_id']],
    on=['fnid', 'year', 'admin1_id'], how='inner'
)

print(f"  Training region-years: bin={len(train_bin):,}, pct={len(train_pct):,}")

years  =  sorted(set(train_bin['year'].unique()) & set(train_pct['year'].unique()))
print(f"  Years available: {years}")

if len(years) < 2:
    print("\nERROR: Need at least 2 years for LOYO CV.")
    sys.exit(1)


# -- 3. Leave-one-year-out cross-validation ---------------------
print(f"\n{'=' * 70}")
print("  LOYO Cross-Validation  (train & predict at admin_2 level)")
print(f"{'=' * 70}")

all_preds  =  []

for hold_year in years:
    print(f"\n  --- Holding out {hold_year} ---")
    tr_bin  =  train_bin[train_bin['year'] != hold_year]
    tr_pct  =  train_pct[train_pct['year'] != hold_year]

    if len(tr_bin) < 10 or len(tr_pct) < 10:
        print(f"    SKIP: too few training obs (bin={len(tr_bin)}, pct={len(tr_pct)})")
        continue

    # Train percentile model
    print(f"    Training percentile model ({len(tr_pct):,} obs)...")
    m_pct  =  HistGradientBoostingRegressor(**cfg)
    m_pct.fit(
        tr_pct[pct_combined].fillna(0).values.astype(np.float32),
        tr_pct['yield'].values
    )

    # Subsample bins & train bins model
    print(f"    Subsampling bins (N={N_PIX}, K={K_SAMP})...")
    aug_X, aug_y  =  subsample_bins(tr_bin, bin_cols, K=K_SAMP, N=N_PIX, seed=42)

    print(f"    Training bins model ({len(aug_y):,} augmented obs)...")
    m_bin  =  HistGradientBoostingRegressor(**cfg)
    m_bin.fit(aug_X.astype(np.float32), aug_y)

    # Predict held-out year at admin_2 level
    te_combo  =  adm2_combo[adm2_combo['year'] == hold_year].copy()
    if len(te_combo) == 0:
        print(f"    SKIP: no admin_2 features for {hold_year}")
        continue

    p_bin  =  m_bin.predict(
        te_combo[bin_cols].fillna(0).values.astype(np.float32)
    ).clip(0)
    p_pct  =  m_pct.predict(
        te_combo[pct_combined].fillna(0).values.astype(np.float32)
    ).clip(0)

    te_combo['pred']  =  W_BIN * p_bin + (1 - W_BIN) * p_pct
    all_preds.append(te_combo[['fnid', 'year', 'admin1_id', 'pred']].copy())
    print(f"    Predicted {len(te_combo):,} admin_2 regions")

# Combine all LOYO predictions
preds  =  pd.concat(all_preds, ignore_index=True)
print(f"\n  Total LOYO predictions: {len(preds):,}")


# -- 4. Merge predictions with admin_2 yields -------------------
eval_df  =  adm2_yields.merge(preds, on=['fnid', 'year', 'admin1_id'], how='inner')
print(f"  Matched with yields: {len(eval_df):,}")


# -- 5. Evaluate (within-R2 grouped by admin_1) -----------------
print(f"\n{'=' * 70}")
print(f"  {'Model':<40s} {'N':>7s} {'R2':>7s} {'Btw':>7s} "
      f"{'Wtn':>7s} {'RMSE':>7s}")
print(f"  {'-' * 65}")

print("\n  --- LOYO All Years (raw) ---")
eval_row(eval_df, 'yield', 'pred', 'AEF Hist Ens. Raw')

# Per-year breakdown
print("\n  --- LOYO By Year ---")
for yr in sorted(eval_df['year'].unique()):
    sub  =  eval_df[eval_df['year'] == yr]
    if len(sub) >= 10:
        eval_row(sub, 'yield', 'pred', f'  {yr}')


# -- 6. Retrain on all data & save predictions -------------------
print(f"\n{'=' * 70}")
print("  Retraining on all data...")

# Percentile model
m_pct_final  =  HistGradientBoostingRegressor(**cfg)
m_pct_final.fit(
    train_pct[pct_combined].fillna(0).values.astype(np.float32),
    train_pct['yield'].values
)

# Bins model
print("  Subsampling bins (full training set)...")
aug_X_all, aug_y_all  =  subsample_bins(
    train_bin, bin_cols, K=K_SAMP, N=N_PIX, seed=42
)
m_bin_final  =  HistGradientBoostingRegressor(**cfg)
m_bin_final.fit(aug_X_all.astype(np.float32), aug_y_all)

# Predict all admin_2 region-years
print("  Predicting all admin_2 region-years...")
p_bin_all  =  m_bin_final.predict(
    adm2_combo[bin_cols].fillna(0).values.astype(np.float32)
).clip(0)
p_pct_all  =  m_pct_final.predict(
    adm2_combo[pct_combined].fillna(0).values.astype(np.float32)
).clip(0)

adm2_combo['pred']  =  W_BIN * p_bin_all + (1 - W_BIN) * p_pct_all

# Save full predictions
out_cols  =  ['fnid', 'year', 'admin1_id', 'pred']
out_path  =  os.path.join(pred_dir, "adm2_aef_hist_ensemble_africa.parquet")
adm2_combo[out_cols].to_parquet(out_path, index=False)
print(f"\n  Saved: {out_path}")
print(f"    {len(adm2_combo):,} admin_2 predictions")

# Save LOYO predictions (for diagnostics)
loyo_path  =  os.path.join(pred_dir, "adm2_aef_hist_ensemble_africa_loyo.parquet")
eval_df.to_parquet(loyo_path, index=False)
print(f"  Saved: {loyo_path}")

print(f"\nRuntime: {(time.time() - t0) / 60:.1f} min")
