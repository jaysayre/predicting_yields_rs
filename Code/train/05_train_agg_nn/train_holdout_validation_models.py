"""
train_holdout_validation_models.py — standardized inputs for Table 1
(validation_mun_level_2022.tex, tab:validation_mun).

Why this exists (2026-08-14): the published Table 1 mixed three evaluation
protocols, and several rows were leaked. The NDVI rows were full-sample random
muni-year CV (N=18,428); the AEF mean / AEF Hist / AEF Hist Ensemble models
were trained at the mun level on ALL municipalities (rf_yield_prediction.py
refits on everything before predicting ADCs; gb_aef_hist_ensemble.py never
holds out), so scoring them on the "held-out 20%" municipalities was partially
in-sample; and the ensemble prediction file covered only 2022 (N=450).

This script retrains every non-Agg-NN model in Table 1 with the seed-42
held-out 20% of municipalities EXCLUDED from training (the same split
agg_constrained_nn.py uses, reproduced exactly as in validation_mun_level.py),
keeping each model's published architecture and hyperparameter grid.
Hyperparameter selection uses an inner split grouped by MUNICIPALITY over the
training municipalities only, so no validation information leaks in. Held-out
predictions cover all years 2017-2024. validation_mun_level.py then scores all
models on the common held-out municipality-year sample.

Models / outputs (all to Data/predictions/):
  NDVI (masked)  mun_aefn2_masked_gb_holdout_preds.parquet          (mun-level)
  NDVI Hist.     mun_harmonic_h3_fixed_gb_holdout_preds.parquet     (mun-level, superseded)
  NDVI Q-Hist.   mun_harmonic_h3_quantile_gb_holdout_preds.parquet  (mun-level, superseded)
  AEF mean       adc_alpha_earth_holdout_preds_maize.parquet        (ADC-level)
  AEF Hist       adc_aef_hist_gb_holdout_preds.parquet              (ADC-level)
  AEF Hist Ens.  adc_aef_hist_ens_holdout_preds.parquet             (ADC-level)
(Agg-NN needs no retraining: adc_agg_nn_preds_maize_phase2.parquet is already
the development model trained on the other 80%.)

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 train_holdout_validation_models.py
"""
import os, sys, time
import numpy  as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from siap_yields import load_muni_yields

sys.stdout.reconfigure(line_buffering=True)

# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir    =  os.path.join(proj_dir, "Data", "alpha_earth")
feat_dir   =  os.path.join(proj_dir, "Data", "harmonic_features")
pred_dir   =  os.path.join(proj_dir, "Data", "predictions")
siap_path  =  os.path.join(home_dir, "Dropbox", "Projects",
                           "Maize_prediction", "Data", "SIAP", "Cleaned",
                           "siap_ag_prod_estimation_by_season.dta")

SEED       =  42
MIN_YEAR   =  2017
YEARS      =  list(range(2017, 2025))
CROP       =  'Maize'
SEASON     =  'Spring-Summer'

# published hyperparameter grids
HGB_GRID   =  [(0.05, 500, 5), (0.05, 1000, 7), (0.03, 1000, 6),
               (0.03, 1500, 8), (0.01, 2000, 6)]                    # gb_harmonic / mun_cv_aef_hist
GB_GRID    =  [(0.05, 1000, 7), (0.05, 1500, 8), (0.03, 2000, 8)]   # rf_yield_prediction IMPROVED
ENS_CFG    =  {'max_iter': 1500, 'max_depth': 8, 'learning_rate': 0.03,
               'min_samples_leaf': 5, 'random_state': 42, 'early_stopping': False}
N_PIX, K_SAMP, W_BIN  =  2, 5, 0.4                                  # gb_aef_hist_ensemble config

mean_cols  =  [f"A{d:02d}" for d in range(64)]
pct_cols   =  [f"A{d:02d}{s}" for d in range(64)
               for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']]


def calc_r2(y, yh):
    ss_tot =  np.mean((y - np.mean(y)) ** 2)
    return 1 - np.mean((y - yh) ** 2) / ss_tot if ss_tot > 0 else 0.0


# ── Reproduce the seed-42 80/20 municipality split ──────
# (identical to validation_mun_level.py / agg_constrained_nn.py)
aef_univ            =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"),
                                       columns=['adcid', 'year'])
aef_univ['muncode'] =  aef_univ['adcid'].str[:5]

siap            =  pd.read_stata(siap_path)
siap['muncode'] =  siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap['yield']   =  siap['q'] / siap['ha_planted']
sm  =  siap[(siap['name'] == CROP) &
            (siap['growing_season'] == SEASON) &
            (siap['year'] >= MIN_YEAR)][['muncode', 'year', 'yield']]

valid  =  sorted(set(zip(aef_univ['muncode'], aef_univ['year'])) &
                 set(zip(sm['muncode'], sm['year'])))
umuns  =  sorted(set(k[0] for k in valid))
np.random.seed(SEED)
np.random.shuffle(umuns)
val_muns  =  set(umuns[int(0.8 * len(umuns)):])
print(f"split: {len(umuns)} municipalities, {len(val_muns)} held out (20%)")


def select_cfg_hgb(X, y, groups):
    """Pick the best HGB_GRID config on an inner 80/20 split grouped by muni."""
    tr, va  =  next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                      random_state=SEED).split(X, y, groups))
    best  =  (None, -np.inf)
    for lr, n_est, depth in HGB_GRID:
        m  =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                            learning_rate=lr, min_samples_leaf=5,
                                            random_state=SEED)
        m.fit(X[tr], y[tr])
        r2  =  calc_r2(y[va], m.predict(X[va]))
        if r2 > best[1]:
            best  =  ((lr, n_est, depth), r2)
    return best[0]


# ============================================================
# A/B. NDVI harmonic histogram models (mun-level HistGB)
# ============================================================
def train_ndvi(features):
    t  =  time.time()
    feats            =  pd.read_parquet(os.path.join(feat_dir, f"muni_{features}.parquet"))
    feats['muncode'] =  feats['muncode'].astype(str).str.zfill(5)
    fcols  =  [c for c in feats.columns if c not in ('muncode', 'year')]

    yields            =  load_muni_yields(crop=CROP, season=SEASON)
    yields['muncode'] =  yields['muncode'].astype(str).str.zfill(5)
    merged  =  feats.merge(yields[['muncode', 'year', 'yield']], on=['muncode', 'year'])

    tr_df  =  merged[~merged['muncode'].isin(val_muns)]
    va_df  =  merged[merged['muncode'].isin(val_muns)]
    print(f"\nNDVI {features}: train {len(tr_df):,} mun-years "
          f"({tr_df['muncode'].nunique():,} muns), predict {len(va_df):,} held-out mun-years")

    X_tr  =  tr_df[fcols].to_numpy(np.float32)
    y_tr  =  tr_df['yield'].to_numpy()
    cfg   =  select_cfg_hgb(X_tr, y_tr, tr_df['muncode'].to_numpy())
    print(f"  cfg (inner group split): {cfg}")

    lr, n_est, depth  =  cfg
    m  =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                        learning_rate=lr, min_samples_leaf=5,
                                        random_state=SEED)
    m.fit(X_tr, y_tr)

    out  =  va_df[['muncode', 'year', 'yield']].copy()
    out['yield_pred']  =  m.predict(va_df[fcols].to_numpy(np.float32))
    path  =  os.path.join(pred_dir, f"mun_harmonic_{features}_gb_holdout_preds.parquet")
    out.to_parquet(path, index=False)
    print(f"  R2 (held-out muns, preview) = {calc_r2(out['yield'].values, out['yield_pred'].values):.4f}")
    print(f"  saved -> {os.path.basename(path)}  ({(time.time()-t)/60:.1f} min)")


# ============================================================
# A2. NDVI (masked) — cropland-masked aefn2 features (mun-level HistGB)
# ============================================================
# The paper's NDVI baseline as of 2026-08-15: replaces the two unmasked h3
# variants above with a single row. Features are the ADC-level aefn2 features
# aggregated to muni-year (area-weighted by the SIAP ag-land proxy) — a
# muni-level cropland extraction was never run. The cache is built by
# analysis/02_accuracy_maize/masked_muni_cv.py; run that first.
def train_ndvi_masked():
    t  =  time.time()
    cache  =  os.path.join(proj_dir, "Data", "cropland_features",
                           "muni_aefn2_masked.parquet")
    if not os.path.exists(cache):
        print(f"\nNDVI (masked): SKIPPED — {os.path.basename(cache)} not found; "
              f"run analysis/02_accuracy_maize/masked_muni_cv.py first")
        return
    feats            =  pd.read_parquet(cache)
    feats['muncode'] =  feats['muncode'].astype(str).str.zfill(5)
    fcols  =  [c for c in feats.columns if c not in ('muncode', 'year')]

    yields            =  load_muni_yields(crop=CROP, season=SEASON)
    yields['muncode'] =  yields['muncode'].astype(str).str.zfill(5)
    merged  =  feats.merge(yields[['muncode', 'year', 'yield']], on=['muncode', 'year'])

    tr_df  =  merged[~merged['muncode'].isin(val_muns)]
    va_df  =  merged[merged['muncode'].isin(val_muns)]
    print(f"\nNDVI (masked): train {len(tr_df):,} mun-years "
          f"({tr_df['muncode'].nunique():,} muns), predict {len(va_df):,} held-out mun-years")

    X_tr  =  tr_df[fcols].to_numpy(np.float32)
    y_tr  =  tr_df['yield'].to_numpy()
    cfg   =  select_cfg_hgb(X_tr, y_tr, tr_df['muncode'].to_numpy())
    print(f"  cfg (inner group split): {cfg}")

    lr, n_est, depth  =  cfg
    m  =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                        learning_rate=lr, min_samples_leaf=5,
                                        random_state=SEED)
    m.fit(X_tr, y_tr)

    out  =  va_df[['muncode', 'year', 'yield']].copy()
    out['yield_pred']  =  m.predict(va_df[fcols].to_numpy(np.float32))
    path  =  os.path.join(pred_dir, "mun_aefn2_masked_gb_holdout_preds.parquet")
    out.to_parquet(path, index=False)
    print(f"  R2 (held-out muns, preview) = {calc_r2(out['yield'].values, out['yield_pred'].values):.4f}")
    print(f"  saved -> {os.path.basename(path)}  ({(time.time()-t)/60:.1f} min)")


# ============================================================
# C. AEF mean (mun-level GradientBoosting, embeds + location + year)
# ============================================================
def train_aef_mean():
    t  =  time.time()
    ae             =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_muns.parquet"))
    ae['muncode']  =  (ae['CVE_ENT'].astype(str).str.zfill(2)
                       + ae['CVE_MUN'].astype(str).str.zfill(3))
    fcols  =  mean_cols + ['CVE_ENT', 'CVE_MUN', 'year']
    ae['CVE_ENT']  =  ae['CVE_ENT'].astype(int)
    ae['CVE_MUN']  =  ae['CVE_MUN'].astype(int)

    merged  =  ae.merge(sm[~sm['muncode'].str.endswith('000')],
                        on=['muncode', 'year'], how='inner')
    merged  =  merged[merged['yield'] > 0]
    tr_df   =  merged[~merged['muncode'].isin(val_muns)]
    print(f"\nAEF mean: train {len(tr_df):,} mun-years ({tr_df['muncode'].nunique():,} muns)")

    X_tr, y_tr  =  tr_df[fcols].to_numpy(), tr_df['yield'].to_numpy()
    gss  =  GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    itr, iva  =  next(gss.split(X_tr, y_tr, tr_df['muncode'].to_numpy()))
    best  =  (None, -np.inf)
    for lr, n_est, depth in GB_GRID:
        m  =  GradientBoostingRegressor(n_estimators=n_est, max_depth=depth,
                                        learning_rate=lr, min_samples_leaf=5,
                                        random_state=SEED, subsample=0.8)
        m.fit(X_tr[itr], y_tr[itr])
        r2  =  calc_r2(y_tr[iva], m.predict(X_tr[iva]))
        print(f"  GB lr={lr} n_est={n_est} depth={depth}  inner R2={r2:.4f}")
        if r2 > best[1]:
            best  =  ((lr, n_est, depth), r2)
    lr, n_est, depth  =  best[0]
    m  =  GradientBoostingRegressor(n_estimators=n_est, max_depth=depth,
                                    learning_rate=lr, min_samples_leaf=5,
                                    random_state=SEED, subsample=0.8)
    m.fit(X_tr, y_tr)
    print(f"  cfg: {best[0]}")

    adc             =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
    adc['CVE_ENT']  =  adc['adcid'].str[:2].astype(int)
    adc['CVE_MUN']  =  adc['adcid'].str[2:5].astype(int)
    adc['yield_pred']  =  m.predict(adc[fcols].fillna(0).to_numpy())
    path  =  os.path.join(pred_dir, "adc_alpha_earth_holdout_preds_maize.parquet")
    adc[['adcid', 'year', 'yield_pred']].to_parquet(path, index=False)
    print(f"  saved -> {os.path.basename(path)}  ({len(adc):,} rows, {(time.time()-t)/60:.1f} min)")


# ============================================================
# D. AEF Hist (mun-level HistGB on 384 pct/stdDev + 64 means)
# ============================================================
def train_aef_hist():
    t  =  time.time()
    pct             =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_hist.parquet"))
    pct['muncode']  =  (pct['CVE_ENT'].astype(str).str.zfill(2)
                        + pct['CVE_MUN'].astype(str).str.zfill(3))
    mean            =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_muns.parquet"))
    mean['muncode'] =  (mean['CVE_ENT'].astype(str).str.zfill(2)
                        + mean['CVE_MUN'].astype(str).str.zfill(3))
    feats  =  pct[['muncode', 'year'] + pct_cols].merge(
        mean[['muncode', 'year'] + mean_cols], on=['muncode', 'year'], how='inner')
    fcols  =  pct_cols + mean_cols

    yields            =  load_muni_yields(crop=CROP, season=SEASON)
    yields['muncode'] =  yields['muncode'].astype(str).str.zfill(5)
    yields  =  yields[~yields['muncode'].str.endswith('000')]
    merged  =  feats.merge(yields[['muncode', 'year', 'yield']], on=['muncode', 'year'])

    tr_df  =  merged[~merged['muncode'].isin(val_muns)]
    print(f"\nAEF Hist: train {len(tr_df):,} mun-years ({tr_df['muncode'].nunique():,} muns)")

    X_tr  =  tr_df[fcols].to_numpy(np.float32)
    y_tr  =  tr_df['yield'].to_numpy()
    cfg   =  select_cfg_hgb(X_tr, y_tr, tr_df['muncode'].to_numpy())
    print(f"  cfg (inner group split): {cfg}")
    lr, n_est, depth  =  cfg
    m  =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                        learning_rate=lr, min_samples_leaf=5,
                                        random_state=SEED)
    m.fit(X_tr, y_tr)

    # ADC-level prediction, year by year to bound memory
    outs  =  []
    for yr in YEARS:
        a  =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"),
                              filters=[('year', '==', yr)])
        am =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"),
                              filters=[('year', '==', yr)])
        a  =  a.merge(am[['adcid', 'year'] + mean_cols], on=['adcid', 'year'], how='inner')
        a  =  a[~a[pct_cols].isna().all(axis=1)]         # drop no-cropland ADCs (all-null features)
        o  =  a[['adcid', 'year']].copy()
        o['yield_pred']  =  m.predict(a[fcols].fillna(0).to_numpy(np.float32))
        outs.append(o)
        print(f"  {yr}: {len(o):,} ADC predictions")
    out   =  pd.concat(outs, ignore_index=True)
    path  =  os.path.join(pred_dir, "adc_aef_hist_gb_holdout_preds.parquet")
    out.to_parquet(path, index=False)
    print(f"  saved -> {os.path.basename(path)}  ({len(out):,} rows, {(time.time()-t)/60:.1f} min)")


# ============================================================
# E. AEF Hist Ensemble (subsampled bins + percentile, fixed config)
# ============================================================
def subsample_bins(train_df, bin_cols, K, N, seed=42):
    """Multinomial subsampling of mun bin distributions (as gb_aef_hist_ensemble.py)."""
    rng  =  np.random.default_rng(seed)
    n_dims, n_bins  =  64, 8
    n_mun  =  len(train_df)
    bin_arr  =  train_df[bin_cols].values.reshape(n_mun, n_dims, n_bins).astype(np.float64)
    bin_arr  =  np.clip(np.nan_to_num(bin_arr, nan=0.0), 0, None)
    sums  =  bin_arr.sum(axis=2, keepdims=True)
    sums[sums < 1e-8]  =  1.0
    bin_arr  =  bin_arr / sums
    yields  =  train_df['yield'].values
    all_bin, all_y  =  [], []
    for k in range(K):
        bin_out  =  np.zeros((n_mun, n_dims, n_bins), dtype=np.float32)
        for d in range(n_dims):
            for i in range(n_mun):
                bin_out[i, d, :]  =  rng.multinomial(N, bin_arr[i, d, :]) / N
        all_bin.append(bin_out.reshape(n_mun, n_dims * n_bins))
        all_y.append(yields)
        print(f"    subsample {k+1}/{K}")
    return np.vstack(all_bin), np.concatenate(all_y)


def train_ensemble():
    t  =  time.time()
    mun_bh    =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet"))
    bin_cols  =  sorted([c for c in mun_bh.columns if '_b' in c and c.startswith('A')])

    mun_pct             =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_hist.parquet"))
    mun_pct['muncode']  =  (mun_pct['CVE_ENT'].astype(str).str.zfill(2)
                            + mun_pct['CVE_MUN'].astype(str).str.zfill(3))
    mun_mean            =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_muns.parquet"))
    mun_mean['muncode'] =  (mun_mean['CVE_ENT'].astype(str).str.zfill(2)
                            + mun_mean['CVE_MUN'].astype(str).str.zfill(3))
    pct_combined  =  pct_cols + mean_cols
    mun_pct_full  =  mun_pct.merge(mun_mean[['muncode', 'year'] + mean_cols],
                                   on=['muncode', 'year'], how='inner')

    siap_train  =  sm[(sm['yield'] > 0) & ~sm['muncode'].str.endswith('000')].copy()
    siap_train  =  siap_train[~siap_train['muncode'].isin(val_muns)]   # HOLDOUT: exclude val muns

    train_bin  =  mun_bh.merge(siap_train, on=['muncode', 'year'], how='inner')
    train_pct  =  mun_pct_full.merge(siap_train, on=['muncode', 'year'], how='inner')
    print(f"\nAEF Hist Ens.: train mun-years bin={len(train_bin):,}, pct={len(train_pct):,} "
          f"(val muns excluded)")

    print("  training percentile model...")
    m_pct  =  HistGradientBoostingRegressor(**ENS_CFG)
    m_pct.fit(train_pct[pct_combined].fillna(0).values.astype(np.float32),
              train_pct['yield'].values)

    print(f"  subsampling bins (N={N_PIX}, K={K_SAMP})...")
    aug_bin, aug_y  =  subsample_bins(train_bin, bin_cols, K=K_SAMP, N=N_PIX, seed=SEED)
    print("  training subsampled bins model...")
    m_bin  =  HistGradientBoostingRegressor(**ENS_CFG)
    m_bin.fit(aug_bin.astype(np.float32), aug_y)

    # ADC-level ensemble predictions for ALL years, year by year
    outs  =  []
    for yr in YEARS:
        adc_bh  =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_binned_hist.parquet"),
                                   filters=[('year', '==', yr)])
        adc_p   =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"),
                                   filters=[('year', '==', yr)])
        adc_m   =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"),
                                   filters=[('year', '==', yr)])
        adc_p['muncode']  =  adc_p['adcid'].str[:5]
        adc_pf  =  adc_p.merge(adc_m[['adcid', 'year'] + mean_cols],
                               on=['adcid', 'year'], how='inner')
        combo   =  adc_bh.merge(adc_pf[['adcid', 'year'] + pct_combined],
                                on=['adcid', 'year'], how='inner')
        null    =  combo[bin_cols].isna().all(axis=1) | combo[pct_cols].isna().all(axis=1)
        combo   =  combo[~null]
        pb  =  m_bin.predict(combo[bin_cols].fillna(0).values.astype(np.float32)).clip(0)
        pp  =  m_pct.predict(combo[pct_combined].fillna(0).values.astype(np.float32)).clip(0)
        o   =  combo[['adcid', 'year']].copy()
        o['muncode']  =  o['adcid'].str[:5]
        o['pred']     =  W_BIN * pb + (1 - W_BIN) * pp
        outs.append(o)
        print(f"  {yr}: {len(o):,} ADC predictions ({int(null.sum()):,} all-null dropped)")
    out   =  pd.concat(outs, ignore_index=True)
    path  =  os.path.join(pred_dir, "adc_aef_hist_ens_holdout_preds.parquet")
    out[['adcid', 'muncode', 'year', 'pred']].to_parquet(path, index=False)
    print(f"  saved -> {os.path.basename(path)}  ({len(out):,} rows, {(time.time()-t)/60:.1f} min)")


if __name__ == '__main__':
    t0  =  time.time()
    # 'masked' only re-runs the NDVI (masked) row — the others are unchanged and
    # each cost many minutes.
    if 'masked' in sys.argv:
        train_ndvi_masked()
        print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
        sys.exit(0)
    train_ndvi('h3_fixed')
    train_ndvi('h3_quantile')
    train_ndvi_masked()
    train_aef_mean()
    train_aef_hist()
    train_ensemble()
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
