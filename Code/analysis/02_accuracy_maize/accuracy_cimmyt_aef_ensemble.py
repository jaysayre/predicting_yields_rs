"""
Evaluate AEF Hist Ensemble predictions against CIMMYT plot-level maize yields.

Since per-plot AEF extractions don't exist, this script:
  1. Trains the maize AEF Hist Ensemble (same config as gb_aef_hist_ensemble.py)
  2. Generates ADC-level predictions for all years 2017-2022
  3. Maps CIMMYT plots to ADCs via the geometry crosswalk
  4. Compares ADC-level predictions to CIMMYT plot yields

This is a cross-scale evaluation: model predicts ADC-average yields,
ground truth is individual field yields.

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 accuracy_cimmyt_aef_ensemble.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# CONFIGURATION
# ============================================================
N_PIX       =  2
K_SAMP      =  5
W_BIN       =  0.4
CROP        =  'Maize'
SEASON      =  'Spring-Summer'
MIN_YEAR    =  2017
MAX_YEAR    =  2022
# ============================================================


# -- Directories ---------------------------------------------
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir    =  os.path.join(proj_dir, "Data", "alpha_earth")
cimmyt_dir =  os.path.join(proj_dir, "Data", "CIMMYT")
table_dir  =  os.path.join(proj_dir, "tables")
overleaf   =  os.path.join(home_dir, "Dropbox", "Overleaf",
                           "Predicting Yields at Scale using RS")
siap_path  =  os.path.join(home_dir, "Dropbox", "Projects",
                            "Maize_prediction", "Data", "SIAP", "Cleaned",
                            "siap_ag_prod_estimation_by_season.dta")

os.makedirs(table_dir, exist_ok=True)


# -- Feature column names ------------------------------------
mean_cols =  [f"A{d:02d}" for d in range(64)]
pct_cols  =  []
for d in range(64):
    for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']:
        pct_cols.append(f"A{d:02d}{s}")
pct_combined =  pct_cols + mean_cols


# -- HistGB config -------------------------------------------
cfg = {
    'max_iter':        1500,
    'max_depth':       8,
    'learning_rate':   0.03,
    'min_samples_leaf': 5,
    'random_state':    42,
    'early_stopping':  False,
}


# -- Helpers -------------------------------------------------
def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = np.array(y[m]), np.array(yh[m])
    if len(y) < 2:
        return np.nan
    st = np.sum((y - np.mean(y))**2)
    return 1 - np.sum((y - yh)**2) / st if st > 0 else np.nan

def within_r2(df, y, p, gc='muncode'):
    sub = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    c = sub.groupby(gc).size()
    sub = sub[sub[gc].isin(c[c >= 2].index)]
    if len(sub) == 0:
        return np.nan
    gm = sub.groupby(gc)[[y, p]].transform('mean')
    return r2(sub[y] - gm[y], sub[p] - gm[p])

def between_r2(df, y, p, gc='muncode'):
    sub = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g = sub.groupby(gc)[[y, p]].mean()
    return r2(g[y], g[p])

def eval_row(df, ycol, pcol, label, gc='muncode'):
    sub = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    n    = len(sub)
    if n < 10:
        print(f"  {label:<50s}  N={n} (too few)")
        return {'N': n, 'R2': np.nan, 'Btw': np.nan, 'Wtn': np.nan, 'RMSE': np.nan}
    ov   = r2(sub[ycol], sub[pcol])
    b    = between_r2(sub, ycol, pcol, gc)
    w    = within_r2(sub, ycol, pcol, gc)
    rmse = np.sqrt(np.mean((sub[ycol].values - sub[pcol].values)**2))
    print(f"  {label:<50s} {n:>8,} {ov:>6.3f} {b:>6.3f} {w:>7.3f} {rmse:>6.3f}")
    return {'N': n, 'R2': ov, 'Btw': b, 'Wtn': w, 'RMSE': rmse}

def subsample_bins(train_df, bin_cols, K, N, seed=42):
    rng = np.random.default_rng(seed)
    n_dims, n_bins = 64, 8
    n_mun = len(train_df)
    bin_arr = train_df[bin_cols].values.reshape(n_mun, n_dims, n_bins).astype(np.float64)
    bin_arr = np.clip(np.nan_to_num(bin_arr, nan=0.0), 0, None)
    sums = bin_arr.sum(axis=2, keepdims=True)
    sums[sums < 1e-8] = 1.0
    bin_arr = bin_arr / sums
    yields = train_df['yield'].values
    all_bin, all_y = [], []
    for k in range(K):
        bin_out = np.zeros((n_mun, n_dims, n_bins), dtype=np.float32)
        for d in range(n_dims):
            for i in range(n_mun):
                bin_out[i, d, :] = rng.multinomial(N, bin_arr[i, d, :]) / N
        all_bin.append(bin_out.reshape(n_mun, n_dims * n_bins))
        all_y.append(yields)
        print(f"    subsample {k+1}/{K}", flush=True)
    return np.vstack(all_bin), np.concatenate(all_y)


# ============================================================
# MAIN
# ============================================================
t0 = time.time()

# -- 1. Train ensemble (same as gb_aef_hist_ensemble.py) -----
print("Loading training data...")
mun_bh   =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet"))
bin_cols  =  sorted([c for c in mun_bh.columns if '_b' in c and c.startswith('A')])

mun_pct  =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_hist.parquet"))
mun_pct['CVE_ENT']  =  mun_pct['CVE_ENT'].astype(str).str.zfill(2)
mun_pct['CVE_MUN']  =  mun_pct['CVE_MUN'].astype(str).str.zfill(3)
mun_pct['muncode']  =  mun_pct['CVE_ENT'] + mun_pct['CVE_MUN']

mun_mean =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_muns.parquet"))
mun_mean['CVE_ENT']  =  mun_mean['CVE_ENT'].astype(str).str.zfill(2)
mun_mean['CVE_MUN']  =  mun_mean['CVE_MUN'].astype(str).str.zfill(3)
mun_mean['muncode']  =  mun_mean['CVE_ENT'] + mun_mean['CVE_MUN']

mun_pct_full = mun_pct.merge(
    mun_mean[['muncode', 'year'] + mean_cols],
    on=['muncode', 'year'], how='inner'
)

siap = pd.read_stata(siap_path)
siap['muncode']  =  siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap['yield']    =  siap['q'] / siap['ha_planted']
siap_train = siap[
    (siap['name'] == CROP)
    & (siap['growing_season'] == SEASON)
    & (siap['year'] >= MIN_YEAR)
]
siap_train = siap_train[
    siap_train['yield'].notna()
    & (siap_train['yield'] > 0)
    & ~siap_train['muncode'].str.endswith('000')
][['muncode', 'year', 'yield']]

train_bin =  mun_bh.merge(siap_train, on=['muncode', 'year'], how='inner')
train_pct =  mun_pct_full.merge(siap_train, on=['muncode', 'year'], how='inner')
print(f"  Training mun-years: bin={len(train_bin):,}, pct={len(train_pct):,}")

print("\nTraining percentile model...")
m_pct = HistGradientBoostingRegressor(**cfg)
m_pct.fit(
    train_pct[pct_combined].fillna(0).values.astype(np.float32),
    train_pct['yield'].values
)

print(f"\nSubsampling bins (N={N_PIX}, K={K_SAMP})...")
aug_bin, aug_y = subsample_bins(train_bin, bin_cols, K=K_SAMP, N=N_PIX, seed=42)

print("Training subsampled bins model...")
m_bin = HistGradientBoostingRegressor(**cfg)
m_bin.fit(aug_bin.astype(np.float32), aug_y)


# -- 2. Load ADC features for ALL years 2017-2022 -----------
print(f"\nLoading ADC features (all years {MIN_YEAR}-{MAX_YEAR})...")

adc_bh_all =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_binned_hist.parquet"))
adc_bh_all =  adc_bh_all[(adc_bh_all['year'] >= MIN_YEAR) & (adc_bh_all['year'] <= MAX_YEAR)].copy()

adc_pct_all =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"))
adc_pct_all =  adc_pct_all[(adc_pct_all['year'] >= MIN_YEAR) & (adc_pct_all['year'] <= MAX_YEAR)].copy()
adc_pct_all['muncode'] = adc_pct_all['adcid'].str[:5]

adc_m_all =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
adc_m_all =  adc_m_all[(adc_m_all['year'] >= MIN_YEAR) & (adc_m_all['year'] <= MAX_YEAR)].copy()

adc_pct_full_all = adc_pct_all.merge(
    adc_m_all[['adcid', 'year'] + mean_cols],
    on=['adcid', 'year'], how='inner'
)

adc_combo_all = adc_bh_all.merge(
    adc_pct_full_all[['adcid', 'year'] + pct_combined],
    on=['adcid', 'year'], how='inner'
)

print(f"  ADC-year observations: {len(adc_combo_all):,}")


# -- 3. Generate ensemble predictions for all ADC-years ------
print("\nGenerating ensemble predictions...")
pred_bin = m_bin.predict(
    adc_combo_all[bin_cols].fillna(0).values.astype(np.float32)
).clip(0)
pred_pct = m_pct.predict(
    adc_combo_all[pct_combined].fillna(0).values.astype(np.float32)
).clip(0)

adc_combo_all['pred'] = W_BIN * pred_bin + (1 - W_BIN) * pred_pct
# Clean adcid: remove dashes to match CIMMYT geo format
adc_combo_all['adcid_clean'] = adc_combo_all['adcid'].str.replace('-', '', regex=False)
adc_combo_all['muncode'] = adc_combo_all['adcid'].str[:5]

print(f"  Predictions generated: {len(adc_combo_all):,}")
print(f"  Years: {sorted(adc_combo_all['year'].unique())}")


# -- 4. Load CIMMYT data ------------------------------------
print("\nLoading CIMMYT data...")

# CIMMYT yields
cimmyt_yields = pd.read_excel(
    os.path.join(cimmyt_dir, "Farmer_plots",
                 "2.-Sowing_harvest_yields_2012-2022_02.xlsx")
)

# Filter to maize, grain, years with AEF
cimmyt = cimmyt_yields[
    (cimmyt_yields['CROP'] == 'MAIZE')
    & (cimmyt_yields['PRODUCT.OBTAINED'] == 'GRAIN')
    & (cimmyt_yields['YEAR'] >= MIN_YEAR)
    & (cimmyt_yields['YEAR'] <= MAX_YEAR)
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'].notna())
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'] > 0)
].copy()
cimmyt = cimmyt.rename(columns={
    'PLOT.ID': 'plot_id',
    'YEAR': 'year',
    'ACTUAL.YIELD.(UNIT/HA)': 'yield_cimmyt',
    'STATE': 'state',
    'MUNICIPALITY': 'municipality',
})
cimmyt['plot_id'] = cimmyt['plot_id'].astype(str)
print(f"  CIMMYT maize grain obs (2017-2022): {len(cimmyt):,}")
print(f"  Unique plots: {cimmyt['plot_id'].nunique():,}")
print(f"  Years: {sorted(cimmyt['year'].unique())}")

# CIMMYT geometries -> ADC mapping
geo = pd.read_csv(os.path.join(cimmyt_dir, "cimmyt_plot_geometries_for_ee.csv"))
geo['plot_id'] = geo['plot_id'].astype(str)
geo['adcid_clean'] = geo['adcid'].str.replace('-', '', regex=False) if geo['adcid'].dtype == object else geo['adcid'].astype(str)
geo['muncode'] = geo['muncode'].astype(str).str.zfill(5)
print(f"  CIMMYT geometries: {len(geo):,}")
print(f"  With ADC: {geo['adcid'].notna().sum():,}")

# Merge CIMMYT yields with geometry (plot_id)
cimmyt = cimmyt.merge(geo[['plot_id', 'adcid', 'muncode', 'area_ha']], on='plot_id', how='inner')
print(f"  After geo merge: {len(cimmyt):,}")
cimmyt['adcid_clean'] = cimmyt['adcid'].str.replace('-', '', regex=False)


# -- 5. Merge CIMMYT with ADC predictions -------------------
print("\nMerging CIMMYT plots with ADC predictions...")
df = cimmyt.merge(
    adc_combo_all[['adcid_clean', 'year', 'pred', 'muncode']].rename(
        columns={'muncode': 'muncode_adc'}),
    left_on=['adcid_clean', 'year'],
    right_on=['adcid_clean', 'year'],
    how='inner'
)
print(f"  Matched CIMMYT plot-years: {len(df):,}")
print(f"  Unique plots: {df['plot_id'].nunique():,}")
print(f"  Mean CIMMYT yield: {df['yield_cimmyt'].mean():.2f} t/ha")
print(f"  Mean predicted:    {df['pred'].mean():.2f} t/ha")


# -- 6. Evaluate ---------------------------------------------
print(f"\n{'='*80}")
print(f"  {'Model':<50s} {'N':>8s} {'R2':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}")
print(f"  {'-'*80}")

print("\n  --- All years (2017-2022) ---")
eval_row(df, 'yield_cimmyt', 'pred', 'AEF Hist Ens. (ADC pred vs CIMMYT plot)')

print("\n  --- By year ---")
for yr in sorted(df['year'].unique()):
    df_yr = df[df['year'] == yr].copy()
    eval_row(df_yr, 'yield_cimmyt', 'pred', f'  Year {yr}')

print("\n  --- Summer season only ---")
summer_mask = cimmyt.columns.str.contains('SEASON', case=False)
if 'WINTER/SUMER.\xa0SEASON' in df.columns:
    df_summer = df[df['WINTER/SUMER.\xa0SEASON'] == 'SUMMER'].copy()
    if len(df_summer) > 0:
        eval_row(df_summer, 'yield_cimmyt', 'pred', 'Summer only')
    df_winter = df[df['WINTER/SUMER.\xa0SEASON'] == 'WINTER'].copy()
    if len(df_winter) > 0:
        eval_row(df_winter, 'yield_cimmyt', 'pred', 'Winter only')

# By hydric regime
if 'HYDRIC.REGIME' in df.columns:
    print("\n  --- By hydric regime ---")
    for regime in df['HYDRIC.REGIME'].dropna().unique():
        df_reg = df[df['HYDRIC.REGIME'] == regime].copy()
        if len(df_reg) >= 50:
            eval_row(df_reg, 'yield_cimmyt', 'pred', f'  {regime}')

# Simple correlation
corr = df[['yield_cimmyt', 'pred']].corr().iloc[0, 1]
print(f"\n  Pearson correlation: {corr:.3f}")

# Save results
out_path = os.path.join(proj_dir, "tables", "accuracy_cimmyt_aef_ensemble.txt")
with open(out_path, 'w') as f:
    f.write("CIMMYT Plot-Level Evaluation of AEF Hist Ensemble\n")
    f.write(f"Matched plot-years: {len(df):,}\n")
    f.write(f"Overall R2: {r2(df['yield_cimmyt'], df['pred']):.3f}\n")
    sub = df[['yield_cimmyt', 'pred', 'muncode']].replace([np.inf, -np.inf], np.nan).dropna()
    f.write(f"Between R2: {between_r2(sub, 'yield_cimmyt', 'pred'):.3f}\n")
    f.write(f"Within R2:  {within_r2(sub, 'yield_cimmyt', 'pred'):.3f}\n")
    f.write(f"RMSE: {np.sqrt(np.mean((sub['yield_cimmyt'].values - sub['pred'].values)**2)):.3f}\n")
    f.write(f"Correlation: {corr:.3f}\n")
print(f"\nSaved: {out_path}")

print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")
