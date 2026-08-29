"""
Evaluate AEF Hist Ensemble at the CIMMYT plot level.

Trains the maize AEF Hist Ensemble on SIAP mun-level data, then generates
predictions using CIMMYT plot-level AEF features (binned hist + percentile).
Compares to CIMMYT plot-level maize grain yields.

Prerequisites: run the GEE extraction scripts first:
  1. ee_alpha_earth_cimmyt.py         -> alpha_earth_cimmyt_plot.parquet (means)
  2. ee_alpha_earth_cimmyt_hist.py    -> alpha_earth_cimmyt_plot_hist.parquet (pct)
  3. ee_alpha_earth_cimmyt_binned_hist.py -> alpha_earth_cimmyt_plot_binned_hist.parquet (bins)
  Then consolidate each with consolidate_aef_cimmyt_all.py.

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 accuracy_cimmyt_plot_level.py
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


# -- Feature column names ------------------------------------
mean_cols =  [f"A{d:02d}" for d in range(64)]
pct_cols  =  []
for d in range(64):
    for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']:
        pct_cols.append(f"A{d:02d}{s}")
pct_combined =  pct_cols + mean_cols

bin_cols_template =  []
for d in range(64):
    for b in range(8):
        bin_cols_template.append(f"A{d:02d}_b{b}")


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
    n = len(sub)
    if n < 10:
        print(f"  {label:<50s}  N={n} (too few)")
        return
    ov   = r2(sub[ycol], sub[pcol])
    b    = between_r2(sub, ycol, pcol, gc)
    w    = within_r2(sub, ycol, pcol, gc)
    rmse = np.sqrt(np.mean((sub[ycol].values - sub[pcol].values)**2))
    print(f"  {label:<50s} {n:>8,} {ov:>6.3f} {b:>6.3f} {w:>7.3f} {rmse:>6.3f}")

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

# -- 1. Check that CIMMYT plot-level extractions exist -------
cimmyt_mean_path =  os.path.join(aef_dir, "alpha_earth_cimmyt_plot.parquet")
cimmyt_hist_path =  os.path.join(aef_dir, "alpha_earth_cimmyt_plot_hist.parquet")
cimmyt_bins_path =  os.path.join(aef_dir, "alpha_earth_cimmyt_plot_binned_hist.parquet")

for path, label in [(cimmyt_mean_path, "means"),
                     (cimmyt_hist_path, "percentiles"),
                     (cimmyt_bins_path, "binned hist")]:
    if not os.path.exists(path):
        print(f"ERROR: Missing {label} file: {path}")
        print(f"Run the GEE extraction scripts first. See docstring.")
        sys.exit(1)

print("Loading CIMMYT plot-level AEF features...")
cimmyt_mean =  pd.read_parquet(cimmyt_mean_path)
cimmyt_hist =  pd.read_parquet(cimmyt_hist_path)
cimmyt_bins =  pd.read_parquet(cimmyt_bins_path)

cimmyt_mean['plot_id'] =  cimmyt_mean['plot_id'].astype(str)
cimmyt_hist['plot_id'] =  cimmyt_hist['plot_id'].astype(str)
cimmyt_bins['plot_id'] =  cimmyt_bins['plot_id'].astype(str)

print(f"  Means:    {len(cimmyt_mean):,} plot-years")
print(f"  Pct/std:  {len(cimmyt_hist):,} plot-years")
print(f"  Bins:     {len(cimmyt_bins):,} plot-years")

# Merge all features for CIMMYT plots
cimmyt_pct_full = cimmyt_hist.merge(
    cimmyt_mean[['plot_id', 'year'] + mean_cols],
    on=['plot_id', 'year'], how='inner'
)
cimmyt_combo = cimmyt_bins.merge(
    cimmyt_pct_full[['plot_id', 'year'] + pct_combined],
    on=['plot_id', 'year'], how='inner'
)
print(f"  Combined: {len(cimmyt_combo):,} plot-years")


# -- 2. Train ensemble on mun-level SIAP data ----------------
print("\nLoading training data...")
mun_bh   =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_binned_hist.parquet"))
bin_cols  =  sorted([c for c in mun_bh.columns if '_b' in c and c.startswith('A')])

mun_pct  =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_mun_hist.parquet"))
mun_pct['CVE_ENT'] =  mun_pct['CVE_ENT'].astype(str).str.zfill(2)
mun_pct['CVE_MUN'] =  mun_pct['CVE_MUN'].astype(str).str.zfill(3)
mun_pct['muncode'] =  mun_pct['CVE_ENT'] + mun_pct['CVE_MUN']

mun_mean =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_muns.parquet"))
mun_mean['CVE_ENT'] =  mun_mean['CVE_ENT'].astype(str).str.zfill(2)
mun_mean['CVE_MUN'] =  mun_mean['CVE_MUN'].astype(str).str.zfill(3)
mun_mean['muncode'] =  mun_mean['CVE_ENT'] + mun_mean['CVE_MUN']

mun_pct_full = mun_pct.merge(
    mun_mean[['muncode', 'year'] + mean_cols],
    on=['muncode', 'year'], how='inner'
)

siap = pd.read_stata(siap_path)
siap['muncode'] =  siap['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap['yield']   =  siap['q'] / siap['ha_planted']
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

train_bin = mun_bh.merge(siap_train, on=['muncode', 'year'], how='inner')
train_pct = mun_pct_full.merge(siap_train, on=['muncode', 'year'], how='inner')
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


# -- 3. Predict on CIMMYT plot-level features ----------------
print("\nGenerating plot-level predictions...")
pred_bin = m_bin.predict(
    cimmyt_combo[bin_cols].fillna(0).values.astype(np.float32)
).clip(0)
pred_pct = m_pct.predict(
    cimmyt_combo[pct_combined].fillna(0).values.astype(np.float32)
).clip(0)

cimmyt_combo['pred'] = W_BIN * pred_bin + (1 - W_BIN) * pred_pct


# -- 4. Load CIMMYT yield data and merge --------------------
print("\nLoading CIMMYT yields...")
cimmyt_yields = pd.read_excel(
    os.path.join(cimmyt_dir, "Farmer_plots",
                 "2.-Sowing_harvest_yields_2012-2022_02.xlsx")
)
cy = cimmyt_yields[
    (cimmyt_yields['CROP'] == 'MAIZE')
    & (cimmyt_yields['PRODUCT.OBTAINED'] == 'GRAIN')
    & (cimmyt_yields['YEAR'] >= MIN_YEAR)
    & (cimmyt_yields['YEAR'] <= MAX_YEAR)
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'].notna())
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'] > 0)
].copy()
cy = cy.rename(columns={
    'PLOT.ID': 'plot_id',
    'YEAR': 'year',
    'ACTUAL.YIELD.(UNIT/HA)': 'yield_cimmyt',
})
cy['plot_id'] = cy['plot_id'].astype(str)

# Merge with predictions
df = cy.merge(
    cimmyt_combo[['plot_id', 'year', 'pred', 'muncode']],
    on=['plot_id', 'year'], how='inner'
)
print(f"  Matched plot-years: {len(df):,}")
print(f"  Mean CIMMYT yield:  {df['yield_cimmyt'].mean():.2f} t/ha")
print(f"  Mean predicted:     {df['pred'].mean():.2f} t/ha")


# -- 5. Evaluate ---------------------------------------------
print(f"\n{'='*80}")
print(f"  {'Model':<50s} {'N':>8s} {'R2':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}")
print(f"  {'-'*80}")

print("\n  --- All years (plot-level features) ---")
eval_row(df, 'yield_cimmyt', 'pred',
         'AEF Hist Ens. (plot-level)')

print("\n  --- By year ---")
for yr in sorted(df['year'].unique()):
    df_yr = df[df['year'] == yr].copy()
    eval_row(df_yr, 'yield_cimmyt', 'pred', f'  Year {yr}')

# Correlation
corr = df[['yield_cimmyt', 'pred']].corr().iloc[0, 1]
print(f"\n  Pearson correlation: {corr:.3f}")

print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")
