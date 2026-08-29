"""
Profile AEF Hist Ensemble accuracy by ADC characteristics.

Trains the maize AEF Hist Ensemble (same config as gb_aef_hist_ensemble.py),
generates ADC-level predictions for 2022, and computes accuracy metrics
stratified by:
  1. Maize planted area (land_input) terciles
  2. Total ADC planted area terciles
  3. Maize share of ADC area terciles
  4. Irrigation share terciles

Outputs a LaTeX table for Overleaf.

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 accuracy_profile_by_adc_chars.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)


# ============================================================
# CONFIGURATION  (must match gb_aef_hist_ensemble.py)
# ============================================================
N_PIX       =  2
K_SAMP      =  5
W_BIN       =  0.4
EVAL_YEAR   =  2022
CROP        =  'Maize'
SEASON      =  'Spring-Summer'
MIN_YEAR    =  2017
# ============================================================


# -- Directories ---------------------------------------------
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir    =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir   =  os.path.join(proj_dir, "Data", "predictions")
inegi_dir  =  os.path.join(proj_dir, "Data", "INEGI", "MD_lab_outputs")
ca2022_dir =  os.path.join(inegi_dir, "LM2304-CA22-2025-09-29-superficie_ENTREGA")
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
pct_combined =  pct_cols + mean_cols   # 448 features


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

def eval_subsample(df, ycol, pcol, label):
    sub = df[[ycol, pcol, 'muncode']].replace([np.inf, -np.inf], np.nan).dropna()
    n    = len(sub)
    if n < 10:
        return {'label': label, 'N': n, 'R2': np.nan, 'Btw': np.nan, 'Wtn': np.nan, 'RMSE': np.nan}
    ov   = r2(sub[ycol], sub[pcol])
    b    = between_r2(sub, ycol, pcol)
    w    = within_r2(sub, ycol, pcol)
    rmse = np.sqrt(np.mean((sub[ycol].values - sub[pcol].values)**2))
    return {'label': label, 'N': n, 'R2': ov, 'Btw': b, 'Wtn': w, 'RMSE': rmse}


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

# -- 1. Load & train (same as gb_aef_hist_ensemble.py) -------
print("Loading data...")
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


# -- 2. Load ADC test data (2022) ----------------------------
print(f"\nLoading ADC test data ({EVAL_YEAR})...")
adc_bh =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_binned_hist.parquet"))
adc_bh =  adc_bh[adc_bh['year'] == EVAL_YEAR].copy()

adc_pct_d =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs_hist.parquet"))
adc_pct_d =  adc_pct_d[adc_pct_d['year'] == EVAL_YEAR].copy()
adc_pct_d['muncode'] = adc_pct_d['adcid'].str[:5]

adc_m =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet"))
adc_m =  adc_m[adc_m['year'] == EVAL_YEAR].copy()

adc_pct_full = adc_pct_d.merge(
    adc_m[['adcid', 'year'] + mean_cols],
    on=['adcid', 'year'], how='inner'
)

adc_combo = adc_bh.merge(
    adc_pct_full[['adcid', 'year'] + pct_combined],
    on=['adcid', 'year'], how='inner'
)

# Drop rows whose features are ENTIRELY absent: the parquets carry a row for
# every ADC but leave values null where the ESA WorldCover cropland mask found
# no pixels. fillna(0) below would otherwise turn those into all-zero vectors
# and emit a constant prediction. Matches the identical drop in
# gb_aef_hist_ensemble.py, so the profile sample equals the main-table sample.
_bin_null =  adc_combo[bin_cols].isna().all(axis=1)
_pct_null =  adc_combo[pct_cols].isna().all(axis=1)
print(f"  dropping {int((_bin_null | _pct_null).sum()):,} ADCs with no cropland "
      f"pixels (all-null features)")
adc_combo =  adc_combo[~(_bin_null | _pct_null)].copy()


# -- 3. Ensemble predictions ---------------------------------
print(f"\nGenerating ensemble predictions (w_bin={W_BIN})...")
pred_bin = m_bin.predict(
    adc_combo[bin_cols].fillna(0).values.astype(np.float32)
).clip(0)
pred_pct = m_pct.predict(
    adc_combo[pct_combined].fillna(0).values.astype(np.float32)
).clip(0)

adc_combo['pred'] = W_BIN * pred_bin + (1 - W_BIN) * pred_pct
adc_combo['adc']  = adc_combo['adcid'].str.replace('-', '', regex=False)


# -- 4. Load INEGI ground truth + characteristics ------------
print("\nLoading INEGI census data...")
ca = pd.read_stata(os.path.join(ca2022_dir, "adc_land_use_ca22_adc07.dta"))

# Maize ground truth
ca_maize = ca[ca['name'] == CROP].copy()
gt = ca_maize[['adc', 'muncode', 'yield', 'land_input', 'share_irrig', 'num_up']].copy()

# Total ADC area across all crops
adc_total = ca.groupby('adc')['land_input'].sum().reset_index().rename(
    columns={'land_input': 'total_area'})

gt = gt.merge(adc_total, on='adc', how='left')
gt['maize_share'] = gt['land_input'] / gt['total_area']

# Spring-summer ground truth
ca_szn = pd.read_stata(os.path.join(ca2022_dir, "adc_land_szn_ca22_adc07.dta"))
gt_pv = ca_szn[(ca_szn['name'] == CROP) & (ca_szn['type'] == 'p-v')][
    ['adc', 'muncode', 'yield']
].rename(columns={'yield': 'yield_pv'})
gt = gt.merge(gt_pv, on=['adc', 'muncode'], how='left')


# -- 5. Merge predictions with GT ----------------------------
df = gt.merge(adc_combo[['adc', 'pred']], on='adc', how='left')
df_valid = df[df['pred'].notna()].copy()

print(f"  ADCs with predictions: {len(df_valid):,}")
print(f"  Unique municipalities: {df_valid['muncode'].nunique():,}")


# -- 6. Overall baseline (for reference) ---------------------
print(f"\n{'='*80}")
print(f"  {'Subsample':<40s} {'N':>8s} {'R2':>6s} {'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}")
print(f"  {'-'*74}")

all_results = []

# Overall
row = eval_subsample(df_valid, 'yield', 'pred', 'All ADCs')
all_results.append(row)
print(f"  {row['label']:<40s} {row['N']:>8,} {row['R2']:>6.3f} {row['Btw']:>6.3f} {row['Wtn']:>7.3f} {row['RMSE']:>6.3f}")

# Spring-summer only
df_pv = df_valid[df_valid['yield_pv'].notna()].copy()
row = eval_subsample(df_pv, 'yield_pv', 'pred', 'Spring-summer only')
all_results.append(row)
print(f"  {row['label']:<40s} {row['N']:>8,} {row['R2']:>6.3f} {row['Btw']:>6.3f} {row['Wtn']:>7.3f} {row['RMSE']:>6.3f}")


# -- 7. Profile by characteristics ---------------------------
def profile_by(df, char_col, char_label, n_groups=3):
    """Split df into terciles of char_col and compute metrics."""
    sub = df[df[char_col].notna() & df['pred'].notna()].copy()
    labels = ['Bottom tercile', 'Middle tercile', 'Top tercile']
    try:
        sub['_grp'] = pd.qcut(sub[char_col], n_groups, labels=False, duplicates='drop')
    except ValueError:
        return []

    # Map the surviving (deduplicated) bins back onto the correct tercile names:
    # the lowest group is the Bottom tercile, the highest the Top tercile, so that
    # a collapsed middle edge never mislabels the upper third as "Middle".
    n_grp = int(sub['_grp'].max()) + 1
    pos = {0: 0} if n_grp == 1 else {g: round(g * (len(labels) - 1) / (n_grp - 1)) for g in range(n_grp)}

    results = []
    for g in range(n_grp):
        grp_df = sub[sub['_grp'] == g]
        lo = grp_df[char_col].min()
        hi = grp_df[char_col].max()
        lbl = f"{char_label}: {labels[pos[g]]} [{lo:.2f}-{hi:.2f}]"
        row = eval_subsample(grp_df, 'yield', 'pred', lbl)
        results.append(row)
        print(f"  {row['label']:<55s} {row['N']:>8,} {row['R2']:>6.3f} {row['Btw']:>6.3f} {row['Wtn']:>7.3f} {row['RMSE']:>6.3f}")
    return results


def profile_irrig(df, char_col='share_irrig', char_label='Irrigation share'):
    """Irrigation share is heavily zero-inflated (~63% of ADCs have none), so
    plain terciles collapse into two bins. Report the un-irrigated ADCs as their
    own group, then split the positively-irrigated ADCs into terciles -- giving a
    meaningful top tercile of the most heavily irrigated ADCs."""
    sub = df[df[char_col].notna() & df['pred'].notna()].copy()
    results = []
    zero = sub[sub[char_col] <= 0]
    row = eval_subsample(zero, 'yield', 'pred', f"{char_label}: None (0\\%)")
    results.append(row)
    print(f"  {row['label']:<55s} {row['N']:>8,} {row['R2']:>6.3f} {row['Btw']:>6.3f} {row['Wtn']:>7.3f} {row['RMSE']:>6.3f}")
    results.extend(profile_by(sub[sub[char_col] > 0], char_col, char_label))
    return results


print("\n  --- By maize planted area (ha) ---")
res = profile_by(df_valid, 'land_input', 'Maize area')
all_results.extend(res)

print("\n  --- By total ADC planted area (ha) ---")
res = profile_by(df_valid, 'total_area', 'Total ADC area')
all_results.extend(res)

print("\n  --- By maize share of ADC area ---")
res = profile_by(df_valid, 'maize_share', 'Maize share')
all_results.extend(res)

print("\n  --- By number of production units ---")
res = profile_by(df_valid, 'num_up', 'Num. prod. units')
all_results.extend(res)


# -- 8. Write LaTeX table ------------------------------------
print("\nWriting LaTeX table...")

lines = []
lines.append(r"\begin{table}[!htbp]")
lines.append(r"\centering")
lines.append(r"\caption{AEF Hist Ensemble prediction accuracy by ADC characteristics, maize vs.\ INEGI 2022 census. RMSE in t/ha.}")
lines.append(r"\label{tab:accuracy_profile}")
lines.append(r"\footnotesize")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\hline")
lines.append(r"Subsample & $N$ & $R^2$ & Between $R^2$ & Within $R^2$ & RMSE \\")
lines.append(r"\hline")

current_section = None
for row in all_results:
    lbl = row['label']

    # Detect section change from the label prefix
    if ':' in lbl:
        section = lbl.split(':')[0].strip()
        if section != current_section:
            if current_section is not None:
                lines.append(r"\hline")
            current_section = section

    def fmt(v):
        if np.isnan(v):
            return "---"
        s = f"{v:.3f}"
        if v < 0:
            s = f"$-${abs(v):.3f}"
        return s

    # Clean label for LaTeX
    tex_lbl = lbl.replace('_', r'\_').replace('#', r'\#')

    lines.append(f"{tex_lbl} & {row['N']:,} & {fmt(row['R2'])} & {fmt(row['Btw'])} & {fmt(row['Wtn'])} & {fmt(row['RMSE'])} \\\\")

lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines)

# Save to tables dir and Overleaf
for out_path in [os.path.join(table_dir, "accuracy_profile_by_adc_chars.tex"),
                 os.path.join(overleaf, "accuracy_profile_by_adc_chars.tex")]:
    with open(out_path, 'w') as f:
        f.write(tex + "\n")
    print(f"  Written: {out_path}")

print(f"\n{tex}")
print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")
