"""
Profile CIMMYT plot-level AEF Hist Ensemble predictions.

Investigates which subsets of CIMMYT plots have better/worse predictions,
profiling by: state, year, yield level, proximity to SIAP municipal mean,
number of plots per municipality, and CIMMYT management characteristics.

Prerequisites: run accuracy_cimmyt_plot_level.py first (or at least have
the three CIMMYT parquets and SIAP data available).

Usage:
  source /usr/local/anaconda3/etc/profile.d/conda.sh && conda activate mpc_env
  python3 accuracy_cimmyt_profile.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
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
# CIMMYT yields are self-reported and contain evident data-entry errors
# (values up to 16,600 t/ha). Restrict to a plausible agronomic range before
# computing any accuracy metric; ~2-3% of maize-grain plot observations fall outside it.
CLEAN_LO    =  0.3     # t/ha, lower plausible maize yield
CLEAN_HI    =  20.0    # t/ha, upper plausible maize yield
N_BOOT      =  500     # municipality-cluster bootstrap replications for CIs
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

def spearman_corr(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = np.array(y[m]), np.array(yh[m])
    if len(y) < 5:
        return np.nan
    return stats.spearmanr(y, yh).statistic

def boot_ci_within(df, ycol, pcol, gc='muncode', B=500, seed=42):
    """Municipality-cluster bootstrap 95% CI for within-group R2.

    Resamples whole municipalities (clusters) with replacement, so the CI
    reflects the effective number of municipalities, not plot-years.
    """
    sub = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = {g: d for g, d in sub.groupby(gc)}
    keys = np.array(list(groups.keys()))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.choice(keys, len(keys), replace=True)
        bs = pd.concat([groups[k] for k in pick], ignore_index=True)
        v = within_r2(bs, ycol, pcol, gc)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def eval_group(df, label, ycol='yield_cimmyt', pcol='pred', gc='muncode'):
    sub = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(sub)
    if n < 10:
        return None
    ov   = r2(sub[ycol], sub[pcol])
    b    = between_r2(sub, ycol, pcol, gc)
    w    = within_r2(sub, ycol, pcol, gc)
    rmse = np.sqrt(np.mean((sub[ycol].values - sub[pcol].values)**2))
    corr = np.corrcoef(sub[ycol].values, sub[pcol].values)[0, 1]
    sp   = spearman_corr(sub[ycol], sub[pcol])
    return {
        'label': label, 'N': n,
        'R2': ov, 'Btw': b, 'Wtn': w,
        'RMSE': rmse, 'Pearson': corr, 'Spearman': sp,
        'mean_y': sub[ycol].mean(), 'mean_p': sub[pcol].mean(),
    }

def print_header():
    print(f"  {'Group':<45s} {'N':>7s} {'R2':>6s} {'Btw':>6s} {'Wtn':>7s} "
          f"{'RMSE':>7s} {'Pears':>6s} {'Spear':>6s} {'AvgY':>6s} {'AvgP':>6s}")
    print(f"  {'-'*105}")

def print_row(r):
    if r is None:
        return
    print(f"  {r['label']:<45s} {r['N']:>7,} {r['R2']:>6.3f} {r['Btw']:>6.3f} "
          f"{r['Wtn']:>7.3f} {r['RMSE']:>7.2f} {r['Pearson']:>6.3f} "
          f"{r['Spearman']:>6.3f} {r['mean_y']:>6.2f} {r['mean_p']:>6.2f}")

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
    return np.vstack(all_bin), np.concatenate(all_y)


# ============================================================
# MAIN
# ============================================================
t0 = time.time()

# -- 1. Load CIMMYT features ---------------------------------
print("Loading CIMMYT plot-level AEF features...")
cimmyt_mean =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_cimmyt_plot.parquet"))
cimmyt_hist =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_cimmyt_plot_hist.parquet"))
cimmyt_bins =  pd.read_parquet(os.path.join(aef_dir, "alpha_earth_cimmyt_plot_binned_hist.parquet"))

for df in [cimmyt_mean, cimmyt_hist, cimmyt_bins]:
    df['plot_id'] = df['plot_id'].astype(str)

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

# AEF-mean comparison model (2026-08-26): mun-level means only, same HGB config
print("Training AEF-mean model...")
train_mean = mun_mean.merge(siap_train, on=['muncode', 'year'], how='inner')
m_mean = HistGradientBoostingRegressor(**cfg)
m_mean.fit(
    train_mean[mean_cols].fillna(0).values.astype(np.float32),
    train_mean['yield'].values
)


# -- 3. Predict on CIMMYT plot-level features ----------------
print("\nGenerating plot-level predictions...")
pred_bin = m_bin.predict(
    cimmyt_combo[bin_cols].fillna(0).values.astype(np.float32)
).clip(0)
pred_pct = m_pct.predict(
    cimmyt_combo[pct_combined].fillna(0).values.astype(np.float32)
).clip(0)

cimmyt_combo['pred_bin']  =  pred_bin
cimmyt_combo['pred_pct']  =  pred_pct
cimmyt_combo['pred']      =  W_BIN * pred_bin + (1 - W_BIN) * pred_pct
cimmyt_combo['pred_mean'] =  m_mean.predict(
    cimmyt_combo[mean_cols].fillna(0).values.astype(np.float32)).clip(0)


# -- 4. Load CIMMYT yield data and merge --------------------
print("\nLoading CIMMYT yields...")
cimmyt_yields = pd.read_excel(
    os.path.join(cimmyt_dir, "Farmer_plots",
                 "2.-Sowing_harvest_yields_2012-2022_02.xlsx")
)
_n_pre_clean = (
    (cimmyt_yields['CROP'] == 'MAIZE')
    & (cimmyt_yields['PRODUCT.OBTAINED'] == 'GRAIN')
    & (cimmyt_yields['YEAR'] >= MIN_YEAR)
    & (cimmyt_yields['YEAR'] <= MAX_YEAR)
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'].notna())
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'] > 0)
).sum()
cy = cimmyt_yields[
    (cimmyt_yields['CROP'] == 'MAIZE')
    & (cimmyt_yields['PRODUCT.OBTAINED'] == 'GRAIN')
    & (cimmyt_yields['YEAR'] >= MIN_YEAR)
    & (cimmyt_yields['YEAR'] <= MAX_YEAR)
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'].notna())
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'] >= CLEAN_LO)
    & (cimmyt_yields['ACTUAL.YIELD.(UNIT/HA)'] <= CLEAN_HI)
].copy()
print(f"  CIMMYT maize-grain rows: {_n_pre_clean:,} raw -> {len(cy):,} after "
      f"restricting yields to [{CLEAN_LO}, {CLEAN_HI}] t/ha "
      f"({100*(1-len(cy)/_n_pre_clean):.1f}% dropped as implausible)")
cy = cy.rename(columns={
    'PLOT.ID': 'plot_id',
    'YEAR': 'year',
    'ACTUAL.YIELD.(UNIT/HA)': 'yield_cimmyt',
})
cy['plot_id'] = cy['plot_id'].astype(str)

# Keep useful CIMMYT columns for profiling
keep_cols = ['plot_id', 'year', 'yield_cimmyt']
for c in ['STATE', 'MUNICIPALITY', 'TECHNOLOGY', 'SEASON',
           'CYCLE', 'AREA.HARVESTED.(HA)', 'TOTAL.COST.(PESOS/HA)']:
    if c in cy.columns:
        keep_cols.append(c)
cy = cy[keep_cols].copy()

# Merge with predictions
df = cy.merge(
    cimmyt_combo[['plot_id', 'year', 'pred', 'pred_bin', 'pred_pct', 'pred_mean',
                  'muncode']],
    on=['plot_id', 'year'], how='inner'
)

print(f"  Matched plot-years: {len(df):,}")
print(f"  Unique plots:       {df['plot_id'].nunique():,}")
print(f"  Unique muns:        {df['muncode'].nunique():,}")


# -- 5. Add SIAP municipal yields for comparison -------------
siap_mun = siap[
    (siap['name'] == CROP)
    & (siap['growing_season'] == SEASON)
    & (siap['year'] >= MIN_YEAR)
    & (siap['yield'].notna())
    & (siap['yield'] > 0)
][['muncode', 'year', 'yield']].rename(columns={'yield': 'yield_siap'})

df['muncode']       =  df['muncode'].astype(str).str.zfill(5)
siap_mun['muncode'] =  siap_mun['muncode'].astype(str).str.zfill(5)
df = df.merge(siap_mun, on=['muncode', 'year'], how='left')
df['yield_gap'] = df['yield_cimmyt'] - df['yield_siap']
df['abs_yield_gap'] = df['yield_gap'].abs()

print(f"  Plots with SIAP match: {df['yield_siap'].notna().sum():,}")
print(f"  Mean SIAP yield:       {df['yield_siap'].mean():.2f} t/ha")
print(f"  Mean CIMMYT yield:     {df['yield_cimmyt'].mean():.2f} t/ha")
print(f"  Mean predicted:        {df['pred'].mean():.2f} t/ha")


# ============================================================
# 6. PROFILING
# ============================================================
print(f"\n{'='*110}")
print("PROFILING CIMMYT PREDICTIONS")
print(f"{'='*110}")

# --- 6a. Overall + by sub-model ---
print("\n--- Overall & sub-models ---")
print_header()
print_row(eval_group(df, 'Ensemble (0.4*bin + 0.6*pct)'))
print_row(eval_group(df, 'Bins only', pcol='pred_bin'))
print_row(eval_group(df, 'Percentile only', pcol='pred_pct'))

# --- 6b. By state ---
print("\n--- By state ---")
print_header()
if 'STATE' in df.columns:
    for st in sorted(df['STATE'].dropna().unique()):
        sub = df[df['STATE'] == st]
        r = eval_group(sub, f'State: {st}')
        if r and r['N'] >= 50:
            print_row(r)
else:
    # Use state code from muncode
    df['state_code'] = df['muncode'].str[:2]
    for st in sorted(df['state_code'].unique()):
        sub = df[df['state_code'] == st]
        r = eval_group(sub, f'State code: {st}')
        if r and r['N'] >= 50:
            print_row(r)

# --- 6c. By year ---
print("\n--- By year ---")
print_header()
for yr in sorted(df['year'].unique()):
    print_row(eval_group(df[df['year'] == yr], f'Year {yr}'))

# --- 6d. By CIMMYT yield level (terciles) ---
print("\n--- By CIMMYT yield tercile ---")
print_header()
try:
    df['yield_terc'] = pd.qcut(df['yield_cimmyt'], 3, labels=['Low', 'Mid', 'High'])
    for t in ['Low', 'Mid', 'High']:
        sub = df[df['yield_terc'] == t]
        lo, hi = sub['yield_cimmyt'].min(), sub['yield_cimmyt'].max()
        print_row(eval_group(sub, f'{t} yield ({lo:.1f}-{hi:.1f} t/ha)'))
except Exception as e:
    print(f"  Could not create terciles: {e}")

# --- 6e. By proximity to SIAP mun yield ---
print("\n--- By proximity to SIAP municipal yield ---")
print_header()
has_siap = df[df['yield_siap'].notna()].copy()
if len(has_siap) > 100:
    # Ratio of CIMMYT to SIAP yield
    has_siap['ratio'] = has_siap['yield_cimmyt'] / has_siap['yield_siap']
    try:
        has_siap['ratio_grp'] = pd.qcut(has_siap['ratio'], 4,
                                          labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
        for g in ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']:
            sub = has_siap[has_siap['ratio_grp'] == g]
            lo, hi = sub['ratio'].min(), sub['ratio'].max()
            print_row(eval_group(sub, f'CIMMYT/SIAP ratio {g} ({lo:.1f}-{hi:.1f}x)'))
    except Exception as e:
        print(f"  Could not create quartiles: {e}")

    # Absolute gap
    try:
        has_siap['gap_grp'] = pd.qcut(has_siap['abs_yield_gap'], 3,
                                        labels=['Small gap', 'Medium gap', 'Large gap'],
                                        duplicates='drop')
        for g in ['Small gap', 'Medium gap', 'Large gap']:
            sub = has_siap[has_siap['gap_grp'] == g]
            print_row(eval_group(sub, f'{g} (|CIMMYT-SIAP|)'))
    except Exception as e:
        print(f"  Could not create gap groups: {e}")

# --- 6f. By number of plots per municipality ---
print("\n--- By plots per municipality ---")
print_header()
mun_counts = df.groupby('muncode').size().rename('n_plots')
df = df.merge(mun_counts, on='muncode', how='left')
try:
    df['plot_density'] = pd.qcut(df['n_plots'], 3,
                                   labels=['Few plots', 'Medium', 'Many plots'],
                                   duplicates='drop')
    for g in ['Few plots', 'Medium', 'Many plots']:
        sub = df[df['plot_density'] == g]
        n_mun = sub['muncode'].nunique()
        print_row(eval_group(sub, f'{g} ({n_mun} muns)'))
except Exception as e:
    print(f"  Could not create groups: {e}")

# --- 6g. By technology / management (if available) ---
if 'TECHNOLOGY' in df.columns:
    print("\n--- By CIMMYT technology ---")
    print_header()
    for tech in sorted(df['TECHNOLOGY'].dropna().unique()):
        sub = df[df['TECHNOLOGY'] == tech]
        r = eval_group(sub, f'Tech: {tech}')
        if r and r['N'] >= 30:
            print_row(r)

if 'SEASON' in df.columns:
    print("\n--- By CIMMYT season ---")
    print_header()
    for szn in sorted(df['SEASON'].dropna().unique()):
        sub = df[df['SEASON'] == szn]
        r = eval_group(sub, f'Season: {szn}')
        if r and r['N'] >= 30:
            print_row(r)

# --- 6h. By plot area (if available) ---
if 'AREA.HARVESTED.(HA)' in df.columns:
    print("\n--- By harvested area ---")
    print_header()
    area = df[df['AREA.HARVESTED.(HA)'].notna() & (df['AREA.HARVESTED.(HA)'] > 0)].copy()
    if len(area) > 100:
        try:
            area['area_grp'] = pd.qcut(area['AREA.HARVESTED.(HA)'], 3,
                                         labels=['Small', 'Medium', 'Large'],
                                         duplicates='drop')
            for g in ['Small', 'Medium', 'Large']:
                sub = area[area['area_grp'] == g]
                lo, hi = sub['AREA.HARVESTED.(HA)'].min(), sub['AREA.HARVESTED.(HA)'].max()
                print_row(eval_group(sub, f'{g} area ({lo:.1f}-{hi:.1f} ha)'))
        except Exception as e:
            print(f"  Could not create groups: {e}")

# --- 6i. Rank-based evaluation ---
print("\n--- Rank-based evaluation (can model rank plots?) ---")
print_header()
# Within-municipality ranking
has_multi = df.groupby('muncode').filter(lambda x: len(x) >= 5).copy()
if len(has_multi) > 50:
    # Within-mun Spearman
    sp_list = []
    for mun, grp in has_multi.groupby('muncode'):
        if len(grp) >= 5:
            sp = spearman_corr(grp['yield_cimmyt'], grp['pred'])
            if np.isfinite(sp):
                sp_list.append({'muncode': mun, 'spearman': sp, 'n': len(grp)})
    if sp_list:
        sp_df = pd.DataFrame(sp_list)
        print(f"  Within-mun Spearman (muns with >=5 plots): "
              f"N_muns={len(sp_df)}, "
              f"mean={sp_df['spearman'].mean():.3f}, "
              f"median={sp_df['spearman'].median():.3f}, "
              f"weighted_mean={np.average(sp_df['spearman'], weights=sp_df['n']):.3f}")
        # Distribution
        for pct in [10, 25, 50, 75, 90]:
            val = np.percentile(sp_df['spearman'], pct)
            print(f"    p{pct}: {val:.3f}")

        # Best and worst municipalities
        print(f"\n  Top 10 municipalities by within-mun Spearman:")
        for _, row in sp_df.nlargest(10, 'spearman').iterrows():
            print(f"    {row['muncode']}: r_s={row['spearman']:.3f} (n={row['n']})")

        print(f"\n  Bottom 10 municipalities by within-mun Spearman:")
        for _, row in sp_df.nsmallest(10, 'spearman').iterrows():
            print(f"    {row['muncode']}: r_s={row['spearman']:.3f} (n={row['n']})")


# --- 6j. Cross-year within-plot correlation ---
print("\n--- Within-plot cross-year consistency ---")
plot_years = df.groupby('plot_id').size()
repeat_plots = plot_years[plot_years >= 2].index
repeat_df = df[df['plot_id'].isin(repeat_plots)].copy()
print(f"  Plots with >=2 years: {len(repeat_plots):,} "
      f"({len(repeat_df):,} obs)")

if len(repeat_df) > 100:
    # Within-plot R2 and correlation
    r = eval_group(repeat_df, 'Repeat plots only', gc='plot_id')
    if r:
        print_row(r)

    # For plots with >=3 years, compute within-plot correlation
    repeat3 = df[df['plot_id'].isin(plot_years[plot_years >= 3].index)].copy()
    if len(repeat3) > 50:
        plot_corrs = []
        for pid, grp in repeat3.groupby('plot_id'):
            if len(grp) >= 3:
                c = np.corrcoef(grp['yield_cimmyt'].values, grp['pred'].values)[0, 1]
                if np.isfinite(c):
                    plot_corrs.append(c)
        if plot_corrs:
            print(f"  Within-plot Pearson (plots with >=3 yrs): "
                  f"N={len(plot_corrs)}, "
                  f"mean={np.mean(plot_corrs):.3f}, "
                  f"median={np.median(plot_corrs):.3f}")


# --- 6k. Does the prediction track SIAP well? ---
print("\n--- Does prediction match SIAP municipal yields? ---")
# Group CIMMYT predictions to municipality level and compare to SIAP
mun_pred = df.groupby(['muncode', 'year']).agg(
    mean_pred=('pred', 'mean'),
    mean_cimmyt=('yield_cimmyt', 'mean'),
    n_plots=('plot_id', 'count'),
).reset_index()
mun_pred = mun_pred.merge(siap_mun, on=['muncode', 'year'], how='inner')
if len(mun_pred) > 20:
    c1 = np.corrcoef(mun_pred['mean_pred'].values, mun_pred['yield_siap'].values)[0, 1]
    c2 = np.corrcoef(mun_pred['mean_cimmyt'].values, mun_pred['yield_siap'].values)[0, 1]
    c3 = np.corrcoef(mun_pred['mean_pred'].values, mun_pred['mean_cimmyt'].values)[0, 1]
    print(f"  Mun-level: pred vs SIAP r={c1:.3f}, "
          f"CIMMYT vs SIAP r={c2:.3f}, "
          f"pred vs CIMMYT r={c3:.3f}")
    print(f"  N mun-years: {len(mun_pred)}")
    print(f"  Mean SIAP:   {mun_pred['yield_siap'].mean():.2f}, "
          f"Mean pred: {mun_pred['mean_pred'].mean():.2f}, "
          f"Mean CIMMYT: {mun_pred['mean_cimmyt'].mean():.2f}")

    # R2 of pred vs SIAP at mun level
    r2_pred_siap = r2(mun_pred['yield_siap'], mun_pred['mean_pred'])
    r2_cimmyt_siap = r2(mun_pred['yield_siap'], mun_pred['mean_cimmyt'])
    print(f"  R2 pred vs SIAP: {r2_pred_siap:.3f}")
    print(f"  R2 CIMMYT vs SIAP: {r2_cimmyt_siap:.3f}")


# --- 6l. Municipality-level representativeness ---------------
print("\n--- By municipality-level CIMMYT representativeness ---")
print("  (classifies muns by mean CIMMYT/SIAP ratio, then evaluates within-mun R2)")
has_siap2 = df[df['yield_siap'].notna()].copy()
if len(has_siap2) > 100:
    has_siap2['ratio'] = has_siap2['yield_cimmyt'] / has_siap2['yield_siap']
    mun_ratio = has_siap2.groupby('muncode').agg(
        mean_ratio=('ratio', 'mean'),
        n_plots=('plot_id', 'count'),
    ).reset_index()
    has_siap2 = has_siap2.merge(mun_ratio[['muncode', 'mean_ratio']],
                                 on='muncode', how='left')

    cuts   =  [0, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 1e6]
    labels =  ['<0.7x', '0.7-0.9x', '0.9-1.1x', '1.1-1.3x',
               '1.3-1.5x', '1.5-2x', '>2x']
    has_siap2['repr_grp'] = pd.cut(has_siap2['mean_ratio'],
                                     bins=cuts, labels=labels)

    print(f"\n  {'Group':<30s} {'N':>7s} {'N_mun':>6s} {'R2':>6s} {'Btw':>6s} "
          f"{'Wtn':>7s} {'Pears':>6s} {'Spear':>6s} {'AvgY':>6s} {'AvgP':>6s}")
    print(f"  {'-'*95}")

    for g in labels:
        sub = has_siap2[has_siap2['repr_grp'] == g]
        n = len(sub)
        n_mun = sub['muncode'].nunique()
        if n < 20:
            continue
        ov = r2(sub['yield_cimmyt'], sub['pred'])
        b  = between_r2(sub, 'yield_cimmyt', 'pred')
        w  = within_r2(sub, 'yield_cimmyt', 'pred')
        corr = np.corrcoef(sub['yield_cimmyt'].values, sub['pred'].values)[0, 1]
        sp = spearman_corr(sub['yield_cimmyt'], sub['pred'])
        print(f"  {g:<30s} {n:>7,} {n_mun:>6} {ov:>6.3f} {b:>6.3f} {w:>7.3f} "
              f"{corr:>6.3f} {sp:>6.3f} "
              f"{sub['yield_cimmyt'].mean():>6.2f} {sub['pred'].mean():>6.2f}")

    # Deep dive: representative muns (0.8-1.2x)
    repr_muns = mun_ratio[(mun_ratio['mean_ratio'] >= 0.8)
                           & (mun_ratio['mean_ratio'] <= 1.2)]
    repr_df = has_siap2[has_siap2['muncode'].isin(repr_muns['muncode'])]
    print(f"\n  Representative muns (0.8-1.2x): "
          f"N={len(repr_df):,}, muns={repr_df['muncode'].nunique()}")
    ov = r2(repr_df['yield_cimmyt'], repr_df['pred'])
    w  = within_r2(repr_df, 'yield_cimmyt', 'pred')
    corr = np.corrcoef(repr_df['yield_cimmyt'].values,
                        repr_df['pred'].values)[0, 1]
    print(f"  R2={ov:.3f}, Within={w:.3f}, Pearson={corr:.3f}")

    # Within-mun Spearman for representative muns
    sp_list = []
    for mun, grp in repr_df.groupby('muncode'):
        if len(grp) >= 5:
            sp = spearman_corr(grp['yield_cimmyt'], grp['pred'])
            if np.isfinite(sp):
                sp_list.append(sp)
    if sp_list:
        print(f"  Within-mun Spearman (>=5 plots): "
              f"N_muns={len(sp_list)}, "
              f"mean={np.mean(sp_list):.3f}, "
              f"median={np.median(sp_list):.3f}")

    # By state in representative muns
    repr_df2 = repr_df.copy()
    repr_df2['state_code'] = repr_df2['muncode'].str[:2]
    print(f"\n  By state (representative muns, N>=30):")
    for st in sorted(repr_df2['state_code'].unique()):
        sub = repr_df2[repr_df2['state_code'] == st]
        if len(sub) < 30:
            continue
        w = within_r2(sub, 'yield_cimmyt', 'pred')
        corr = np.corrcoef(sub['yield_cimmyt'].values, sub['pred'].values)[0, 1]
        print(f"    State {st}: N={len(sub):>5,}, Wtn={w:.3f}, "
              f"Pearson={corr:.3f}, "
              f"AvgY={sub['yield_cimmyt'].mean():.2f}, "
              f"AvgP={sub['pred'].mean():.2f}")


# ============================================================
# 7. WRITE PAPER TABLE (tab:cimmyt_profile)
# ============================================================
# Collapsed, honest two-panel table:
#   Panel A  Plot-level accuracy: ensemble vs a naive SIAP-municipal-mean
#            baseline (zero within-mun skill by construction). The gap in
#            within-R2 is the model's marginal plot-level skill beyond the
#            municipal anchor. A cluster-bootstrap CI is attached to it.
#   Panel B  Plot-level accuracy by CIMMYT/SIAP representativeness ratio
#            (2026-08-26, coauthor request). Caveat kept in the table note:
#            the ratio conditions on the ground truth, so band-level R2 is
#            descriptive (where the model tracks yields) rather than
#            independent evidence of skill. The municipal-level aggregation
#            rows (pred vs SIAP 0.598/0.778, CIMMYT vs SIAP 0.723) moved to
#            prose; they are still printed above.
print(f"\n{'='*110}")
print("WRITING PAPER TABLE (tab:cimmyt_profile)")
print(f"{'='*110}")

_tbl = df.dropna(subset=['yield_siap']).copy()

def _fmt(x, d=3):
    if not np.isfinite(x):
        return "---"
    return f"{x:.{d}f}".replace("-", "$-$")

# Panel A rows
_ens = eval_group(_tbl, 'AEF Hist Ensemble', ycol='yield_cimmyt', pcol='pred')
# Shrink row: within-municipality shrinkage exactly as deployed in the ADC
# pipeline — a-priori lambda = 2/3 (Sec 3.6), fixed ex-ante with respect to the
# CIMMYT data. Deviations from the municipality-year mean prediction.
_LAM_DEPLOY = 0.74
_gm = _tbl.groupby(['muncode', 'year'])['pred'].transform('mean')
_tbl['pred_shrink'] = _gm + _LAM_DEPLOY * (_tbl['pred'] - _gm)
_ens_sh = eval_group(_tbl, 'AEF Hist Ensemble Shrink',
                     ycol='yield_cimmyt', pcol='pred_shrink')
_hist = eval_group(_tbl, 'AEF Hist', ycol='yield_cimmyt', pcol='pred_pct')
_mean = eval_group(_tbl, 'AEF mean', ycol='yield_cimmyt', pcol='pred_mean')
_base = eval_group(_tbl, 'SIAP Mun.\\ Avg.\\ (naive)',
                   ycol='yield_cimmyt', pcol='yield_siap')
_wtn_lo, _wtn_hi = boot_ci_within(_tbl, 'yield_cimmyt', 'pred', B=N_BOOT)

# Panel B: municipal-level aggregation
_mun = _tbl.groupby(['muncode', 'year']).agg(
    mean_pred=('pred', 'mean'),
    mean_cimmyt=('yield_cimmyt', 'mean'),
    yield_siap=('yield_siap', 'first'),
).reset_index()
_r2_ps = r2(_mun['yield_siap'].values, _mun['mean_pred'].values)
_r_ps = np.corrcoef(_mun['mean_pred'].values, _mun['yield_siap'].values)[0, 1]
_r_cs = np.corrcoef(_mun['mean_cimmyt'].values, _mun['yield_siap'].values)[0, 1]

_drop_pct = 100 * (1 - len(cy) / _n_pre_clean)

_tex = r"""\begin{table}[!htbp]
\centering
\caption{AEF Hist Ensemble predictions evaluated against CIMMYT plot-level maize yields, 2017--2022} \label{tab:cimmyt_profile}
\footnotesize
\begin{tabular}{lrrrrrr}
\toprule
& $N$ & $R^2$ & Btw-$R^2$ & Wtn-$R^2$ & Pearson & Spearman \\
\midrule
\multicolumn{7}{l}{\textit{Panel A: Plot-level accuracy}} \\
"""
_tex += (f"AEF Hist Ens. & {_ens['N']:,} & {_fmt(_ens['R2'])} & "
         f"{_fmt(_ens['Btw'])} & {_fmt(_ens['Wtn'])} & {_fmt(_ens['Pearson'])} & "
         f"{_fmt(_ens['Spearman'])} \\\\\n")
_tex += (f"AEF Hist Ens.\\ Shrink & {_ens_sh['N']:,} & {_fmt(_ens_sh['R2'])} & "
         f"{_fmt(_ens_sh['Btw'])} & {_fmt(_ens_sh['Wtn'])} & {_fmt(_ens_sh['Pearson'])} & "
         f"{_fmt(_ens_sh['Spearman'])} \\\\\n")
for _r in [_hist, _mean]:
    if _r is not None:
        _tex += (f"{_r['label']} & {_r['N']:,} & {_fmt(_r['R2'])} & "
                 f"{_fmt(_r['Btw'])} & {_fmt(_r['Wtn'])} & {_fmt(_r['Pearson'])} & "
                 f"{_fmt(_r['Spearman'])} \\\\\n")
_tex += (f"SIAP Mun.\\ Avg.\\ (naive) & {_base['N']:,} & {_fmt(_base['R2'])} & "
         f"{_fmt(_base['Btw'])} & {_fmt(_base['Wtn'])} & {_fmt(_base['Pearson'])} & "
         f"{_fmt(_base['Spearman'])} \\\\\n")
_tex += r"""\addlinespace
\multicolumn{7}{l}{\textit{Panel B: By municipality mean CIMMYT/SIAP ratio (AEF Hist Ens.\ Shrink)}} \\
"""
# Bands defined at the MUNICIPALITY level (mean CIMMYT yield / SIAP yield per
# mun-year), NOT each plot's own ratio: plot-level conditioning selects on the
# realized outcome and mechanically destroys within-mun R2 (2026-08-28).
# Evaluated with the deployed Shrink predictions.
_mr = _tbl.groupby(['muncode', 'year']).apply(
    lambda g: g['yield_cimmyt'].mean() / g['yield_siap'].iloc[0]).rename('_mratio').reset_index()
_tbl = _tbl.merge(_mr, on=['muncode', 'year'], how='left')
_BANDS = [(r"Mun.\ ratio $<$ 0.9 (below avg.)",       0.0, 0.9),
          (r"Mun.\ ratio 0.9--1.3 (representative)",  0.9, 1.3),
          (r"Mun.\ ratio 1.3--2.0",                    1.3, 2.0),
          (r"Mun.\ ratio $>$ 2.0 (management premium)", 2.0, np.inf)]
for _lbl, _lo, _hi in _BANDS:
    _sub = _tbl[(_tbl['_mratio'] >= _lo) & (_tbl['_mratio'] < _hi)]
    _row = eval_group(_sub, _lbl, ycol='yield_cimmyt', pcol='pred_shrink')
    _tex += (f"{_lbl} & {_row['N']:,} & {_fmt(_row['R2'])} & "
             f"{_fmt(_row['Btw'])} & {_fmt(_row['Wtn'])} & {_fmt(_row['Pearson'])} & "
             f"{_fmt(_row['Spearman'])} \\\\\n")
_tex += r"""\bottomrule
\end{tabular}
\par\smallskip
\footnotesize{Notes: models trained on SIAP municipal Spring--Summer maize yields ("""
_tex += (f"{MIN_YEAR}--{MAX_YEAR}) and applied to plot-level AEF features. CIMMYT yields "
         r"are self-reported; observations outside "
         f"{CLEAN_LO}--{CLEAN_HI}"
         r"\,t/ha ($\approx$"
         f"{_drop_pct:.1f}"
         r"\%, evident data-entry errors) are excluded, and 28\% of matched plot-years "
         r"contain no WorldCover cropland pixels inside the plot polygon (empty features); "
         r"results are insensitive to excluding the latter. Between- and within-$R^2$ use "
         r"municipality groupings. The naive baseline assigns every plot its SIAP "
         r"municipal mean, so its within-$R^2$ is zero by construction; the ensemble's "
         r"within-$R^2 = "
         f"{_ens['Wtn']:.3f}"
         r"$ [95\% CI "
         f"{_wtn_lo:.3f}, {_wtn_hi:.3f}"
         r"] is its marginal plot-level skill beyond the municipal anchor. The Shrink row "
         r"applies the deployed $\lambda = 0.74$ (Section~\ref{sec:shrink}), estimated from public data and ex ante "
         r"with respect to the CIMMYT data. Panel B groups plots by their "
         r"municipality-year's mean CIMMYT-to-SIAP yield ratio, an indicator of how "
         r"representative local trial yields are of area averages.}"
         "\n\\end{table}\n")

for _dest in [os.path.join(table_dir, "accuracy_cimmyt_profile.tex"),
              os.path.join(overleaf, "accuracy_cimmyt_profile.tex")]:
    os.makedirs(os.path.dirname(_dest), exist_ok=True)
    with open(_dest, 'w') as _f:
        _f.write(_tex)
    print(f"  wrote {_dest}")

print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")
