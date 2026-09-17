"""
Generate accuracy_africa_profile.tex (Table \ref{tab:africa_profile}).

East-Africa extension of the AEF Hist Ensemble (w=0.5 bins + 0.6 percentile,
HistGradientBoosting), reproducing all four panels:

  Panel A  Admin-2 LOYO CV on HarvestStat Africa maize yields:
             - all countries pooled (group by admin1_id)
             - Kenya only (47 counties; HarvestStat keys Kenya at admin_1=county)
  Panel B  Kenya admin-2 LOYO, by held-out year
  Panel C  Kenya-trained model applied to GROW-Africa One Acre Fund maize plots,
           additive ex-post corrected within county, by representativeness tier
  Panel D  Selected Kenya counties (plot level, corrected)

Plot -> county mapping: GROW gadm_l1 (GADM L1_GID) -> Africa_L1_LUT.csv admin1
name -> HarvestStat Kenya admin_1 (county) -> fnid. Correction anchors each
plot to its county-year HarvestStat yield (mirrors the Mexico same-year SIAP
correction). Representativeness ratio = mean(plot yield)/HarvestStat county yield.

Writes to tables/ (review before replacing the Overleaf copy).
Run:  ~/miniforge3/envs/geo_env/bin/python accuracy_africa_profile.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

# ── Config (matches Mexico) ─────────────────────────────
N_PIX, K_SAMP, W_BIN, SEED = 2, 5, 0.5, 42
cfg = dict(max_iter=1500, max_depth=8, learning_rate=0.03,
           min_samples_leaf=5, random_state=SEED, early_stopping=False)

home_dir   = os.path.expanduser("~")
proj_dir   = os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
hv_dir     = os.path.join(proj_dir, "Data", "HarvestStat_Africa")
grow_dir   = os.path.join(proj_dir, "Data", "GROW_Africa")
lut_dir    = os.path.join(grow_dir, "Spatial-Look-Up-Tables")
table_dir  = os.path.join(proj_dir, "tables")
os.makedirs(table_dir, exist_ok=True)

mean_cols = [f"A{d:02d}" for d in range(64)]
pct_cols  = [f"A{d:02d}{s}" for d in range(64)
             for s in ['_p10', '_p25', '_p50', '_p75', '_p90', '_stdDev']]
pct_comb  = pct_cols + mean_cols
bin_cols  = [f"A{d:02d}_b{b}" for d in range(64) for b in range(8)]


# ── Metrics ──────────────────────────────────────────────
def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = np.asarray(y)[m], np.asarray(yh)[m]
    if len(y) < 2:
        return np.nan
    st = np.sum((y - y.mean())**2)
    return 1 - np.sum((y - yh)**2) / st if st > 0 else np.nan

def between_r2(df, y, p, gc):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g = s.groupby(gc)[[y, p]].mean()
    return r2(g[y], g[p])

def within_r2(df, y, p, gc):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c = s.groupby(gc).size()
    s = s[s[gc].isin(c[c >= 2].index)]
    if len(s) == 0:
        return np.nan
    gm = s.groupby(gc)[[y, p]].transform('mean')
    return r2(s[y] - gm[y], s[p] - gm[p])

def rmse(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    return np.sqrt(np.mean((np.asarray(y)[m] - np.asarray(yh)[m])**2))


def vsub(df, K=K_SAMP, N=N_PIX, seed=SEED):
    """Vectorized multinomial subsampling of region bin distributions."""
    rng = np.random.default_rng(seed)
    n = len(df)
    arr = df[bin_cols].values.reshape(n, 64, 8).astype(np.float64)
    arr = np.clip(np.nan_to_num(arr), 0, None)
    s = arr.sum(2, keepdims=True); s[s < 1e-8] = 1.0; arr /= s
    cum = np.cumsum(arr, axis=2)
    Y = df['yield'].values
    Xs, Ys = [], []
    for k in range(K):
        u = rng.random((n, 64, N))
        idx = (u[..., None] > cum[:, :, None, :]).sum(3)   # (n,64,N) bin index
        out = np.zeros((n, 64, 8), np.float32)
        for j in range(N):
            np.add.at(out, (np.arange(n)[:, None], np.arange(64)[None, :], idx[:, :, j]), 1.0/N)
        Xs.append(out.reshape(n, 512)); Ys.append(Y)
    return np.vstack(Xs), np.concatenate(Ys)


def fit_ensemble(tr_bin, tr_pct):
    mp = HistGradientBoostingRegressor(**cfg)
    mp.fit(tr_pct[pct_comb].fillna(0).values.astype(np.float32), tr_pct['yield'].values)
    aX, aY = vsub(tr_bin)
    mb = HistGradientBoostingRegressor(**cfg)
    mb.fit(aX.astype(np.float32), aY)
    return mb, mp

def predict_ensemble(mb, mp, X_bin, X_pct):
    pb = mb.predict(X_bin[bin_cols].fillna(0).values.astype(np.float32)).clip(0)
    pp = mp.predict(X_pct[pct_comb].fillna(0).values.astype(np.float32)).clip(0)
    return W_BIN * pb + (1 - W_BIN) * pp


# ============================================================
t0 = time.time()
print("Loading HarvestStat...")
ay = pd.read_parquet(os.path.join(hv_dir, "hvstat_africa_maize_adm2.parquet"))
fm = pd.read_parquet(os.path.join(hv_dir, "hvstat_africa_fnid_mapping.parquet"))
bh = pd.read_parquet(os.path.join(hv_dir, "aef_africa_adm2_binned_hist.parquet")
                     ).merge(fm[['fnid', 'admin1_id']], on='fnid', how='inner')
pc = pd.read_parquet(os.path.join(hv_dir, "aef_africa_adm2_hist.parquet")
                     ).merge(fm[['fnid', 'admin1_id']], on='fnid', how='inner')
mn = pd.read_parquet(os.path.join(hv_dir, "aef_africa_adm2_mean.parquet")
                     ).merge(fm[['fnid', 'admin1_id']], on='fnid', how='inner')
pcf = pc.merge(mn[['fnid', 'year'] + mean_cols], on=['fnid', 'year'], how='inner')
combo = bh.merge(pcf[['fnid', 'year', 'admin1_id'] + pct_comb],
                 on=['fnid', 'year', 'admin1_id'], how='inner')
tb = bh.merge(ay[['fnid', 'year', 'yield', 'admin1_id']],
              on=['fnid', 'year', 'admin1_id'], how='inner')
tp = pcf.merge(ay[['fnid', 'year', 'yield', 'admin1_id']],
               on=['fnid', 'year', 'admin1_id'], how='inner')


def loyo(tb_, tp_, combo_):
    yrs = sorted(set(tb_['year']) & set(tp_['year']))
    preds = []
    for hy in yrs:
        rb, rp = tb_[tb_['year'] != hy], tp_[tp_['year'] != hy]
        if len(rb) < 10:
            continue
        te = combo_[combo_['year'] == hy].copy()
        if len(te) == 0:
            continue
        mb, mp = fit_ensemble(rb, rp)
        te['pred'] = predict_ensemble(mb, mp, te, te)
        preds.append(te[['fnid', 'year', 'admin1_id', 'pred']])
    pr = pd.concat(preds, ignore_index=True)
    return ay.merge(pr, on=['fnid', 'year', 'admin1_id'], how='inner')


# ── Panel A: all-countries + Kenya-only LOYO ────────────
print("\nPanel A: all-countries LOYO...")
ed_all = loyo(tb, tp, combo)
n_ctry = fm.loc[fm['fnid'].isin(ed_all['fnid']), 'country_code'].nunique()
A_all = (len(ed_all), r2(ed_all['yield'], ed_all['pred']),
         between_r2(ed_all, 'yield', 'pred', 'admin1_id'), rmse(ed_all['yield'], ed_all['pred']))
print(f"  all ({n_ctry}): N={A_all[0]} R2={A_all[1]:.3f} Btw={A_all[2]:.3f} RMSE={A_all[3]:.3f}")

ke_fnid = set(fm[fm['country_code'] == 'KE']['fnid'])
tbk, tpk, ck = tb[tb['fnid'].isin(ke_fnid)], tp[tp['fnid'].isin(ke_fnid)], combo[combo['fnid'].isin(ke_fnid)]
print("Panel A/B: Kenya LOYO...")
ed_ke = loyo(tbk, tpk, ck)
A_ke = (len(ed_ke), r2(ed_ke['yield'], ed_ke['pred']), ed_ke['yield'].mean(), rmse(ed_ke['yield'], ed_ke['pred']))
print(f"  Kenya: N={A_ke[0]} R2={A_ke[1]:.3f} meanY={A_ke[2]:.2f} RMSE={A_ke[3]:.3f}")

# ── Panel B: Kenya LOYO by year ─────────────────────────
B_rows = []
for yr in sorted(ed_ke['year'].unique()):
    s = ed_ke[ed_ke['year'] == yr]
    if len(s) >= 10:
        B_rows.append((int(yr), len(s), r2(s['yield'], s['pred']), rmse(s['yield'], s['pred'])))
        print(f"  {yr}: N={len(s)} R2={B_rows[-1][2]:.3f} RMSE={B_rows[-1][3]:.3f}")

# ── Panels C/D: GROW plot-level ─────────────────────────
print("\nPanels C/D: train final Kenya model, predict GROW plots...")
mb_k, mp_k = fit_ensemble(tbk, tpk)

gp = pd.read_parquet(os.path.join(grow_dir, "grow_africa_maize_plots.parquet"))
gp = gp[gp['country'] == 'Kenya'].copy()
g_mean = pd.read_parquet(os.path.join(grow_dir, "aef_africa_plots_mean.parquet"))
g_hist = pd.read_parquet(os.path.join(grow_dir, "aef_africa_plots_hist.parquet"))
g_bin  = pd.read_parquet(os.path.join(grow_dir, "aef_africa_plots_binned.parquet"))
g_pct  = g_hist.merge(g_mean[['plot_id', 'year'] + mean_cols], on=['plot_id', 'year'], how='inner')

# map plot -> county name -> HarvestStat fnid
l1 = pd.read_csv(os.path.join(lut_dir, "Africa_L1_LUT.csv"))
l1 = l1[l1['Country'] == 'Kenya'][['L1_GID', 'admin1']].rename(columns={'admin1': 'county'})
gp = gp.merge(l1, left_on='gadm_l1', right_on='L1_GID', how='left')
ke_map = fm[fm['country_code'] == 'KE'][['fnid', 'admin_1']].rename(columns={'admin_1': 'county'})
gp = gp.merge(ke_map, on='county', how='left')
gp = gp.dropna(subset=['fnid'])
print(f"  GROW Kenya plots mapped to a HarvestStat county: {len(gp):,} / matched fnids {gp['fnid'].nunique()}")

# plot AEF features + predictions
gp = gp.merge(g_bin[['plot_id', 'year'] + bin_cols], on=['plot_id', 'year'], how='inner')
gp = gp.merge(g_pct[['plot_id', 'year'] + pct_comb], on=['plot_id', 'year'], how='inner')
gp['pred'] = predict_ensemble(mb_k, mp_k, gp, gp)

# HarvestStat county-year yields for correction + ratio
hv_ky = ay[ay['fnid'].isin(ke_fnid)][['fnid', 'year', 'yield']].rename(columns={'yield': 'hv_yield'})
gp = gp.merge(hv_ky, on=['fnid', 'year'], how='left')
# additive correction: shift plot preds within county-year to county-year HarvestStat mean
cy = gp.dropna(subset=['hv_yield']).groupby(['fnid', 'year']).agg(
    mean_pred=('pred', 'mean')).reset_index()
cy = cy.merge(hv_ky, on=['fnid', 'year'], how='left')
cy['shift'] = cy['hv_yield'] - cy['mean_pred']
gp = gp.merge(cy[['fnid', 'year', 'shift']], on=['fnid', 'year'], how='left')
gp['pred_corr'] = (gp['pred'] + gp['shift'].fillna(0.0)).clip(lower=0)
gp = gp.dropna(subset=['yield', 'pred_corr', 'hv_yield'])

# representativeness ratio per county = mean(plot yield)/mean(HarvestStat yield)
cr = gp.groupby('fnid').agg(plot_y=('yield', 'mean'), hv_y=('hv_yield', 'mean'),
                            county=('county', 'first')).reset_index()
cr['ratio'] = cr['plot_y'] / cr['hv_y']
gp = gp.merge(cr[['fnid', 'ratio']], on='fnid', how='left')

# Panel C: tiers
def tier(r):
    if r < 1.2:  return 0
    if r < 1.6:  return 1
    if r < 2.2:  return 2
    return 3
gp['tier'] = gp['ratio'].apply(tier)
tier_lab = ["Most repr.\\ (ratio $< 1.2$)", "Somewhat repr.\\ (ratio 1.2--1.6)",
            "Less repr.\\ (ratio 1.6--2.2)", "Unrepresentative (ratio $> 2.2$)"]
C_rows = []
for t in range(4):
    s = gp[gp['tier'] == t]
    if len(s) == 0:
        continue
    C_rows.append((tier_lab[t], len(s), r2(s['yield'], s['pred_corr']),
                   between_r2(s, 'yield', 'pred_corr', 'fnid'),
                   within_r2(s, 'yield', 'pred_corr', 'fnid'),
                   rmse(s['yield'], s['pred_corr']), s['yield'].mean()))
    print(f"  tier {t}: N={len(s)} R2={C_rows[-1][2]:.3f} ratio_mean={s['ratio'].mean():.2f}")

# Panel D: selected counties (within-R2 by year)
sel = ["Trans Nzoia", "Bungoma", "Uasin Gishu", "Kakamega", "Kericho"]
D_rows = []
for cty in sel:
    s = gp[gp['county'] == cty]
    if len(s) == 0:
        continue
    D_rows.append((cty, s['ratio'].iloc[0], len(s), r2(s['yield'], s['pred_corr']),
                   within_r2(s, 'yield', 'pred_corr', 'year'),
                   rmse(s['yield'], s['pred_corr']), s['yield'].mean()))
    print(f"  {cty}: N={len(s)} ratio={s['ratio'].iloc[0]:.2f} R2={D_rows[-1][3]:.3f}")


# ── Write LaTeX ─────────────────────────────────────────
def f3(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    return (f"{v:.3f}").replace("-", "$-$") if v < 0 else f"{v:.3f}"
def f2(v):
    return "---" if v is None or not np.isfinite(v) else f"{v:.2f}"

L = [r"\begin{table}[!ht]", r"\centering",
     r"\caption{AEF Hist Ensemble applied to East African maize yields: HarvestStat "
     r"training and GROW-Africa plot-level evaluation} \label{tab:africa_profile}",
     r"\footnotesize", r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r"& $N$ & $R^2$ & Btw-$R^2$ & Wtn-$R^2$ & RMSE & Mean yield \\", r"\midrule",
     r"\multicolumn{7}{l}{\textit{Panel A: Admin-2 level LOYO CV (HarvestStat training data)}} \\",
     f"All countries ({n_ctry}) & {A_all[0]:,} & {f3(A_all[1])} & {f3(A_all[2])} & --- & {f3(A_all[3])} & --- \\\\",
     f"Kenya only (47 counties) & {A_ke[0]:,} & {f3(A_ke[1])} & --- & --- & {f3(A_ke[3])} & {f2(A_ke[2])} \\\\",
     r"\addlinespace",
     r"\multicolumn{7}{l}{\textit{Panel B: Kenya admin-2 LOYO by year}} \\"]
for yr, n, rr, rm in B_rows:
    L.append(f"{yr} & {n} & {f3(rr)} & --- & --- & {f3(rm)} & --- \\\\")
L += [r"\addlinespace",
      r"\multicolumn{7}{l}{\textit{Panel C: Plot-level predictions by county representativeness (corrected)}} \\"]
for lab, n, rr, bt, wt, rm, my in C_rows:
    L.append(f"{lab} & {n:,} & {f3(rr)} & {f3(bt)} & {f3(wt)} & {f3(rm)} & {f2(my)} \\\\")
L += [r"\addlinespace",
      r"\multicolumn{7}{l}{\textit{Panel D: Selected Kenya counties (plot-level, corrected)}} \\"]
for cty, ra, n, rr, wt, rm, my in D_rows:
    L.append(f"{cty} (ratio {ra:.2f}) & {n:,} & {f3(rr)} & --- & {f3(wt)} & {f3(rm)} & {f2(my)} \\\\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\par\smallskip",
      r"\footnotesize{Notes: Panel A reports leave-one-year-out cross-validation at the admin-2 "
      r"level using HarvestStat Africa yields (2017--2024) and AEF Hist Ensemble ($w=0.5$ bins "
      r"$+0.6$ percentile, HistGradientBoosting). Panel B breaks down Kenya LOYO by held-out year. "
      r"Panels C--D evaluate the Kenya-trained model applied to GROW-Africa One Acre Fund plot-level "
      r"maize yields (2017--2020) using plot-level AEF features. ``Ratio'' is the mean ratio of "
      r"plot-level yield to HarvestStat admin-2 yield for each county. The additive correction shifts "
      r"plot predictions within each county-year to match the HarvestStat yield. Within-$R^2$ in "
      r"Panel D uses year as the grouping variable within each county.}",
      r"\end{table}", ""]

out = os.path.join(table_dir, "accuracy_africa_profile.tex")
with open(out, "w") as fh:
    fh.write("\n".join(L))
print(f"\nWrote {out}")
print(f"Runtime: {(time.time()-t0)/60:.1f} min")
