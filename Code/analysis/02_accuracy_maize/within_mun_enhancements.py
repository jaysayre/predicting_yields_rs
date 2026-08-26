"""
Within-municipality enhancements for ADC-level maize yield predictions.

Two levers on the AEF Hist Ensemble's within-municipality R² (baseline 0.152):

  1. SHRINKAGE (deployable, free).  within-R2 = 2*rho*r - r^2, where
     rho = within-mun Pearson(pred_dev, obs_dev) and r = SD(pred_dev)/SD(obs_dev).
     The current model has r (0.71) > rho (0.46): predicted within-mun deviations
     are OVER-spread for their accuracy.  Scaling each ADC's deviation-from-mun-mean
     by lambda* = rho/r maximizes within-R2 at rho^2.  Pure post-processing: no
     retrain, no new GEE pull, no ADC-level labels.  lambda is chosen by grouped
     CV (leave-muns-out) to avoid in-sample optimism.

  2. TWO-STAGE RESIDUAL CEILING (oracle diagnostic, NOT deployable).  Trains a
     HistGB on within-mun AEF deviations (ADC embedding - mun mean) to predict the
     within-mun yield deviation, scored out-of-fold with GroupKFold(muncode).
     This REQUIRES ADC-level ground-truth labels in training, so it violates the
     "train on aggregate only" premise -- it is reported as an upper bound on how
     much within-mun signal the AEF embeddings carry at all.  If even the oracle
     cannot beat rho^2 by much, the bottleneck is the embeddings, not the training
     scheme.

Outputs (under ~/Dropbox/Projects/Maize_prediction/):
  tables/accuracy_within_enhancements_2022.tex   -- supplementary table
  plots/coauthor_extras_paper/fig_within_shrinkage_curve.png  (regenerated)

Run:  ~/miniforge3/envs/geo_env/bin/python within_mun_enhancements.py
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
aef_dir    =  os.path.join(proj_dir, "Data", "alpha_earth")
pred_dir   =  os.path.join(proj_dir, "Data", "predictions")
agland_path=  os.path.join(proj_dir, "Data", "SIAP_agland", "Output",
                           "2007_adcs_agland_area.csv")
table_dir  =  os.path.join(proj_dir, "tables")
plot_dir   =  os.path.join(proj_dir, "plots", "coauthor_extras_paper")
os.makedirs(table_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ── Inputs ───────────────────────────────────────────────
eval_path  =  os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet")  # adc,muncode,yield,yield_pv,pred,pred_corr
adc_emb_path= os.path.join(aef_dir, "alpha_earth_mex_adcs.parquet")    # adcid,year,A00..A63

mean_cols  =  [f"A{d:02d}" for d in range(64)]
N_FOLDS    =  5
SEED       =  42

cfg = dict(max_iter=600, max_depth=6, learning_rate=0.05,
           min_samples_leaf=20, random_state=SEED, early_stopping=False)


# ── Metric helpers (match gb_aef_hist_ensemble.py) ───────
def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = np.asarray(y)[m], np.asarray(yh)[m]
    if len(y) < 2:
        return np.nan
    st = np.sum((y - y.mean())**2)
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

def metrics(df, ycol, pcol):
    sub = df[[ycol, pcol, 'muncode']].replace([np.inf, -np.inf], np.nan).dropna()
    rmse = np.sqrt(np.mean((sub[ycol].values - sub[pcol].values)**2))
    return dict(N=len(sub), R2=r2(sub[ycol], sub[pcol]),
                Btw=between_r2(sub, ycol, pcol), Wtn=within_r2(sub, ycol, pcol),
                RMSE=rmse)


def mun_demean(df, col, gc='muncode'):
    """Within-mun deviation of `col` (NaN-safe; only muns with >=2 valid rows)."""
    g = df.groupby(gc)[col]
    return df[col] - g.transform('mean')


# ── Cross-validated shrinkage factor lambda* = rho/r ─────
def cv_lambda(df, pcol, ycol, gc='muncode'):
    """Grouped CV: estimate lambda on train muns, score on held-out muns,
    return the lambda maximizing pooled out-of-fold within-R2."""
    sub = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    cnt = sub.groupby(gc)[ycol].transform('size')
    sub = sub[cnt >= 2].reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    lam_grid = np.linspace(0.0, 1.4, 57)
    # accumulate OOF deviations under each candidate lambda
    oof_a, oof_b = [], []
    for tr, te in gkf.split(sub, groups=sub[gc]):
        trd, ted = sub.iloc[tr], sub.iloc[te]
        # lambda from TRAIN fold
        a = (trd[ycol] - trd.groupby(gc)[ycol].transform('mean')).values
        b = (trd[pcol] - trd.groupby(gc)[pcol].transform('mean')).values
        rho = np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))
        rr  = np.sqrt(np.sum(b*b)/np.sum(a*a))
        lam = rho/rr
        # apply to TEST fold deviations
        at = (ted[ycol] - ted.groupby(gc)[ycol].transform('mean')).values
        bt = (ted[pcol] - ted.groupby(gc)[pcol].transform('mean')).values
        oof_a.append((at, bt, lam))
    # pooled OOF within-R2 at the CV-chosen lambda (per-fold lambda applied)
    A  = np.concatenate([x[0] for x in oof_a])
    B  = np.concatenate([x[1]*x[2] for x in oof_a])
    wr2_cv = 1 - np.sum((A - B)**2)/np.sum(A*A)
    lam_mean = float(np.mean([x[2] for x in oof_a]))
    return lam_mean, wr2_cv


def apply_shrink(df, pcol, lam, gc='muncode'):
    g = df.groupby(gc)[pcol]
    return g.transform('mean') + lam * (df[pcol] - g.transform('mean'))


# ============================================================
print("Loading eval frame + ADC embeddings...")
ev = pd.read_parquet(eval_path)
emb = pd.read_parquet(adc_emb_path)
emb = emb[emb['year'] == 2022].copy()
emb['adc'] = emb['adcid'].str.replace('-', '', regex=False)
emb = emb.drop_duplicates('adc')

d = ev.merge(emb[['adc'] + mean_cols], on='adc', how='left')

# agland → irrigation share + log area (within-mun deviation features)
ag = pd.read_csv(agland_path)
ag['adc'] = ag['adc07'].str.replace('-', '', regex=False)
ag['irrig_share'] = (ag['siap_irrig_area'] / ag['siap_agland_area']
                     ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
ag['log_adc_area'] = np.log1p(ag['adc_area'].clip(lower=0))
d = d.merge(ag[['adc', 'irrig_share', 'log_adc_area']], on='adc', how='left')

# ----- combined-season eval set (has pred & yield) -----
print("\n=== SHRINKAGE (combined season) ===")
lam_raw,  wr2_raw_cv  = cv_lambda(d, 'pred',      'yield')
lam_corr, wr2_corr_cv = cv_lambda(d, 'pred_corr', 'yield')
print(f"  raw : CV lambda*={lam_raw:.3f}  OOF within-R2={wr2_raw_cv:.3f}")
print(f"  corr: CV lambda*={lam_corr:.3f}  OOF within-R2={wr2_corr_cv:.3f}")

d['pred_shrink']      = apply_shrink(d, 'pred',      lam_raw)
d['pred_corr_shrink'] = apply_shrink(d, 'pred_corr', lam_corr)


# ----- two-stage residual ceiling (oracle, GroupKFold) -----
print("\n=== TWO-STAGE RESIDUAL CEILING (oracle; needs ADC labels) ===")
dev_cols = [f"{c}_dev" for c in mean_cols]
# only rows with embeddings + yield, muns with >=2 ADCs
m = d['pred'].notna() & d['yield'].notna() & d['A00'].notna()
ds = d[m].copy()
cnt = ds.groupby('muncode')['yield'].transform('size')
ds = ds[cnt >= 2].reset_index(drop=True)

for c in mean_cols:
    ds[f"{c}_dev"] = mun_demean(ds, c)
ds['irrig_dev'] = mun_demean(ds, 'irrig_share')
ds['area_dev']  = mun_demean(ds, 'log_adc_area')
ds['y_dev']     = mun_demean(ds, 'yield')

feat = dev_cols + ['irrig_dev', 'area_dev']
X = ds[feat].fillna(0.0).values.astype(np.float32)
y = ds['y_dev'].values.astype(np.float32)

gkf = GroupKFold(n_splits=N_FOLDS)
oof = np.zeros(len(ds))
for k, (tr, te) in enumerate(gkf.split(X, groups=ds['muncode'])):
    mdl = HistGradientBoostingRegressor(**cfg)
    mdl.fit(X[tr], y[tr])
    oof[te] = mdl.predict(X[te])
    print(f"  fold {k+1}/{N_FOLDS} done")
ds['resid_dev'] = oof

# combined model: ensemble mun-mean (between) + oracle residual dev (within)
ds['pred_2stage'] = ds.groupby('muncode')['pred'].transform('mean') + ds['resid_dev']
# corrected variant: SIAP-anchored mun mean (= pred_corr mun mean) + residual dev
ds['pred_corr_2stage'] = ds.groupby('muncode')['pred_corr'].transform('mean') + ds['resid_dev']
# + shrink the oracle residual too
lam2, _ = cv_lambda(ds.assign(_p=ds['pred_2stage']), '_p', 'yield')
ds['pred_2stage_shrink'] = (ds.groupby('muncode')['pred'].transform('mean')
                            + lam2 * ds['resid_dev'])

# raw within-R2 of the oracle residual alone (ceiling on rho^2 achievable)
rho_oracle = np.corrcoef(ds['y_dev'], ds['resid_dev'])[0, 1]
print(f"  oracle within rho = {rho_oracle:.3f}  (ceiling within-R2 ~ rho^2 = {rho_oracle**2:.3f})")


# ── Assemble results table (combined + P-V) ─────────────
def row(label, df, pcol, ycol):
    m = metrics(df, ycol, pcol)
    return (label, m['N'], m['R2'], m['Btw'], m['Wtn'], m['RMSE'])

rows_comb = [
    row("AEF Hist Ens.\\ (baseline)",        d,  'pred',              'yield'),
    row(f"\\quad + within-mun shrink ($\\lambda$={lam_raw:.2f})", d, 'pred_shrink', 'yield'),
    row("AEF Hist Ens.\\ Corr.",             d,  'pred_corr',         'yield'),
    row(f"\\quad + within-mun shrink ($\\lambda$={lam_corr:.2f})", d, 'pred_corr_shrink', 'yield'),
    row("Two-stage residual (oracle)",       ds, 'pred_2stage',       'yield'),
    row(f"\\quad + shrink ($\\lambda$={lam2:.2f})", ds, 'pred_2stage_shrink', 'yield'),
]

# P-V season
dpv = d[d['yield_pv'].notna()].copy()
lam_pv, _ = cv_lambda(dpv, 'pred', 'yield_pv')
dpv['pred_shrink'] = apply_shrink(dpv, 'pred', lam_pv)
rows_pv = [
    row("AEF Hist Ens.\\ (baseline)", dpv, 'pred', 'yield_pv'),
    row(f"\\quad + within-mun shrink ($\\lambda$={lam_pv:.2f})", dpv, 'pred_shrink', 'yield_pv'),
]

print("\n  {:<42s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}".format(
    "Model", "N", "R2", "Btw", "Wtn", "RMSE"))
for lab, n, ov, b, w, rm in rows_comb + rows_pv:
    print(f"  {lab:<42s} {n:>7,} {ov:>7.3f} {b:>7.3f} {w:>7.3f} {rm:>7.3f}")


# ── Write LaTeX table ───────────────────────────────────
def fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    s = f"{v:.3f}"
    return s.replace("-", "$-$") if v < 0 else s

lines = [
    r"\begin{table}[!ht]", r"\centering",
    r"\caption{Within-municipality enhancements to the AEF Hist Ensemble (maize, ADC level, 2022). "
    r"Shrinkage rescales each ADC's predicted deviation from its municipality mean by $\lambda$ "
    r"(chosen by leave-municipalities-out CV); it requires no retraining or ADC-level labels. "
    r"The two-stage residual model is an \emph{oracle} upper bound that uses ADC-level census "
    r"labels in cross-validated training, so it is not deployable under aggregate-only training.} "
    r"\label{tab:within_enhancements}",
    r"\footnotesize",
    r"\begin{tabular}{lrrrrr}", r"\toprule",
    r"Model & $N$ & $R^2$ & Btw-$R^2$ & Wtn-$R^2$ & RMSE \\", r"\midrule",
    r"\multicolumn{6}{l}{\textit{Combined season}} \\",
]
for lab, n, ov, b, w, rm in rows_comb:
    lines.append(f"{lab} & {n:,} & {fmt(ov)} & {fmt(b)} & {fmt(w)} & {fmt(rm)} \\\\")
lines.append(r"\addlinespace")
lines.append(r"\multicolumn{6}{l}{\textit{Spring-summer (P-V) season}} \\")
for lab, n, ov, b, w, rm in rows_pv:
    lines.append(f"{lab} & {n:,} & {fmt(ov)} & {fmt(b)} & {fmt(w)} & {fmt(rm)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

out_tex = os.path.join(table_dir, "accuracy_within_enhancements_2022.tex")
with open(out_tex, "w") as f:
    f.write("\n".join(lines))
print(f"\nWrote {out_tex}")


# ── Shrinkage curve figure (regenerate) ─────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sub = d[['yield', 'pred', 'muncode']].dropna().copy()
cnt = sub.groupby('muncode')['yield'].transform('size')
sub = sub[cnt >= 2]
a = (sub['yield'] - sub.groupby('muncode')['yield'].transform('mean')).values
b = (sub['pred']  - sub.groupby('muncode')['pred'].transform('mean')).values
SSa = np.sum(a*a)
rho = np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))
rr  = np.sqrt(np.sum(b*b)/np.sum(a*a))
lams = np.linspace(0, 1.4, 141)
w = [1 - np.sum((a - l*b)**2)/SSa for l in lams]
lstar = rho/rr
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(lams, w, color="#4A4A4A", lw=2)
ax.axvline(1.0, ls=":", color="#999", lw=1)
ax.axvline(lstar, ls="--", color="#B22", lw=1.3)
ax.scatter([1.0], [1 - np.sum((a-b)**2)/SSa], color="#999", zorder=5)
ax.scatter([lstar], [rho**2], color="#B22", zorder=5)
ax.annotate(f"current  $\\lambda$=1\nwithin-$R^2$={1-np.sum((a-b)**2)/SSa:.3f}",
            (1.0, 1-np.sum((a-b)**2)/SSa), xytext=(1.03, 0.06), fontsize=10, color="#555")
ax.annotate(f"optimal  $\\lambda$*={lstar:.2f}\nwithin-$R^2$=$\\rho^2$={rho**2:.3f}",
            (lstar, rho**2), xytext=(0.04, 0.18), fontsize=10, color="#B22")
ax.set_xlabel(r"within-municipality shrinkage factor  $\lambda$")
ax.set_ylabel(r"within-municipality $R^2$")
ax.set_title("Free within-$R^2$ gain from shrinking predicted deviations", fontsize=12)
ax.grid(True, ls=(0, (1, 3)), lw=.6, color="#D0D0D0"); ax.set_axisbelow(True)
fig.tight_layout()
out_fig = os.path.join(plot_dir, "fig_within_shrinkage_curve.png")
fig.savefig(out_fig, dpi=150)
print(f"Wrote {out_fig}")
print("\nDone.")
