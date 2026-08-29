"""
Apply the within-municipality shrinkage post-processing to ALL models.

For each model's ADC-level 2022 predictions, scale each ADC's predicted
deviation-from-municipality-mean by lambda* = rho/r (chosen by leave-
municipalities-out CV), which maximizes the within-municipality R^2 at rho^2.
This is a costless post-processing applicable to any model. We report the
shrunk metrics alongside the raw ones so a "<Model> Shrink" row can be added
to the main accuracy tables (combined and spring-summer seasons).

Ground truth + AEF Hist Ens predictions come from adc_aef_hist_ens_eval.parquet;
the other models' ADC-level 2022 predictions are merged in by ADC id.

Run:  ~/miniforge3/envs/geo_env/bin/python within_mun_shrink_all_models.py
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

home_dir  = os.path.expanduser("~")
proj_dir  = os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
pred_dir  = os.path.join(proj_dir, "Data", "predictions")
table_dir = os.path.join(proj_dir, "tables")
os.makedirs(table_dir, exist_ok=True)

N_FOLDS = 5

# ── metric helpers (match gb_aef_hist_ensemble.py) ──────
def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh)
    y, yh = np.asarray(y)[m], np.asarray(yh)[m]
    if len(y) < 2: return np.nan
    st = np.sum((y - y.mean())**2)
    return 1 - np.sum((y - yh)**2)/st if st > 0 else np.nan

def within_r2(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c = s.groupby(gc).size(); s = s[s[gc].isin(c[c >= 2].index)]
    if len(s) == 0: return np.nan
    gm = s.groupby(gc)[[y, p]].transform("mean")
    return r2(s[y] - gm[y], s[p] - gm[p])

def between_r2(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g = s.groupby(gc)[[y, p]].mean(); return r2(g[y], g[p])

def metrics(df, y, p):
    s = df[[y, p, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
    return dict(N=len(s), R2=r2(s[y], s[p]), Btw=between_r2(s, y, p),
                Wtn=within_r2(s, y, p),
                RMSE=np.sqrt(np.mean((s[y].values - s[p].values)**2)))

def cv_lambda(df, pcol, ycol, gc="muncode"):
    """Leave-municipalities-out CV lambda* = rho/r (pooled per-fold)."""
    s = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    cnt = s.groupby(gc)[ycol].transform("size"); s = s[cnt >= 2].reset_index(drop=True)
    if s[gc].nunique() < N_FOLDS: return np.nan
    gkf = GroupKFold(n_splits=N_FOLDS); lams = []
    for tr, _ in gkf.split(s, groups=s[gc]):
        d = s.iloc[tr]
        a = (d[ycol] - d.groupby(gc)[ycol].transform("mean")).values
        b = (d[pcol] - d.groupby(gc)[pcol].transform("mean")).values
        denom = np.sqrt(np.sum(a*a)*np.sum(b*b))
        if denom <= 0: continue
        rho = np.sum(a*b)/denom; rr = np.sqrt(np.sum(b*b)/np.sum(a*a))
        if rr > 0: lams.append(rho/rr)
    return float(np.clip(np.mean(lams), 0, 1)) if lams else np.nan

def apply_shrink(df, pcol, lam, gc="muncode"):
    g = df.groupby(gc)[pcol]
    return g.transform("mean") + lam * (df[pcol] - g.transform("mean"))


# ── load GT + merge every model ─────────────────────────
gt = pd.read_parquet(os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet"))
gt = gt[["adc", "muncode", "yield", "yield_pv", "pred"]].rename(columns={"pred": "AEF Hist Ens."})

MODELS = [  # (label, file, col)
    ("RS",            "adc_yield_preds_corrected_2022.csv", "yield_pred_ls"),
    ("AEF mean",         "adc_alpha_earth_preds.csv",          "yield_pred"),
    ("Agg-NN",        "adc_mlp_yield_preds.csv",            "pred_yield"),
    ("AEF Hist",      "adc_aef_hist_gb_preds.parquet",      "yield_pred"),
    ("3-Period Hist.","adc_3period_hist_gb_preds.parquet",  "yield_pred"),
]
for label, f, col in MODELS:
    df = pd.read_parquet(os.path.join(pred_dir, f)) if f.endswith("parquet") else pd.read_csv(os.path.join(pred_dir, f))
    if "year" in df.columns: df = df[df["year"] == 2022]
    key = "adc" if "adc" in df.columns else "adcid"
    df["adc"] = df[key].astype(str).str.replace("-", "", regex=False)
    df = df[["adc", col]].rename(columns={col: label}).dropna().drop_duplicates("adc")
    gt = gt.merge(df, on="adc", how="left")

ORDER = ["RS", "3-Period Hist.", "AEF mean", "Agg-NN", "AEF Hist", "AEF Hist Ens."]

# ── compute raw + shrink per model, both seasons ────────
def run(ycol, season):
    print(f"\n=== {season} (target {ycol}) ===")
    print(f"  {'Model':<16s} {'N':>7s} {'R2':>6s} {'Btw':>6s} {'Wtn':>6s} {'RMSE':>6s}  lam")
    rows = []
    for label in ORDER:
        sub = gt[["muncode", ycol, label]].dropna().rename(columns={label: "p"})
        if len(sub) < 50: continue
        lam = cv_lambda(sub, "p", ycol)
        sub["ps"] = apply_shrink(sub, "p", lam)
        mr = metrics(sub, ycol, "p"); ms = metrics(sub, ycol, "ps")
        print(f"  {label+' Raw':<16s} {mr['N']:>7,} {mr['R2']:>6.3f} {mr['Btw']:>6.3f} {mr['Wtn']:>6.3f} {mr['RMSE']:>6.3f}")
        print(f"  {label+' Shr':<16s} {ms['N']:>7,} {ms['R2']:>6.3f} {ms['Btw']:>6.3f} {ms['Wtn']:>6.3f} {ms['RMSE']:>6.3f}  {lam:.2f}")
        rows.append((label, lam, mr, ms))
    return rows

rows_comb = run("yield", "Combined season")
rows_pv   = run("yield_pv", "Spring-summer (P-V)")

# ── write a supplementary tex with the per-model Shrink rows ──
def fmt(v):
    if v is None or not np.isfinite(v): return "---"
    return (f"{v:.3f}").replace("-", "$-$") if v < 0 else f"{v:.3f}"

def block(rows, title):
    L = [rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\"]
    for label, lam, mr, ms in rows:
        L.append(f"{label} Shrink & {ms['N']:,} & {fmt(ms['R2'])} & {fmt(ms['Btw'])} & {fmt(ms['Wtn'])} & {fmt(ms['RMSE'])} \\\\")
    return L

tex = [r"\begin{table}[!ht]", r"\centering",
       r"\caption{Within-municipality shrinkage applied to all models (maize, ADC level, 2022). "
       r"Each model's predicted within-municipality deviations are scaled by a leave-municipalities-out "
       r"cross-validated $\lambda$, which maximizes within-$R^2$ at no data or training cost.} "
       r"\label{tab:shrink_all}", r"\footnotesize", r"\begin{tabular}{lrrrrr}", r"\toprule",
       r"Model & $N$ & $R^2$ & Btw-$R^2$ & Wtn-$R^2$ & RMSE \\", r"\midrule"]
tex += block(rows_comb, "Combined season")
tex += [r"\addlinespace"] + block(rows_pv, "Spring-summer (P-V) season")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out = os.path.join(table_dir, "accuracy_shrink_all_models_2022.tex")
with open(out, "w") as fh: fh.write("\n".join(tex))
print(f"\nWrote {out}")
