"""
Unified generator for the combined- and spring-summer-season ADC-level accuracy
tables (Tables \ref{tab:accuracy_combined}, \ref{tab:accuracy_spring_summer}).

Produces, for every model (grouped Landsat-derived vs AEF-derived), three rows:
  Raw      - uncorrected ADC predictions
  Corr.    - additive ex-post correction toward the DGSIAP municipal yield, using
             EX-ANTE agricultural-land weights (siap_agland_area) to form the
             predicted municipal mean -- NOT census planted area (land_input)
  Shrink   - within-municipality shrinkage of the raw deviations (CV lambda)
plus the DGSIAP municipal-average benchmark. Metrics are computed against the
INEGI 2022 census at the ADC level (combined = yield, spring-summer = yield_pv).

This consolidates the previously hand-assembled tables into one reproducible
script and makes every model's correction use the same ex-ante weighting.

Run:  ~/miniforge3/envs/geo_env/bin/python 4_accuracy_main_2022.py
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold

home = os.path.expanduser("~")
proj = os.path.join(home, "Dropbox", "Projects", "Maize_prediction")
P    = os.path.join(proj, "Data", "predictions")
plot_dir = os.path.join(proj, "plots")
os.makedirs(plot_dir, exist_ok=True)

def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh); y, yh = np.asarray(y)[m], np.asarray(yh)[m]
    return 1 - np.sum((y - yh)**2)/np.sum((y - y.mean())**2) if m.sum() > 1 else np.nan
def within_r2(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c = s.groupby(gc).size(); s = s[s[gc].isin(c[c >= 2].index)]
    if len(s) == 0: return np.nan
    gm = s.groupby(gc)[[y, p]].transform("mean"); return r2(s[y]-gm[y], s[p]-gm[p])
def between_r2(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g = s.groupby(gc)[[y, p]].mean(); return r2(g[y], g[p])
def met(df, ycol, pcol):
    s = df[[ycol, pcol, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
    return (len(s), r2(s[ycol], s[pcol]), between_r2(s, ycol, pcol),
            within_r2(s, ycol, pcol), np.sqrt(np.mean((s[ycol].values-s[pcol].values)**2)))
LAM_DEPLOY = 0.72   # deployed shrinkage factor: public irrigation-projection point estimate (Sec 3.6)
def cv_lambda(df, pcol, ycol, gc="muncode"):
    s = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    cnt = s.groupby(gc)[ycol].transform("size"); s = s[cnt >= 2].reset_index(drop=True)
    if s[gc].nunique() < 5: return np.nan
    lams = []
    for tr, _ in GroupKFold(5).split(s, groups=s[gc]):
        d = s.iloc[tr]
        a = (d[ycol]-d.groupby(gc)[ycol].transform("mean")).values
        b = (d[pcol]-d.groupby(gc)[pcol].transform("mean")).values
        den = np.sqrt(np.sum(a*a)*np.sum(b*b))
        if den > 0:
            rho = np.sum(a*b)/den; rr = np.sqrt(np.sum(b*b)/np.sum(a*a))
            if rr > 0: lams.append(rho/rr)
    return float(np.clip(np.mean(lams), 0, 1)) if lams else np.nan
def shrink(df, pcol, lam, gc="muncode"):
    g = df.groupby(gc)[pcol]; return g.transform("mean") + lam*(df[pcol]-g.transform("mean"))
def boot_ci(df, ycol, pcol, nrep=1000, seed=42, gc="muncode"):
    """Municipality-cluster bootstrap 95% CI for (overall R2, within R2)."""
    sb_ = df[[ycol, pcol, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sb_) < 100: return None
    gm = sb_.groupby(gc)[[ycol, pcol]].transform("mean")
    d = pd.DataFrame({"m": sb_[gc].values,
        "sse": (sb_[ycol]-sb_[pcol])**2, "y": sb_[ycol], "y2": sb_[ycol]**2,
        "wn": ((sb_[pcol]-gm[pcol])-(sb_[ycol]-gm[ycol]))**2,
        "wd": (sb_[ycol]-gm[ycol])**2})
    g = d.groupby("m").agg(n=("y", "size"), sse=("sse", "sum"), sy=("y", "sum"),
                           sy2=("y2", "sum"), wn=("wn", "sum"), wd=("wd", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(nrep, len(g)))
    n = g["n"].values[idx].sum(1); sse = g["sse"].values[idx].sum(1)
    sy = g["sy"].values[idx].sum(1); sy2 = g["sy2"].values[idx].sum(1)
    ov = 1 - sse/(sy2 - sy**2/n)
    wt = 1 - g["wn"].values[idx].sum(1)/g["wd"].values[idx].sum(1)
    pc = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return pc(ov), pc(wt)
def correct(df, pcol, gc="muncode"):
    d = df[df[pcol].notna()].copy(); d["wv"] = d[pcol]*d["corr_w"]
    a = d.groupby(gc).agg(wv=("wv","sum"), w=("corr_w","sum"), siap=("siap","first")).reset_index()
    a["diff"] = a["wv"]/a["w"] - a["siap"]
    out = df.merge(a[[gc,"diff"]], on=gc, how="left")
    return (out[pcol] - out["diff"]).clip(lower=0)

# ── data ────────────────────────────────────────────────
ev = pd.read_parquet(os.path.join(P, "adc_aef_hist_ens_eval.parquet"))
ev = ev[["adc","muncode","yield","yield_pv","land_input","pred"]].rename(columns={"pred":"AEF Hist Ens."})
# Main tables are evaluated on the ensemble-eligible sample only (2026-08-28):
# ADCs with zero ESA WorldCover cropland pixels have all-null masked features,
# so other models' "coverage" there is nominal (AEF Hist R2 = -0.045 on them).
ev = ev[ev["AEF Hist Ens."].notna()].copy()
# fall-winter (O-I) ground truth
ca2022_dir = os.path.join(proj, "Data", "INEGI", "MD_lab_outputs",
                          "LM2304-CA22-2025-09-29-superficie_ENTREGA")
ca_szn = pd.read_stata(os.path.join(ca2022_dir, "adc_land_szn_ca22_adc07.dta"))
oi = ca_szn[(ca_szn["name"] == "Maize") & (ca_szn["type"] == "o-i")][["adc", "yield"]].rename(columns={"yield": "yield_oi"})
ev = ev.merge(oi, on="adc", how="left")
ag = pd.read_csv(os.path.join(proj,"Data","SIAP_agland","Output","2007_adcs_agland_area.csv"))
ag["adc"] = ag["adcid"].astype(str).str.replace("-","",regex=False)
ev = ev.merge(ag[["adc","siap_agland_area"]], on="adc", how="left")
ev["corr_w"] = np.where(ev["siap_agland_area"] > 0, ev["siap_agland_area"], ev["land_input"])
siap = pd.read_stata(os.path.join(home,"Dropbox/Projects/Maize_prediction/Data/SIAP/Cleaned/siap_ag_prod_estimation_by_season.dta"))
siap["muncode"] = siap["muncode"].apply(lambda x: str(int(x)).zfill(5))
s22 = siap[(siap["name"]=="Maize")&(siap["year"]==2022)]; s22 = s22[~s22["muncode"].str.endswith("000")]

# ── Season-matched DGSIAP municipal anchors (fixed 2026-08-15) ──────────────
# The anchor must match the census target being scored. Previously ONE
# all-seasons anchor was used for every table, so the P-V and O-I tables
# corrected against a combined-season municipal mean -- a season mismatch that
# removes a bias defined on a different quantity than the one being scored.
# It was worth ~+0.08 R2 on the P-V corrected row (0.403 true -> 0.481), in the
# direction that flatters the model. The anchor also defines the DGSIAP benchmark
# row and the common-sample intersection, so all three are now season-matched.
#   combined      -> all seasons summed   (unchanged; correct for `yield`)
#   spring_summer -> Spring-Summer only   (matches `yield_pv`)
#   fall_winter   -> Fall-Winter only     (matches `yield_oi`)
def _anchor(sub):
    g = sub.groupby("muncode").agg(q=("q","sum"), ha=("ha_planted","sum")).reset_index()
    g["siap"] = g["q"]/g["ha"]
    return g.set_index("muncode")["siap"]

ANCHORS = {"combined":      _anchor(s22),
           "spring_summer": _anchor(s22[s22["growing_season"]=="Spring-Summer"]),
           "fall_winter":   _anchor(s22[s22["growing_season"]=="Fall-Winter"])}
ANCHOR_NOTE = {"combined":      "all growing seasons",
               "spring_summer": "the spring-summer season only",
               "fall_winter":   "the fall-winter season only"}
for _t, _a in ANCHORS.items():
    print(f"  anchor {_t:14s}: {len(_a):,} municipalities")

def set_anchor(tag):
    """Point ev['siap'] at the season-matched municipal anchor."""
    ev["siap"] = ev["muncode"].map(ANCHORS[tag])

ev["siap"] = ev["muncode"].map(ANCHORS["combined"])   # default

# Harmonic NDVI models (ls_harmonic_features extraction, 2017-2024 muni-trained;
# see harmonic_adc_eval.py). Replaces the old RS CNN + 3-period hist rows.
# Only the 3-period-window pair is shown; the 2-period and raw-coefficient
# variants underperform (see harmonic_adc_eval_summary.csv) and are omitted.
LANDSAT = [("NDVI","adc_aefn2_masked_preds.parquet","pred")]  # aefn2 masked baseline (replaces h3 NDVI Hist/Q-Hist)
# "AEF mean" = train/03_train_rf_gb/1_rf_yield_prediction.py (2026-09-03: IMPROVED=False, RF on the 64 means alone).
# Until 2026-09-02 this row read adc_alpha_earth_preds.csv, an untracked Nov-2025 RF run with no producer.
AEFM    = [("AEF mean","adc_alpha_earth_preds_maize.parquet","yield_pred"),
           ("Agg-NN","adc_mlp_yield_preds.csv","pred_yield"),
           ("AEF Hist","adc_aef_hist_bins_gb_preds.parquet","yield_pred"),
           ("AEF Hist Ens.", None, None)]  # already in eval frame
for nm,f,c in LANDSAT+AEFM:
    if f is None: continue
    d = pd.read_parquet(os.path.join(P,f)) if f.endswith("parquet") else pd.read_csv(os.path.join(P,f))
    if "year" in d.columns: d = d[d["year"]==2022]
    k = "adc" if "adc" in d.columns else "adcid"; d["adc"] = d[k].astype(str).str.replace("-","",regex=False)
    ev = ev.merge(d[["adc",c]].rename(columns={c:nm}).dropna().drop_duplicates("adc"), on="adc", how="left")

def fmt(v, neg_math=True):
    if v is None or not np.isfinite(v): return "---"
    s = f"{v:.3f}"
    return s.replace("-", "$-$") if (neg_math and v < 0) else s

def model_rows(label, season_y):
    """Return (raw, corr, shrink) metric tuples for one model + season."""
    rows = []
    raw = met(ev, season_y, label); rows.append((f"{label} Raw", raw, boot_ci(ev, season_y, label)))
    ev["_c"] = correct(ev, label); cr = met(ev, season_y, "_c"); rows.append((f"{label} Corr.", cr, boot_ci(ev, season_y, "_c")))
    lam = LAM_DEPLOY; ev["_s"] = shrink(ev, label, lam)
    sh = met(ev, season_y, "_s"); rows.append((f"{label} Shrink", sh, boot_ci(ev, season_y, "_s")))
    return rows

def build_table(season_y, label_season, fname, tag, show_ci=True):
    set_anchor(tag)                      # season-matched DGSIAP anchor
    orc_season = {"combined": "combined", "spring_summer": "spring_summer"}.get(tag)
    ci_note = (r" Municipality-cluster bootstrap 95\% confidence intervals for $R^2$ "
               r"and Within $R^2$ in brackets.") if show_ci else ""
    orc_note = (r" The Oracle (ADC-trained) row is a non-deployable upper bound that trains "
                r"HistGradientBoosting directly on ADC-level census labels (5-fold GroupKFold "
                r"over municipalities); it bounds how much yield signal the embeddings carry."
                ) if orc_season else ""
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Accuracy metrics for maize yield predictions vs.\ INEGI 2022 census, "
         rf"ADC level --- {label_season}. Corrected rows use ex-ante agricultural-land "
         rf"weights for the municipal anchor, and the anchor is the DGSIAP municipal "
         rf"maize yield for {ANCHOR_NOTE[tag]}, matching the census target scored "
         rf"here. All rows are evaluated on the ADCs for which the AEF Hist "
         rf"Ensemble is defined (at least one cropland pixel; see the sample "
         rf"accounting in the appendix).{ci_note} RMSE in t/ha.{orc_note}}}",
         rf"\label{{tab:accuracy_{tag}}}", r"\footnotesize", r"\begin{tabular}{lrrrrr}", r"\hline",
         r"Model & $N$ & $R^2$ & Between $R^2$ & Within $R^2$ & RMSE \\", r"\hline",
         r"\multicolumn{6}{l}{\textit{Landsat-derived features}} \\"]
    def emit(group):
        for nm,_,_ in group:
            for tagn, m, ci in model_rows(nm, season_y):
                n, ov, bt, wt, rm = m
                # Uniform inter-word spacing after abbreviation periods ("Hist.", "Ens.")
                # so a model's Raw/Corr./Shrink rows are typeset identically.
                lab = tagn.replace("Hist. Raw", "Hist.\\ Raw").replace("Hist. Corr.", "Hist.\\ Corr.").replace("Hist. Shrink", "Hist.\\ Shrink")
                lab = lab.replace("Ens. Raw", "Ens.\\ Raw").replace("Ens. Corr.", "Ens.\\ Corr.").replace("Ens. Shrink", "Ens.\\ Shrink")
                L.append(f"{lab} & {n:,} & {fmt(ov)} & {fmt(bt)} & {fmt(wt)} & {fmt(rm)} \\\\")
                if ci and show_ci:
                    (olo, ohi), (wlo, whi) = ci
                    L.append(f" & & \\scriptsize[{fmt(olo)}, {fmt(ohi)}] & & "
                             f"\\scriptsize[{fmt(wlo)}, {fmt(whi)}] & \\\\")
    emit(LANDSAT)
    L += [r"\addlinespace", r"\multicolumn{6}{l}{\textit{AEF-derived features}} \\"]
    emit(AEFM)
    # Benchmark: DGSIAP municipal average + (combined/P-V) the ADC-trained oracle ceiling
    sb = ev.assign(_siap=ev["siap"])
    n, ov, bt, wt, rm = met(sb, season_y, "_siap")
    L += [r"\addlinespace", r"\multicolumn{6}{l}{\textit{Benchmark}} \\",
          f"DGSIAP Mun.\\ Avg. & {n:,} & {fmt(ov)} & {fmt(bt)} & {fmt(0.0)} & {fmt(rm)} \\\\"]
    sci = boot_ci(sb, season_y, "_siap")
    if sci and show_ci:
        (olo, ohi), _ = sci
        L.append(f" & & \\scriptsize[{fmt(olo)}, {fmt(ohi)}] & & & \\\\")
    orc_path = os.path.join(P, "oracle_ceiling_2022.csv")
    if orc_season and os.path.exists(orc_path):
        o = pd.read_csv(orc_path); o = o[o["season"] == orc_season]
        if len(o):
            r = o.iloc[0]
            L.append(f"Oracle (ADC-trained) & {int(r['N']):,} & {fmt(r['R2'])} & "
                     f"{fmt(r['Btw'])} & {fmt(r['Wtn'])} & {fmt(r['RMSE'])} \\\\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    out = os.path.join(plot_dir, fname)
    with open(out, "w") as f: f.write("\n".join(L))
    print(f"Wrote {out}")

def build_common_sample_table(season_y, label_season, fname, tag):
    """Every model evaluated on the SAME ADCs (intersection of all model
    predictions + DGSIAP), so cross-model differences are not sample composition."""
    set_anchor(tag)                      # season-matched DGSIAP anchor
    models =  [nm for nm,_,_ in LANDSAT + AEFM]
    cs =  ev.dropna(subset=models + ["siap", season_y]).copy()
    n_cs =  len(cs)
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Common-sample accuracy for maize yield predictions vs.\ INEGI 2022 "
         rf"census, ADC level --- {label_season}. Every model is evaluated on the "
         rf"\emph{{same}} {n_cs:,} ADCs (the intersection of ADCs for which all Landsat- "
         rf"and AEF-derived models produce a prediction and a DGSIAP municipal yield for "
         rf"{ANCHOR_NOTE[tag]} exists), "
         r"so cross-model differences are not driven by sample composition. RMSE in t/ha.}",
         rf"\label{{tab:common_sample_{tag}}}", r"\begin{tabular}{lrrrrr}", r"\hline",
         r"Model & $N$ & $R^2$ & Between $R^2$ & Within $R^2$ & RMSE \\", r"\hline",
         r"\multicolumn{6}{l}{\textit{Landsat-derived features}} \\"]
    def emit(group):
        for nm,_,_ in group:
            raw =  met(cs, season_y, nm)
            lam =  LAM_DEPLOY; cs["_s"] =  shrink(cs, nm, lam)
            sh  =  met(cs, season_y, "_s")
            for tagn, m in [(f"{nm} Raw", raw), (f"{nm} Shrink", sh)]:
                n, ov, bt, wt, rm = m
                lab = tagn.replace("Hist. Raw", "Hist.\\ Raw").replace("Hist. Shrink", "Hist.\\ Shrink")
                lab = lab.replace("Ens. Raw", "Ens.\\ Raw").replace("Ens. Shrink", "Ens.\\ Shrink")
                L.append(f"{lab} & {n:,} & {fmt(ov)} & {fmt(bt)} & {fmt(wt)} & {fmt(rm)} \\\\")
    emit(LANDSAT)
    L += [r"\addlinespace", r"\multicolumn{6}{l}{\textit{AEF-derived features}} \\"]
    emit(AEFM)
    cs["_siap"] =  cs["siap"]
    n, ov, bt, wt, rm =  met(cs, season_y, "_siap")
    L += [r"\addlinespace", r"\multicolumn{6}{l}{\textit{Benchmark}} \\",
          f"DGSIAP Mun.\\ Avg. & {n:,} & {fmt(ov)} & {fmt(bt)} & {fmt(0.0)} & {fmt(rm)} \\\\",
          r"\hline", r"\end{tabular}", r"\end{table}", ""]
    out = os.path.join(plot_dir, fname)
    with open(out, "w") as f: f.write("\n".join(L))
    print(f"Wrote {out}  (common sample N={n_cs:,})")

print("Combined-season rows:")
set_anchor("combined")
for nm,_,_ in LANDSAT+AEFM:
    for t,m,_ in model_rows(nm,"yield"): print(f"  {t:24s} N={m[0]:>6,} R2={m[1]:.3f} Btw={m[2]:.3f} Wtn={m[3]:.3f} RMSE={m[4]:.3f}")
build_table("yield", "Combined season", "accuracy_combined_2022.tex", "combined", show_ci=False)
build_table("yield_pv", "spring-summer (P-V) season", "accuracy_spring_summer_2022.tex", "spring_summer", show_ci=False)
build_table("yield_oi", "fall-winter (O-I) season", "accuracy_fall_winter_2022.tex", "fall_winter", show_ci=False)
build_common_sample_table("yield", "Combined season", "common_sample_combined_2022.tex", "combined")
build_common_sample_table("yield_pv", "spring-summer (P-V) season", "common_sample_spring_summer_2022.tex", "spring_summer")

# ── Scatter figures (replaces the notebook's old RS-based panels) ──
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

base_font_size = 12  # match \documentclass[12pt]{article}
mpl.use("pgf")  # typeset via LaTeX/pgf so fonts match the paper (Times, as in fig 2)
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,            # don't let mpl override fonts
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.unicode_minus": False,
    "font.size": base_font_size,            # default text
    "axes.titlesize": base_font_size + 3,   # plot titles
    "axes.labelsize": base_font_size + 2,   # x/y labels
    "xtick.labelsize": base_font_size,      # tick labels
    "ytick.labelsize": base_font_size,
    "legend.fontsize": base_font_size,
    "figure.titlesize": base_font_size + 3,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#404040",
    "axes.labelcolor": "#404040",
    "xtick.color": "#404040",
    "ytick.color": "#404040",
    "grid.color": "#D0D0D0",
    "grid.linestyle": (0, (1, 3)),   # fine dotted
    "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=["#4A4A4A"]),  # default series color
    "pgf.preamble": r"""
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
""",
})

def save_pdf_png(fig, stem):
    """pgf backend writes the PDF; rasterize it to PNG with pdftoppm (as in 15_fig_methodology_diagram.py)."""
    pdf = os.path.join(plot_dir, stem + ".pdf")
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", pdf, os.path.join(plot_dir, stem)], check=True)

def scatter_panel(ax, y, yh, title, wr2=None):
    m = np.isfinite(y) & np.isfinite(yh)
    ax.hexbin(y[m], yh[m], gridsize=45, bins="log", cmap="viridis",
              extent=(0, 12, 0, 12), linewidths=0, rasterized=True)  # raster: pgf cannot hold the hex paths
    ax.plot([0, 12], [0, 12], "--", color="#B0B0B0", lw=1.0)
    ax.set_xlim(0, 12); ax.set_ylim(0, 12)
    # numbers stay in text mode (Times digits, as in the rest of the paper); only
    # R^2 and a negative sign are math, matching fmt() in the tables above
    stat = f"$R^2$ = {fmt(r2(y[m], yh[m]))}"
    if wr2 is not None:
        if np.isfinite(wr2) and abs(wr2) < 5e-4: wr2 = 0.0   # no "$-$0.000" for the DGSIAP panel
        stat += f",  within-$R^2$ = {fmt(wr2)}"
    ax.set_title(f"{title}\n{stat}", fontsize=base_font_size)
    ax.set_xlabel("Reported Yield (t/ha)", fontsize=base_font_size - 1)
    ax.tick_params(labelsize=base_font_size - 2, length=0)

SCATTER = ["NDVI", "AEF mean", "Agg-NN"]        # raw + corrected pairs
for m in SCATTER:                                        # corrected columns
    ev[f"_{m}_corr"] = correct(ev, m)

pairs = [(m, m + " Raw") for m in SCATTER]
cols  = []
for m in SCATTER:
    cols += [(m, f"{m} Raw"), (f"_{m}_corr", f"{m} Corr.")]
cols += [("siap", "DGSIAP")]

# ── Combined-season figure: the deployed Shrink specification of every model
#    plus the DGSIAP municipal-average benchmark. The shrink columns are built
#    exactly as the table rows are (shrink of the RAW predictions at
#    LAM_DEPLOY), so the panel R2s match Table \ref{tab:accuracy_combined}.
#    NOTE: set_anchor("combined") comes AFTER the `correct()` loop above, so the
#    seasonal figure's corrected panels are untouched; it only points
#    ev["siap"] at the combined-season anchor used by the benchmark panel.
set_anchor("combined")
COMBINED_PANELS = ["AEF Hist Ens.", "DGSIAP", "AEF Hist", "Agg-NN", "AEF mean", "NDVI"]
panels = []
for m in COMBINED_PANELS:
    if m == "DGSIAP":
        panels.append(("siap", "DGSIAP Mun. Avg."))
        continue
    c = f"_{m}_shrink"
    ev[c] = shrink(ev, m, LAM_DEPLOY)
    panels.append((c, f"{m} Shrink"))

ncol = 3
grid = [panels[i:i + ncol] for i in range(0, len(panels), ncol)]
fig, axes = plt.subplots(len(grid), ncol, figsize=(4.0 * ncol, 8.0))
for row, ax_row in zip(grid, axes):
    for (c, nm), ax in zip(row, ax_row):
        scatter_panel(ax, ev["yield"].values, ev[c].values, nm,
                      wr2=within_r2(ev, "yield", c))
for ax in axes[0]:
    ax.set_xlabel("")
for ax in axes[:, 0]:
    ax.set_ylabel("Predicted Yield (t/ha)", fontsize=base_font_size - 1)
fig.suptitle("Predicted vs. Reported Maize Yield (Combined Season, 2022)", y=1.01)
fig.tight_layout()
save_pdf_png(fig, "accuracy_scatter_combined_2022")
print("Wrote accuracy_scatter_combined_2022.{pdf,png}")

scols = [(c, nm) for c, nm in panels if c != "siap"]      # seasonal: the deployed Shrink panels, no DGSIAP panel
fig, axes = plt.subplots(2, len(scols), figsize=(3.3 * len(scols), 7.0))
for row, ycol, lab in [(0, "yield_oi", "fall-winter"), (1, "yield_pv", "spring-summer")]:
    for ax, (c, nm) in zip(axes[row], scols):
        scatter_panel(ax, ev[ycol].values, ev[c].values, nm)
    axes[row][0].set_ylabel(f"Predicted Yield (t/ha)\n[{lab}]", fontsize=base_font_size - 1)
fig.suptitle("Predicted vs. Reported Maize Yield by Season (2022)", y=1.01)
fig.tight_layout()
save_pdf_png(fig, "accuracy_scatter_seasonal_2022")
print("Wrote accuracy_scatter_seasonal_2022.{pdf,png}")
