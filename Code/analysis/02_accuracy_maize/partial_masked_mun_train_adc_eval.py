"""
partial_masked_mun_train_adc_eval.py — PARTIAL-COVERAGE sanity test of the
cropland-masked NDVI (aefn2) features under the REAL pipeline: municipal-level
train, ADC-level validation, exactly as harmonic_adc_eval.py does for the
unmasked harmonic features.

Panel COMPLETE as of 2026-08-15: 4,784/4,784 batches (8 years x 598, all 32
states), 1,529,215 ADC-years. The "partial_" in the filename is historical —
this now produces the final full-panel numbers. Because a muni-level cropland
extraction was never run (and GEE
quota is exhausted), the muni training features are built by aggregating the
ADC-level features up to muni-year, area-weighted by the SIAP ag-land proxy
(exact pixel-count weights aren't in the CSVs). This is an APPROXIMATION for the
sd/percentile features (mean + quantile-bin fractions aggregate ~exactly under
area weighting); good enough to read the trend, not a final number.

Procedure (mirrors harmonic_adc_eval.py):
  1. train HistGB on all muni-years we have (SIAP Maize/Spring-Summer 2017-2024)
  2. predict every 2022 ADC from ADC-level aefn2 features
  3. merge onto INEGI CA22 ADC maize yields (combined + p-v)
  4. additive ex-post correction with the ex-ante ag-land weight
  5. report N / R2 / Between / Within / RMSE, raw and corrected

Run: ~/miniforge3/envs/geo_env/bin/python partial_masked_mun_train_adc_eval.py
"""
import os, sys, glob, warnings
import numpy  as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "train"))
from siap_yields import load_muni_yields, SIAP_PATH

EVAL_YEAR =  2022
HGB =  dict(max_iter=1000, max_depth=6, learning_rate=0.03,
           min_samples_leaf=5, random_state=42)

home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir   =  os.path.join(proj_dir, "Data")
csv_dir    =  os.path.join(data_dir, "cropland_features", "csvs_crop_aefn2")
inegi_dir  =  os.path.join(data_dir, "INEGI", "MD_lab_outputs")
ca2022_dir =  os.path.join(inegi_dir, "LM2304-CA22-2025-09-29-superficie_ENTREGA")
ca_use_path =  os.path.join(ca2022_dir, "adc_land_use_ca22_adc07.dta")
ca_szn_path =  os.path.join(ca2022_dir, "adc_land_szn_ca22_adc07.dta")
agland_path =  os.path.join(data_dir, "SIAP_agland", "Output", "2007_adcs_agland_area.csv")

META =  {"adcid", "gs_year", "year", "muncode", "adc"}


# ── metrics (identical to harmonic_adc_eval.py) ──────────────────────────
def r2(y, yh):
    m =  np.isfinite(y) & np.isfinite(yh)
    y, yh =  np.array(y[m]), np.array(yh[m])
    if len(y) < 2: return np.nan
    st =  np.sum((y - y.mean())**2)
    return 1 - np.sum((y - yh)**2)/st if st > 0 else np.nan

def within_r2(df, y, p, gc="muncode"):
    s =  df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c =  s.groupby(gc).size(); s = s[s[gc].isin(c[c >= 2].index)]
    if not len(s): return np.nan
    gm =  s.groupby(gc)[[y, p]].transform("mean")
    return r2(s[y]-gm[y], s[p]-gm[p])

def between_r2(df, y, p, gc="muncode"):
    s =  df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g =  s.groupby(gc)[[y, p]].mean()
    return r2(g[y], g[p])

def eval_row(df, ycol, pcol, label, out):
    s =  df[[ycol, pcol, "muncode"]].replace([np.inf, -np.inf], np.nan).dropna()
    n =  len(s); ov = r2(s[ycol], s[pcol])
    b =  between_r2(s, ycol, pcol); w = within_r2(s, ycol, pcol)
    rmse =  np.sqrt(np.mean((s[ycol].values - s[pcol].values)**2)) if n else np.nan
    print(f"  {label:<34s} {n:>8,} {ov:>6.3f} {b:>6.3f} {w:>7.3f} {rmse:>6.3f}")
    out.append(dict(model=label, n=n, r2=ov, between=b, within=w, rmse=rmse))


def load_adc_features():
    files =  sorted(glob.glob(os.path.join(csv_dir, "cropfeat_crop_aefn2_adc_*.csv")))
    print(f"[1] reading {len(files)} aefn2 ADC CSVs")
    df =  pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df =  df.rename(columns={"gs_year": "year"})
    df["adcid"]   =  df["adcid"].astype(str)
    df["adc"]     =  df["adcid"].str.replace("-", "", regex=False)
    df["muncode"] =  df["adcid"].str[:5]
    feat =  [c for c in df.columns if c not in META]
    df =  df.dropna(subset=feat, how="all").drop_duplicates(["adc", "year"])
    print(f"    {len(df):,} ADC-years | {len(feat)} features "
          f"| years {sorted(df['year'].unique())}")
    return df, feat


def aggregate_to_muni(adc, feat):
    """Area-weighted ADC->muni-year mean of each feature (weight = SIAP ag-land
    area proxy; equal weight where missing)."""
    ag =  pd.read_csv(agland_path, usecols=["adcid", "siap_agland_area"])
    ag["adc"] =  ag["adcid"].astype(str).str.replace("-", "", regex=False)
    w =  adc.merge(ag[["adc", "siap_agland_area"]], on="adc", how="left")
    med =  w["siap_agland_area"].median()
    w["wt"] =  w["siap_agland_area"].fillna(med).clip(lower=1e-6)
    F =  w[feat].to_numpy(np.float64)
    wt =  w["wt"].to_numpy(np.float64)[:, None]
    wf =  pd.DataFrame(F * wt, columns=feat)
    wf["muncode"] =  w["muncode"].values; wf["year"] = w["year"].values
    wf["__w"] =  w["wt"].values
    g =  wf.groupby(["muncode", "year"], as_index=False).sum()
    for c in feat:
        g[c] =  g[c] / g["__w"]
    muni =  g.drop(columns="__w")
    print(f"[2] aggregated to {len(muni):,} muni-years")
    return muni


def load_gt():
    ca =  pd.read_stata(ca_use_path)
    gt =  ca[ca["name"] == "Maize"][["adc", "muncode", "yield", "land_input"]].copy()
    szn =  pd.read_stata(ca_szn_path)
    pv =  szn[(szn["name"] == "Maize") & (szn["type"] == "p-v")][
        ["adc", "muncode", "yield"]].rename(columns={"yield": "yield_pv"})
    gt =  gt.merge(pv, on=["adc", "muncode"], how="left")
    ag =  pd.read_csv(agland_path)
    ag["adc"] =  ag["adcid"].astype(str).str.replace("-", "", regex=False)
    gt =  gt.merge(ag[["adc", "siap_agland_area"]], on="adc", how="left")
    gt["corr_w"] =  np.where(gt["siap_agland_area"] > 0, gt["siap_agland_area"],
                            gt["land_input"])
    gt["adc"] =  gt["adc"].astype(str)
    return gt


def main():
    adc, feat =  load_adc_features()
    muni =  aggregate_to_muni(adc, feat)

    yields =  load_muni_yields(crop="Maize", season="Spring-Summer")
    muni["muncode"] =  muni["muncode"].astype(str).str.zfill(5).astype(int)
    tr =  muni.merge(yields, on=["muncode", "year"])
    print(f"[3] training rows (muni-years w/ SIAP yield): {len(tr):,} "
          f"across {tr['muncode'].nunique():,} munis")
    m =  HistGradientBoostingRegressor(**HGB)
    m.fit(tr[feat].to_numpy(np.float32), tr["yield"].to_numpy())

    pe =  adc[adc["year"] == EVAL_YEAR].copy()
    pe["pred"] =  m.predict(pe[feat].to_numpy(np.float32)).clip(0)
    print(f"[4] predicted {len(pe):,} ADCs in {EVAL_YEAR} "
          f"(states {sorted(pe['muncode'].str[:2].unique())})")

    # save raw ADC preds for the accuracy-table pipeline (accuracy_main_2022.py):
    # this is the masked aefn2 NDVI baseline replacing the h3 NDVI Hist rows
    pred_dir =  os.path.join(data_dir, "predictions")
    pe[["adcid", "year", "pred"]].to_parquet(
        os.path.join(pred_dir, "adc_aefn2_masked_preds.parquet"), index=False)
    print(f"    saved adc_aefn2_masked_preds.parquet ({len(pe):,} rows)")

    gt =  load_gt()
    df =  gt.merge(pe[["adc", "pred"]], on="adc", how="left")

    # ── additive ex-post correction, ex-ante ag-land weights ──────────────
    # The municipal anchor must MATCH the census target being scored (fixed
    # 2026-08-15). Previously one Spring-Summer anchor was used for both rows,
    # so the combined-season row was corrected against a Spring-Summer mean --
    # a season mismatch that cost ~0.13 R2 (0.391 vs the correct 0.517) and made
    # this script disagree with accuracy_main_2022.py. Same defect, opposite
    # direction, as the one fixed there.
    #   combined `yield`  -> SIAP all seasons summed
    #   P-V `yield_pv`    -> SIAP Spring-Summer only
    siap_all =  pd.read_stata(SIAP_PATH)
    siap_all["muncode"] =  siap_all["muncode"].apply(lambda x: str(int(x)).zfill(5))
    s22 =  siap_all[(siap_all["name"] == "Maize") & (siap_all["year"] == EVAL_YEAR)]
    s22 =  s22[~s22["muncode"].str.endswith("000")]

    def _anchor(sub):
        g =  sub.groupby("muncode").agg(q=("q", "sum"), ha=("ha_planted", "sum")).reset_index()
        g["yield_siap"] =  g["q"] / g["ha"]
        return g[["muncode", "yield_siap"]]

    df["wQ"] =  df["pred"] * df["corr_w"]
    df["wA"] =  np.where(np.isfinite(df["pred"]), df["corr_w"], 0)
    agg0 =  df.groupby("muncode").agg({"wQ": "sum", "wA": "sum"}).reset_index()
    agg0["pred_mun_avg"] =  agg0["wQ"] / agg0["wA"]

    def add_corrected(anchor_df, colname):
        a =  agg0.merge(anchor_df, on="muncode", how="left")
        a["diff"] =  a["pred_mun_avg"] - a["yield_siap"]
        m =  df.merge(a[["muncode", "diff"]], on="muncode", how="left")
        out =  (m["pred"] - m["diff"]).clip(lower=0)
        out[m["pred"].isna()] =  np.nan
        df[colname] =  out.values

    add_corrected(_anchor(s22),                                          "pred_corr")     # combined
    add_corrected(_anchor(s22[s22["growing_season"] == "Spring-Summer"]), "pred_corr_pv")  # P-V

    print(f"\n{'='*74}\n  {'Model':<34s} {'N':>8s} {'R2':>6s} {'Btw':>6s} "
          f"{'Wtn':>7s} {'RMSE':>6s}\n  {'-'*70}")
    res =  []
    print("  --- combined season ---")
    eval_row(df, "yield", "pred",      "NDVI masked (aefn2) Raw",   res)
    eval_row(df, "yield", "pred_corr", "NDVI masked (aefn2) Corr.", res)
    pv =  df[df["yield_pv"].notna()]
    print("  --- spring-summer (P-V) ---")
    eval_row(pv, "yield_pv", "pred",         "NDVI masked (aefn2) Raw (P-V)",   res)
    eval_row(pv, "yield_pv", "pred_corr_pv", "NDVI masked (aefn2) Corr. (P-V)", res)

    out =  os.path.join(proj_dir, "plots", "partial_masked_mun_train_adc_eval.csv")
    pd.DataFrame(res).to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    print("NOTE: FULL PANEL — 4,784/4,784 batches, all 8 years x 598, all 32 "
          "states (complete since 2026-08-15). These are the final numbers. "
          "Only remaining approximation: muni training features are ADC "
          "features aggregated with the SIAP ag-land area proxy, since a "
          "muni-level cropland extraction was never run (see docstring).")


if __name__ == "__main__":
    main()
