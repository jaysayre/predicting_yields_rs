"""
matched_mask_vs_unmasked.py — matched-sample comparison under the REAL pipeline
(muni-train -> ADC-validate), on the IDENTICAL set of 2022 census ADCs, so the
only thing that varies is the feature source:

  NDVI masked   (aefn2, 150 per-dim feats, cropland-masked)  <- partial 2022
  NDVI unmasked (h3 quantile 2D-hist, 768 feats, paper's current NDVI Hist)
  AEF Hist Ens  (ceiling reference; predictions from adc_aef_hist_ens_eval)

Masked + unmasked are trained here (HistGB, all muni-years, SIAP Maize/S-S);
AEF uses its already-saved eval predictions. All three are evaluated on the
common ADC set only. Reuses the aggregation + correction logic of
partial_masked_mun_train_adc_eval.py.

Run: ~/miniforge3/envs/geo_env/bin/python matched_mask_vs_unmasked.py
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

home_dir  =  os.path.expanduser("~")
proj_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir  =  os.path.join(proj_dir, "Data")
csv_dir   =  os.path.join(data_dir, "cropland_features", "csvs_crop_aefn2")
harm_dir  =  os.path.join(data_dir, "harmonic_features")
pred_dir  =  os.path.join(data_dir, "predictions")
inegi_dir =  os.path.join(data_dir, "INEGI", "MD_lab_outputs")
ca_dir    =  os.path.join(inegi_dir, "LM2304-CA22-2025-09-29-superficie_ENTREGA")
ca_use    =  os.path.join(ca_dir, "adc_land_use_ca22_adc07.dta")
ca_szn    =  os.path.join(ca_dir, "adc_land_szn_ca22_adc07.dta")
agland    =  os.path.join(data_dir, "SIAP_agland", "Output", "2007_adcs_agland_area.csv")
META =  {"adcid", "gs_year", "year", "muncode", "adc"}


def r2(y, yh):
    m = np.isfinite(y) & np.isfinite(yh); y, yh = np.array(y[m]), np.array(yh[m])
    if len(y) < 2: return np.nan
    st = np.sum((y - y.mean())**2)
    return 1 - np.sum((y - yh)**2)/st if st > 0 else np.nan

def decomp(df, y, p, gc="muncode"):
    s = df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    c = s.groupby(gc).size(); s2 = s[s[gc].isin(c[c >= 2].index)]
    gm = s2.groupby(gc)[[y, p]].transform("mean")
    g  = s.groupby(gc)[[y, p]].mean()
    ov = r2(s[y], s[p]); bt = r2(g[y], g[p]); wt = r2(s2[y]-gm[y], s2[p]-gm[p])
    rmse = np.sqrt(np.mean((s[y].values - s[p].values)**2))
    return len(s), ov, bt, wt, rmse

def show(df, y, p, label, out):
    n, ov, bt, wt, rmse = decomp(df, y, p)
    print(f"  {label:<32s} {n:>7,} {ov:>6.3f} {bt:>6.3f} {wt:>7.3f} {rmse:>6.3f}")
    out.append(dict(model=label, n=n, r2=ov, between=bt, within=wt, rmse=rmse))


def load_gt():
    ca = pd.read_stata(ca_use)
    gt = ca[ca["name"] == "Maize"][["adc", "muncode", "yield", "land_input"]].copy()
    szn = pd.read_stata(ca_szn)
    pv = szn[(szn["name"] == "Maize") & (szn["type"] == "p-v")][
        ["adc", "muncode", "yield"]].rename(columns={"yield": "yield_pv"})
    gt = gt.merge(pv, on=["adc", "muncode"], how="left")
    ag = pd.read_csv(agland); ag["adc"] = ag["adc07"].astype(str).str.replace("-", "", regex=False)
    gt = gt.merge(ag[["adc", "siap_agland_area"]], on="adc", how="left")
    gt["corr_w"] = np.where(gt["siap_agland_area"] > 0, gt["siap_agland_area"], gt["land_input"])
    gt["adc"] = gt["adc"].astype(str)
    return gt


def correct(df, pcol, siap_mun):
    """additive ex-post correction with ex-ante ag-land weights -> pcol+'_corr'."""
    d = df.copy()
    d["wQ"] = d[pcol] * d["corr_w"]
    d["wA"] = np.where(np.isfinite(d[pcol]), d["corr_w"], 0)
    agg = d.groupby("muncode").agg({"wQ": "sum", "wA": "sum"}).reset_index()
    agg["pma"] = agg["wQ"] / agg["wA"]
    agg = agg.merge(siap_mun, on="muncode", how="left")
    agg["diff"] = agg["pma"] - agg["yield_siap"]
    d = d.merge(agg[["muncode", "diff"]], on="muncode", how="left")
    out = (d[pcol] - d["diff"]).clip(lower=0)
    out[d[pcol].isna()] = np.nan
    return out.values


# ── masked (aefn2) ───────────────────────────────────────────────────────
def masked_preds(yields):
    files = sorted(glob.glob(os.path.join(csv_dir, "cropfeat_crop_aefn2_adc_*.csv")))
    adc = pd.concat((pd.read_csv(f) for f in files), ignore_index=True).rename(
        columns={"gs_year": "year"})
    adc["adcid"] = adc["adcid"].astype(str)
    adc["adc"] = adc["adcid"].str.replace("-", "", regex=False)
    adc["muncode"] = adc["adcid"].str[:5]
    feat = [c for c in adc.columns if c not in META]
    adc = adc.dropna(subset=feat, how="all").drop_duplicates(["adc", "year"])
    ag = pd.read_csv(agland, usecols=["adc07", "siap_agland_area"])
    ag["adc"] = ag["adc07"].astype(str).str.replace("-", "", regex=False)
    w = adc.merge(ag, on="adc", how="left")
    w["wt"] = w["siap_agland_area"].fillna(w["siap_agland_area"].median()).clip(lower=1e-6)
    wf = pd.DataFrame(w[feat].to_numpy(float) * w["wt"].to_numpy(float)[:, None], columns=feat)
    wf["muncode"] = w["muncode"].values; wf["year"] = w["year"].values; wf["__w"] = w["wt"].values
    g = wf.groupby(["muncode", "year"], as_index=False).sum()
    for c in feat: g[c] = g[c] / g["__w"]
    g["muncode"] = g["muncode"].astype(str).str.zfill(5).astype(int)
    tr = g.merge(yields, on=["muncode", "year"])
    m = HistGradientBoostingRegressor(**HGB).fit(tr[feat].to_numpy(np.float32), tr["yield"].to_numpy())
    pe = adc[adc["year"] == EVAL_YEAR].copy()
    pe["pred_masked"] = m.predict(pe[feat].to_numpy(np.float32)).clip(0)
    print(f"  masked: {len(tr):,} train rows, {len(pe):,} ADCs predicted")
    return pe[["adc", "pred_masked"]]


# ── unmasked (h3 quantile 2D-hist) ───────────────────────────────────────
def unmasked_preds(yields):
    muni = pd.read_parquet(os.path.join(harm_dir, "muni_h3_quantile.parquet"))
    feat = [c for c in muni.columns if c.startswith("h3q_")]
    muni["muncode"] = muni["muncode"].astype(str).str.zfill(5).astype(int)
    tr = muni.merge(yields, on=["muncode", "year"])
    m = HistGradientBoostingRegressor(**HGB).fit(tr[feat].to_numpy(np.float32), tr["yield"].to_numpy())
    adc = pd.read_parquet(os.path.join(harm_dir, "adc_h3_quantile.parquet"),
                          filters=[("year", "==", EVAL_YEAR)])
    adc["adc"] = adc["adcid"].astype(str).str.replace("-", "", regex=False)
    adc["pred_unmasked"] = m.predict(adc[feat].to_numpy(np.float32)).clip(0)
    print(f"  unmasked: {len(tr):,} train rows, {len(adc):,} ADCs predicted")
    return adc[["adc", "pred_unmasked"]]


def main():
    yields = load_muni_yields(crop="Maize", season="Spring-Summer")   # TRAINING labels (S-S)
    gt = load_gt()

    # ── Municipal anchor for the ex-post correction (fixed 2026-08-15) ────
    # Every row below is scored against the COMBINED-season census target
    # (`yield`), so the anchor must be the combined-season SIAP municipal yield
    # (all growing seasons summed) -- NOT the Spring-Summer training labels,
    # which is what this used before. The mismatch understated every Corr. row
    # (cf. 0.391 vs 0.517 in partial_masked_mun_train_adc_eval.py). Raw rows are
    # unaffected: the anchor only enters the correction.
    siap_all = pd.read_stata(SIAP_PATH)
    siap_all["muncode"] = siap_all["muncode"].apply(lambda x: str(int(x)).zfill(5))
    s22 = siap_all[(siap_all["name"] == "Maize") & (siap_all["year"] == EVAL_YEAR)]
    s22 = s22[~s22["muncode"].str.endswith("000")]
    siap_mun = (s22.groupby("muncode")
                   .agg(q=("q", "sum"), ha=("ha_planted", "sum")).reset_index())
    siap_mun["yield_siap"] = siap_mun["q"] / siap_mun["ha"]
    siap_mun = siap_mun[["muncode", "yield_siap"]]

    print("[predict]")
    pm = masked_preds(yields)
    pu = unmasked_preds(yields)
    aef = pd.read_parquet(os.path.join(pred_dir, "adc_aef_hist_ens_eval.parquet"))[
        ["adc", "pred", "pred_corr"]].rename(columns={"pred": "pred_aef", "pred_corr": "pred_aef_corr"})
    aef["adc"] = aef["adc"].astype(str)

    df = gt.merge(pm, on="adc", how="inner").merge(pu, on="adc", how="inner").merge(aef, on="adc", how="inner")
    df = df.dropna(subset=["pred_masked", "pred_unmasked", "pred_aef"])
    print(f"\n[matched common ADCs] {len(df):,}  "
          f"(states {sorted(df['muncode'].astype(str).str.zfill(5).str[:2].unique())})")

    df["pred_masked_corr"]   = correct(df, "pred_masked",   siap_mun)
    df["pred_unmasked_corr"] = correct(df, "pred_unmasked", siap_mun)

    print(f"\n{'='*72}\n  {'Model (combined season)':<32s} {'N':>7s} {'R2':>6s} "
          f"{'Btw':>6s} {'Wtn':>7s} {'RMSE':>6s}\n  {'-'*68}")
    res = []
    show(df, "yield", "pred_unmasked",      "NDVI unmasked (h3q) Raw",  res)
    show(df, "yield", "pred_unmasked_corr", "NDVI unmasked (h3q) Corr.", res)
    show(df, "yield", "pred_masked",        "NDVI masked (aefn2) Raw",  res)
    show(df, "yield", "pred_masked_corr",   "NDVI masked (aefn2) Corr.", res)
    show(df, "yield", "pred_aef",           "AEF Hist Ens Raw  [ceil]", res)
    show(df, "yield", "pred_aef_corr",      "AEF Hist Ens Corr [ceil]", res)

    out = os.path.join(proj_dir, "plots", "matched_mask_vs_unmasked.csv")
    pd.DataFrame(res).to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    print("NOTE: matched sample; masked NDVI has partial 2022 coverage + approximate "
          "muni aggregation, so its numbers are a lower bound, not final.")


if __name__ == "__main__":
    main()
