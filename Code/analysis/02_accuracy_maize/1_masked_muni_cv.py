"""
1_masked_muni_cv.py — municipality-level CV R2 for the cropland-masked NDVI
(aefn2) feature set, the paper's NDVI baseline.

Replaces the 0.686 placeholder that was computed off partial back-years. Uses
the SAME protocol as mun_cv_aef_hist.py (identical HGB grid, config selection,
random muni-year K-fold) so the NDVI and AEF muni-level numbers are comparable.

CV scheme is random muni-year 5-fold, NOT leave-one-year-out -- see the
muni-cv-random-kfold decision (2026-08-07); do not reintroduce LOYO.

A muni-level cropland extraction was never run, so muni features are the ADC
features aggregated to muni-year, area-weighted by the SIAP ag-land proxy --
the same approximation 2_masked_adc_eval.py makes (mean and
quantile-bin fractions aggregate ~exactly under area weighting; sd/percentiles
approximately).

Outputs
  Data/cropland_features/muni_aefn2_masked.parquet        cached muni features
                                                          (reused by
                                                          train_holdout_validation_models.py)
  Data/predictions/mun_aefn2_masked_gb_kfold_preds.parquet  out-of-sample preds

Run: ~/miniforge3/envs/geo_env/bin/python 1_masked_muni_cv.py
     [--rebuild]   force re-aggregation of the muni feature cache
"""
import os, sys, glob, time, warnings
import numpy  as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble        import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "train"))
from siap_yields import load_muni_yields

# ── Directories ──────────────────────────────────────────
home_dir    =  os.path.expanduser("~")
proj_dir    =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir    =  os.path.join(proj_dir, "Data")
feat_dir    =  os.path.join(data_dir, "cropland_features")
csv_dir     =  os.path.join(feat_dir, "csvs_crop_aefn2")
pred_dir    =  os.path.join(data_dir, "predictions")

# ── Inputs ───────────────────────────────────────────────
agland_path =  os.path.join(data_dir, "SIAP_agland", "Output",
                            "2007_adcs_agland_area.csv")   # ADC ag-land area weights

# ── Outputs ──────────────────────────────────────────────
muni_cache  =  os.path.join(feat_dir, "muni_aefn2_masked.parquet")        # muni-year features
kfold_path  =  os.path.join(pred_dir, "mun_aefn2_masked_gb_kfold_preds.parquet")  # OOS preds

FOLDS    =  5
SEED     =  42
CROP     =  'Maize'
SEASON   =  'Spring-Summer'
HGB_GRID =  [(0.05, 500, 5), (0.05, 1000, 7), (0.03, 1000, 6),
             (0.03, 1500, 8), (0.01, 2000, 6)]      # same grid as mun_cv_aef_hist.py
META     =  {"adcid", "gs_year", "year", "muncode", "adc"}


def calc_r2(y, yh):
    ss_tot =  np.mean((y - y.mean()) ** 2)
    return 1 - np.mean((y - yh) ** 2) / ss_tot if ss_tot > 0 else 0.0


def fit_grid(X_tr, y_tr, X_va, y_va):
    best =  (None, -np.inf)
    for lr, n_est, depth in HGB_GRID:
        m =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                           learning_rate=lr, min_samples_leaf=5,
                                           random_state=SEED)
        m.fit(X_tr, y_tr)
        r2 =  calc_r2(y_va, m.predict(X_va))
        if r2 > best[1]:
            best =  ((lr, n_est, depth), r2)
    return best[0]


def build_muni_features():
    """Area-weighted ADC -> muni-year aggregation of the aefn2 features."""
    files =  sorted(glob.glob(os.path.join(csv_dir, "cropfeat_crop_aefn2_adc_*.csv")))
    print(f"[1] reading {len(files)} aefn2 ADC CSVs")
    adc =  pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    adc =  adc.rename(columns={"gs_year": "year"})
    adc["adcid"]   =  adc["adcid"].astype(str)
    adc["adc"]     =  adc["adcid"].str.replace("-", "", regex=False)
    adc["muncode"] =  adc["adcid"].str[:5]
    feat =  [c for c in adc.columns if c not in META]
    adc  =  adc.dropna(subset=feat, how="all").drop_duplicates(["adc", "year"])
    print(f"    {len(adc):,} ADC-years | {len(feat)} features | "
          f"years {sorted(adc['year'].unique())}")

    ag =  pd.read_csv(agland_path, usecols=["adcid", "siap_agland_area"])
    ag["adc"] =  ag["adcid"].astype(str).str.replace("-", "", regex=False)
    w =  adc.merge(ag[["adc", "siap_agland_area"]], on="adc", how="left")
    med =  w["siap_agland_area"].median()
    w["wt"] =  w["siap_agland_area"].fillna(med).clip(lower=1e-6)

    F  =  w[feat].to_numpy(np.float64)
    wt =  w["wt"].to_numpy(np.float64)[:, None]
    wf =  pd.DataFrame(F * wt, columns=feat)
    wf["muncode"] =  w["muncode"].values
    wf["year"]    =  w["year"].values
    wf["__w"]     =  w["wt"].values
    g =  wf.groupby(["muncode", "year"], as_index=False).sum()
    for c in feat:
        g[c] =  g[c] / g["__w"]
    muni =  g.drop(columns="__w")
    print(f"[2] aggregated to {len(muni):,} muni-years -> caching")
    muni.to_parquet(muni_cache, index=False)
    return muni


def main():
    t0 =  time.time()
    if os.path.exists(muni_cache) and "--rebuild" not in sys.argv:
        muni =  pd.read_parquet(muni_cache)
        print(f"[1-2] loaded cached muni features: {len(muni):,} muni-years "
              f"({os.path.basename(muni_cache)}; pass --rebuild to regenerate)")
    else:
        muni =  build_muni_features()

    fcols =  [c for c in muni.columns if c not in ("muncode", "year")]
    muni["muncode"] =  muni["muncode"].astype(str).str.zfill(5)

    yields =  load_muni_yields(crop=CROP, season=SEASON)
    yields["muncode"] =  yields["muncode"].astype(str).str.zfill(5)
    yields =  yields[~yields["muncode"].str.endswith("000")]   # drop state aggregates

    merged =  muni.merge(yields[["muncode", "year", "yield"]], on=["muncode", "year"])
    print(f"[3] NDVI masked muni-years: {len(merged):,}  features: {len(fcols)}  "
          f"munis: {merged['muncode'].nunique():,}")

    X =  merged[fcols].to_numpy(np.float32)
    y =  merged["yield"].to_numpy()

    # random muni-year K-fold (matches mun_cv_aef_hist.py)
    preds =  np.full(len(y), np.nan)
    for i, (tr, te) in enumerate(KFold(n_splits=FOLDS, shuffle=True,
                                       random_state=SEED).split(X), 1):
        cfg =  fit_grid(X[tr], y[tr], X[te], y[te])
        lr, n_est, depth =  cfg
        m =  HistGradientBoostingRegressor(max_iter=n_est, max_depth=depth,
                                           learning_rate=lr, min_samples_leaf=5,
                                           random_state=SEED)
        m.fit(X[tr], y[tr]); preds[te] =  m.predict(X[te])
        print(f"    fold {i}/{FOLDS}: cfg={cfg}  fold R2={calc_r2(y[te], preds[te]):.4f}")

    r2_k =  calc_r2(y, preds)
    print(f"\n{'='*64}")
    print(f"NDVI (masked, aefn2) muni-level R2, random muni-year {FOLDS}-fold "
          f"= {r2_k:.4f}")
    print(f"  (replaces the 0.686 placeholder computed on partial back-years)")
    print(f"{'='*64}")

    out =  merged[["muncode", "year", "yield"]].copy()
    out["yield_pred"] =  preds
    out.to_parquet(kfold_path, index=False)
    print(f"saved -> {os.path.basename(kfold_path)}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
