"""
harmonic_adc_eval.py — ADC-level evaluation of the harmonic NDVI models against
the INEGI 2022 Censo Agropecuario, mirroring `gb_aef_hist_ensemble.py` exactly
(same ground truth, same ex-ante ag-land correction, same metrics) so the NDVI
and AEF rows of the accuracy table are directly comparable.

Procedure per feature set:
  1. train HistGB on ALL municipality-years (SIAP Maize/Spring-Summer, 2017-2024)
  2. predict every ADC in EVAL_YEAR from the ADC-level harmonic features
  3. merge onto INEGI CA22 ADC maize yields (combined + p-v season)
  4. apply the additive ex-post correction using the ex-ante ag-land area proxy
     (NOT census planted area, which would not be known ex-ante)
  5. report N / R2 / Between-R2 / Within-R2 / RMSE, raw and corrected

Usage:
  ~/miniforge3/envs/geo_env/bin/python harmonic_adc_eval.py                  # all sets
  ~/miniforge3/envs/geo_env/bin/python harmonic_adc_eval.py --features h3_fixed
"""
import os
import sys
import argparse

import numpy  as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from siap_yields import load_muni_yields, SIAP_PATH

ALL_SETS =  ['h3_fixed', 'h3_quantile', 'h2_fixed', 'h2_quantile', 'coefs']

parser =  argparse.ArgumentParser()
parser.add_argument('--features', default=None, choices=ALL_SETS)
parser.add_argument('--eval_year', type=int, default=2022)
parser.add_argument('--crop', default='Maize')
parser.add_argument('--season', default='Spring-Summer')
parser.add_argument('--winsor', type=float, default=0.005)
parser.add_argument('--subsample', type=int, default=0,
                    help='K synthetic ADC-like draws per muni-year (0 = off). Mirrors '
                         'subsample_bins() in gb_aef_hist_ensemble.py: muni histograms '
                         'are built from ~100k pixels but ADC ones from ~1.5k, so the '
                         'model otherwise trains and predicts on different distributions.')
args =  parser.parse_args()

# Empirical ADC pixel-count distribution, measured over 12,335 ADCs in 2022
# (p5 57 / p25 476 / p50 1,560 / p75 4,821 / p95 42,544). Synthetic draws sample
# N from this so the augmented training rows carry realistic ADC-level noise.
ADC_NPIX_Q =  np.array([57, 476, 1560, 4821, 42544], dtype=float)

EVAL_YEAR =  args.eval_year
SETS      =  [args.features] if args.features else ALL_SETS

# ── Directories ──────────────────────────────────────────
home_dir   =  os.path.expanduser("~")
proj_dir   =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction")
data_dir   =  os.path.join(proj_dir, "Data")
feat_dir   =  os.path.join(data_dir, "harmonic_features")
pred_dir   =  os.path.join(data_dir, "predictions")
inegi_dir  =  os.path.join(data_dir, "INEGI", "MD_lab_outputs")
ca2022_dir =  os.path.join(inegi_dir, "LM2304-CA22-2025-09-29-superficie_ENTREGA")
os.makedirs(pred_dir, exist_ok=True)

# ── Inputs ───────────────────────────────────────────────
ca_use_path  =  os.path.join(ca2022_dir, "adc_land_use_ca22_adc07.dta")   # INEGI ADC maize yields
ca_szn_path  =  os.path.join(ca2022_dir, "adc_land_szn_ca22_adc07.dta")   # by season (p-v)
agland_path  =  os.path.join(data_dir, "SIAP_agland", "Output",
                             "2007_adcs_agland_area.csv")                 # ex-ante area proxy

# HGB config: the grid's mid setting, used for every set (no per-year tuning
# here — this is a fit-on-everything, predict-out-of-domain evaluation).
HGB =  dict(max_iter=1000, max_depth=6, learning_rate=0.03,
            min_samples_leaf=5, random_state=42)


def r2(y, yh):
    m =  np.isfinite(y) & np.isfinite(yh)
    y, yh =  np.array(y[m]), np.array(yh[m])
    if len(y) < 2:
        return np.nan
    st =  np.sum((y - np.mean(y)) ** 2)
    return 1 - np.sum((y - yh) ** 2) / st if st > 0 else np.nan


def within_r2(df, y, p, gc='muncode'):
    sub =  df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    c   =  sub.groupby(gc).size()
    sub =  sub[sub[gc].isin(c[c >= 2].index)]
    if len(sub) == 0:
        return np.nan
    gm =  sub.groupby(gc)[[y, p]].transform('mean')
    return r2(sub[y] - gm[y], sub[p] - gm[p])


def between_r2(df, y, p, gc='muncode'):
    sub =  df[[y, p, gc]].replace([np.inf, -np.inf], np.nan).dropna()
    g   =  sub.groupby(gc)[[y, p]].mean()
    return r2(g[y], g[p])


def eval_row(df, ycol, pcol, label, out):
    sub =  df[[ycol, pcol, 'muncode']].replace([np.inf, -np.inf], np.nan).dropna()
    n    =  len(sub)
    ov   =  r2(sub[ycol], sub[pcol])
    b    =  between_r2(sub, ycol, pcol)
    w    =  within_r2(sub, ycol, pcol)
    rmse =  np.sqrt(np.mean((sub[ycol].values - sub[pcol].values) ** 2))
    print(f"  {label:<45s} {n:>8,} {ov:>6.3f} {b:>6.3f} {w:>7.3f} {rmse:>6.3f}")
    out.append(dict(model=label, n=n, r2=ov, between=b, within=w, rmse=rmse))


def subsample_hists(X, n_blocks, K, seed=42):
    """K synthetic ADC-like copies of X by multinomial resampling each 256-bin
    block at a realistic ADC pixel count. Returns (K*n, d) stacked."""
    rng =  np.random.default_rng(seed)
    n, d =  X.shape
    bs   =  d // n_blocks
    out  =  np.empty((K * n, d), dtype=np.float32)
    for k in range(K):
        npix =  rng.choice(ADC_NPIX_Q, size=n)          # empirical ADC pixel counts
        npix =  np.maximum(npix * rng.uniform(0.5, 2.0, size=n), 20).astype(int)
        for b in range(n_blocks):
            blk =  np.clip(np.nan_to_num(X[:, b*bs:(b+1)*bs], nan=0.0), 0, None)
            # float64 + a hair of slack: np.multinomial rejects pvals whose
            # float64-cast sum exceeds 1.0, which float32 rows routinely do.
            p   =  blk.astype(np.float64)
            s   =  p.sum(axis=1, keepdims=True); s[s < 1e-12] =  1.0
            p   =  p / (s * (1.0 + 1e-9))
            for i in range(n):
                out[k*n + i, b*bs:(b+1)*bs] =  rng.multinomial(npix[i], p[i]) / npix[i]
    return out


def load_gt():
    ca =  pd.read_stata(ca_use_path)
    gt =  ca[ca['name'] == args.crop][['adc', 'muncode', 'yield', 'land_input']].copy()
    szn =  pd.read_stata(ca_szn_path)
    gt_pv =  szn[(szn['name'] == args.crop) & (szn['type'] == 'p-v')][
        ['adc', 'muncode', 'yield']].rename(columns={'yield': 'yield_pv'})
    gt =  gt.merge(gt_pv, on=['adc', 'muncode'], how='left')
    agland =  pd.read_csv(agland_path)
    agland['adc'] =  agland['adc07'].astype(str).str.replace('-', '', regex=False)
    gt =  gt.merge(agland[['adc', 'siap_agland_area']], on='adc', how='left')
    gt['corr_w'] =  np.where(gt['siap_agland_area'] > 0, gt['siap_agland_area'],
                             gt['land_input'])
    return gt


def run_set(fs, gt, yields, siap_mun, siap_mun_all, results):
    name  =  'harmonic_coefs' if fs == 'coefs' else fs
    muni  =  pd.read_parquet(os.path.join(feat_dir, f"muni_{name}.parquet"))
    fcols =  [c for c in muni.columns if c not in ('muncode', 'year')]

    if fs == 'coefs' and args.winsor > 0:
        lo, hi =  muni[fcols].quantile(args.winsor), muni[fcols].quantile(1 - args.winsor)
        muni[fcols] =  muni[fcols].clip(lo, hi, axis=1)

    muni['muncode'] =  muni['muncode'].astype(str).str.zfill(5).astype(int)
    tr =  muni.merge(yields, on=['muncode', 'year'])
    Xtr =  tr[fcols].to_numpy(np.float32)
    ytr =  tr['yield'].to_numpy()
    if args.subsample > 0 and fs != 'coefs':          # coefs aren't a histogram
        nb  =  len(fcols) // 256
        Xtr =  np.vstack([Xtr, subsample_hists(Xtr, nb, args.subsample)])
        ytr =  np.concatenate([ytr] + [ytr] * args.subsample)
        print(f"  augmented: {len(ytr):,} training rows ({nb} blocks x {args.subsample} draws)")
    m  =  HistGradientBoostingRegressor(**HGB)
    m.fit(Xtr, ytr)

    # push the year filter into the parquet read: adc_h3_*.parquet is 2.34M x 768
    # (~7 GB) and only the EVAL_YEAR slice is needed.
    adc =  pd.read_parquet(os.path.join(feat_dir, f"adc_{name}.parquet"),
                           filters=[('year', '==', EVAL_YEAR)])
    if fs == 'coefs' and args.winsor > 0:
        adc[fcols] =  adc[fcols].clip(lo, hi, axis=1)
    adc['pred'] =  m.predict(adc[fcols].to_numpy(np.float32)).clip(0)
    adc['adc']  =  adc['adcid'].astype(str).str.replace('-', '', regex=False)

    df =  gt.merge(adc[['adc', 'pred']], on='adc', how='left')

    # ── additive ex-post correction, ex-ante ag-land weights ─────────────
    # The municipal anchor must MATCH the census target scored (fixed
    # 2026-08-15). Previously ONE season-filtered anchor (siap_mun, built from
    # --season, i.e. Spring-Summer by default) was reused for BOTH blocks, so
    # the combined-season rows were corrected against a Spring-Summer municipal
    # mean. That mismatch is worth ~0.13 R2 (cf. 0.391 vs 0.517 for the masked
    # NDVI baseline). Combined -> all seasons summed; P-V -> Spring-Summer.
    df['wQ'] =  df['pred'] * df['corr_w']
    df['wA'] =  np.where(np.isfinite(df['pred']), df['corr_w'], 0)
    agg0 =  df.groupby('muncode').agg({'wQ': 'sum', 'wA': 'sum'}).reset_index()
    agg0['pred_mun_avg'] =  agg0['wQ'] / agg0['wA']

    def _add_corrected(anchor_df, colname):
        a =  agg0.merge(anchor_df, on='muncode', how='left')
        a['diff'] =  a['pred_mun_avg'] - a['yield_siap']
        m =  df.merge(a[['muncode', 'diff']], on='muncode', how='left')
        out =  (m['pred'] - m['diff']).clip(lower=0)
        out[m['pred'].isna()] =  np.nan
        df[colname] =  out.values

    _add_corrected(siap_mun_all, 'pred_corr')      # combined: all seasons
    _add_corrected(siap_mun,     'pred_corr_pv')   # P-V: --season (Spring-Summer)

    print(f"\n  --- {fs}: combined season ---")
    eval_row(df, 'yield', 'pred',      f'NDVI {fs} Raw',   results)
    eval_row(df, 'yield', 'pred_corr', f'NDVI {fs} Corr.', results)
    pv =  df[df['yield_pv'].notna()]
    print(f"  --- {fs}: spring-summer (P-V) ---")
    eval_row(pv, 'yield_pv', 'pred',         f'NDVI {fs} Raw (P-V)',   results)
    eval_row(pv, 'yield_pv', 'pred_corr_pv', f'NDVI {fs} Corr. (P-V)', results)

    adc[['adcid', 'year', 'pred']].to_parquet(
        os.path.join(pred_dir, f"adc_harmonic_{fs}_preds.parquet"), index=False)
    return df


def main():
    gt     =  load_gt()
    yields =  load_muni_yields(crop=args.crop, season=args.season)
    print(f"GT rows: {len(gt):,}   muni training rows: {len(yields):,}")

    # Season-matched municipal anchors (see run_set): the P-V block uses the
    # --season anchor, the combined block uses all growing seasons summed.
    sm =  yields[yields['year'] == EVAL_YEAR].copy()
    sm['muncode'] =  sm['muncode'].astype(str).str.zfill(5)
    siap_mun =  sm[['muncode', 'yield']].rename(columns={'yield': 'yield_siap'})

    _sa =  pd.read_stata(SIAP_PATH)
    _sa['muncode'] =  _sa['muncode'].apply(lambda x: str(int(x)).zfill(5))
    _sa =  _sa[(_sa['name'] == args.crop) & (_sa['year'] == EVAL_YEAR)]
    _sa =  _sa[~_sa['muncode'].str.endswith('000')]
    siap_mun_all =  (_sa.groupby('muncode')
                        .agg(q=('q', 'sum'), ha=('ha_planted', 'sum')).reset_index())
    siap_mun_all['yield_siap'] =  siap_mun_all['q'] / siap_mun_all['ha']
    siap_mun_all =  siap_mun_all[['muncode', 'yield_siap']]
    print(f"anchors: combined {len(siap_mun_all):,} munis | "
          f"{args.season} {len(siap_mun):,} munis")

    print(f"\n{'='*80}\n  {'Model':<45s} {'N':>8s} {'R2':>6s} {'Btw':>6s} "
          f"{'Wtn':>7s} {'RMSE':>6s}\n  {'-'*77}")
    results =  []
    for fs in SETS:
        try:
            run_set(fs, gt, yields, siap_mun, siap_mun_all, results)
        except Exception as e:
            print(f"  !! {fs} failed: {e}")
    pd.DataFrame(results).to_csv(
        os.path.join(pred_dir, "harmonic_adc_eval_summary.csv"), index=False)
    print(f"\nsaved -> {os.path.join(pred_dir, 'harmonic_adc_eval_summary.csv')}")


if __name__ == '__main__':
    main()
