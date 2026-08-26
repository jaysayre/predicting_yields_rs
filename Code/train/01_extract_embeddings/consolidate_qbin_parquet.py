"""
consolidate_qbin_parquet.py
===========================
Consolidate GEE CSV exports of quantile-bin AEF histograms (from
ee_alpha_earth_quantile_hist.py) into parquets matching the schema of the
fixed-bin versions.

Inputs:  Data/alpha_earth/mun_qbin_hist_csvs/aef_qbinhist_mun_state_*.csv
         Data/alpha_earth/adcs_qbin_hist_csvs/aef_qbinhist_state_*.csv
Outputs: Data/alpha_earth/alpha_earth_mex_mun_qbin_hist.parquet
         Data/alpha_earth/alpha_earth_mex_adcs_qbin_hist.parquet

Usage:
  ~/miniforge3/envs/geo_env/bin/python consolidate_qbin_parquet.py
"""
import os
import glob

import pandas as pd

home_dir =  os.path.expanduser("~")
aef_dir  =  os.path.join(home_dir, "Dropbox", "Projects", "Maize_prediction",
                         "Data", "alpha_earth")

bin_cols =  [f"A{d:02d}_b{b}" for d in range(64) for b in range(8)]


def consolidate(pattern, out_path, id_setup):
    files =  sorted(glob.glob(pattern))
    print(f"{len(files)} CSVs for {os.path.basename(out_path)}")
    dfs =  []
    for f in files:
        df =  pd.read_csv(f)
        dfs.append(id_setup(df))
    out =  pd.concat(dfs, ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}  {out.shape}")
    return out


def mun_ids(df):
    df['CVE_ENT'] =  df['CVE_ENT'].astype(str).str.zfill(2)
    df['CVE_MUN'] =  df['CVE_MUN'].astype(str).str.zfill(3)
    df['muncode'] =  df['CVE_ENT'] + df['CVE_MUN']
    return df[['CVE_ENT', 'CVE_MUN', 'year', 'muncode'] + bin_cols]


def adc_ids(df):
    df['adcid']   =  df['adcid'].astype(str)
    df['muncode'] =  df['adcid'].str[:5]
    return df[['adcid', 'muncode', 'year'] + bin_cols]


consolidate(os.path.join(aef_dir, "mun_qbin_hist_csvs", "aef_qbinhist_mun_*.csv"),
            os.path.join(aef_dir, "alpha_earth_mex_mun_qbin_hist.parquet"), mun_ids)
consolidate(os.path.join(aef_dir, "adcs_qbin_hist_csvs", "aef_qbinhist_state_*.csv"),
            os.path.join(aef_dir, "alpha_earth_mex_adcs_qbin_hist.parquet"), adc_ids)
