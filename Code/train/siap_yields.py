"""
siap_yields.py — the ONE definition of municipality maize yields.

Replaces `Data/muni_grano_yields.csv` (deprecated 2026-07-26), which stopped at
2022, had no builder script in the repo, and was a DIFFERENT series from the one
the AEF models are scored on: on the overlapping years the two agreed exactly
only 52% of the time (corr 0.972, mean 2.55 vs 2.44). Models trained on it were
therefore not comparable to the published AEF numbers.

The canonical series is the one `gb_aef_hist_ensemble.py` builds inline:
SIAP by-season, filtered to a crop and growing season, yield = q / ha_planted.
Covers 1980-2024 (so the NDVI panel's full 2017-2024 window is usable).

Import from any training script:
    import sys, os
    sys.path.insert(0, os.path.join(<...>, "Code", "train"))
    from siap_yields import load_muni_yields
    yields = load_muni_yields()                       # Maize, Spring-Summer
"""
import os

import pandas as pd

SIAP_PATH =  os.path.join(os.path.expanduser("~"), "Dropbox", "Projects",
                          "The Promise of Crop Substitution", "data", "SIAP",
                          "Cleaned", "siap_ag_prod_estimation_by_season.dta")


def load_muni_yields(crop='Maize', season='Spring-Summer', path=None,
                     min_year=None, max_year=None):
    """Municipality-year maize yields (t/ha) from SIAP.

    Returns columns: muncode (int), year (int), yield (float), ha_planted.
    Rows with missing or non-positive yield are dropped.
    """
    siap =  pd.read_stata(path or SIAP_PATH,
                          columns=['year', 'muncode', 'name', 'growing_season',
                                   'q', 'ha_planted'])
    siap =  siap[(siap['name'] == crop) & (siap['growing_season'] == season)].copy()
    siap['yield']   =  siap['q'] / siap['ha_planted']
    siap =  siap[siap['yield'].notna() & (siap['yield'] > 0)]
    siap['muncode'] =  siap['muncode'].astype(str).str.zfill(5).astype(int)
    siap['year']    =  siap['year'].astype(int)
    if min_year is not None:
        siap =  siap[siap['year'] >= min_year]
    if max_year is not None:
        siap =  siap[siap['year'] <= max_year]
    return siap[['muncode', 'year', 'yield', 'ha_planted']].reset_index(drop=True)
