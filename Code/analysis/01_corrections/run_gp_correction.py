"""
run_gp_correction.py
====================
Standalone script equivalent of 2_gp_correction_2022.ipynb.
Runs the GP correction for RS, AEF, and MLP predictions.
"""

import os
import time
import numpy  as np
import pandas as pd
import geopandas as gpd
from   scipy.spatial.distance import cdist
from   itertools import product

### Directories
dropbox_dir   =  os.path.join(os.path.expanduser("~"), "Dropbox", "Projects")
poppy_dir     =  os.path.join(dropbox_dir,  "Maize_prediction")
crop_sub_dir  =  os.path.join(dropbox_dir,  "The Promise of Crop Substitution")
siap_dir      =  os.path.join(crop_sub_dir, "data", "SIAP", "Cleaned")
joel_dir      =  os.path.join(poppy_dir,    "Data", "predictions")
py_md_lab_dir =  os.path.join(poppy_dir,    "Data", "INEGI", "MD_lab_outputs")
ca2022_adc_d  =  os.path.join(py_md_lab_dir,"LM2304-CA22-2025-09-29-superficie_ENTREGA")
mun_shp_dir   =  os.path.join(dropbox_dir,  "Avocado_Deforestation", "Data", "raw", "spatial", "Municipality_shp")

### Inputs
adc_22_07_path =  os.path.join(ca2022_adc_d, "adc_land_use_ca22_adc07.dta")
rs_preds_file  =  os.path.join(joel_dir, "adc_yield_preds_corrected_2022.csv")
aef_preds_file =  os.path.join(joel_dir, "adc_alpha_earth_preds.csv")
mlp_preds_file =  os.path.join(joel_dir, "adc_mlp_yield_preds.csv")
siap_path      =  os.path.join(siap_dir, "siap_ag_prod_estimation_ca2007.dta")
mun_shp_path   =  os.path.join(mun_shp_dir, "MUNICIPIOS.shp")

### Outputs
gp_out_path    =  os.path.join(joel_dir, "adc_gp_yield_preds_2022.csv")


### ------------------------------------------------------------------ ###
### Load Data
### ------------------------------------------------------------------ ###

print("--- Loading data ---")
t0 =  time.time()

adc_df             =  pd.read_stata(adc_22_07_path)
adc_df             =  adc_df[adc_df['name'] == 'Maize']
adc_df             =  adc_df[['adc', 'muncode', 'land_input']]

rs_df              =  pd.read_csv(rs_preds_file)
rs_df['adc']       =  rs_df['adcid'].str.replace('-', '', regex=False)
rs_df              =  rs_df.rename(columns={'yield_pred_ls': 'yield_pred_rs'})
rs_df              =  rs_df[['adc', 'yield_pred_rs']]

aef_df             =  pd.read_csv(aef_preds_file)
aef_df['adc']      =  aef_df['adcid'].str.replace('-', '', regex=False)
aef_df             =  aef_df.rename(columns={'yield_pred': 'yield_pred_aef'})
aef_df             =  aef_df[['adc', 'yield_pred_aef']]

mlp_df             =  pd.read_csv(mlp_preds_file)
mlp_df             =  mlp_df[mlp_df['year'] == 2022]
mlp_df['adc']      =  mlp_df['adcid'].str.replace('-', '', regex=False)
mlp_df             =  mlp_df.rename(columns={'pred_yield': 'yield_pred_mlp'})
mlp_df             =  mlp_df[['adc', 'yield_pred_mlp']]

siap_df            =  pd.read_stata(siap_path)
siap_df            =  siap_df[siap_df['name'] == 'Maize']
siap_df['yield']   =  siap_df['q'] / siap_df['ha_planted']
siap_df['muncode'] =  siap_df['muncode'].apply(lambda x: str(int(x)).zfill(5))
siap_df            =  siap_df[siap_df['year'] == 2022]
siap_df            =  siap_df[['muncode', 'yield']].rename(columns={'yield': 'yield_siap'})

mun_gdf             =  gpd.read_file(mun_shp_path)
mun_gdf['muncode']  =  mun_gdf['CVE_ENT'] + mun_gdf['CVE_MUN']
mun_gdf             =  mun_gdf.to_crs(epsg=4326)
mun_gdf['cen_lon']  =  mun_gdf.geometry.centroid.x
mun_gdf['cen_lat']  =  mun_gdf.geometry.centroid.y
mun_centroids       =  mun_gdf[['muncode', 'cen_lat', 'cen_lon']].copy()

### Merge
df =  adc_df.copy()
df =  df.merge(rs_df,  on='adc', how='left')
df =  df.merge(aef_df, on='adc', how='left')
df =  df.merge(mlp_df, on='adc', how='left')
df =  df.merge(mun_centroids, on='muncode', how='left')
df['CVE_ENT'] =  df['muncode'].str.slice(0, 2)

print(f"  Load time: {time.time()-t0:.1f}s")
print(f"  ADCs: {len(df):,}")
print(f"  RS: {df['yield_pred_rs'].notna().sum():,}")
print(f"  AEF: {df['yield_pred_aef'].notna().sum():,}")
print(f"  MLP: {df['yield_pred_mlp'].notna().sum():,}")


### ------------------------------------------------------------------ ###
### GP correction functions
### ------------------------------------------------------------------ ###

def deep_gp_predict(H_train, y_train, D2_train, H_test, D2_test_train,
                    sigma, sigma_b, sigma_e, r_loc, b_prior):
    N =  len(y_train)
    p =  H_train.shape[1]
    B_inv =  np.eye(p) / (sigma_b**2)
    K  =  sigma**2 * np.exp(-D2_train / (2.0 * r_loc**2))
    K +=  sigma_e**2 * np.eye(N)
    try:
        L =  np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K +=  1e-6 * np.eye(N)
        L  =  np.linalg.cholesky(K)
    K_inv_H =  np.linalg.solve(L, H_train)
    K_inv_H =  np.linalg.solve(L.T, K_inv_H)
    K_inv_y =  np.linalg.solve(L, y_train)
    K_inv_y =  np.linalg.solve(L.T, K_inv_y)
    HtKinv_H =  H_train.T @ K_inv_H
    A        =  B_inv + HtKinv_H
    rhs      =  K_inv_H.T @ y_train + B_inv @ b_prior
    beta_bar =  np.linalg.solve(A, rhs)
    resid =  y_train - H_train @ beta_bar
    alpha =  np.linalg.solve(L, resid)
    alpha =  np.linalg.solve(L.T, alpha)
    K_star =  sigma**2 * np.exp(-D2_test_train / (2.0 * r_loc**2))
    preds  =  H_test @ beta_bar + K_star @ alpha
    return preds


def gp_loso_cv_fast(H_all, y_all, D2_full, state_ids, states,
                    sigma, sigma_b, sigma_e, r_loc, b_prior):
    all_resid =  []
    for st in states:
        mask_te =  state_ids == st
        mask_tr =  ~mask_te
        if mask_tr.sum() < 10 or mask_te.sum() == 0:
            continue
        preds =  deep_gp_predict(
            H_all[mask_tr], y_all[mask_tr],
            D2_full[np.ix_(mask_tr, mask_tr)],
            H_all[mask_te],
            D2_full[np.ix_(mask_te, mask_tr)],
            sigma, sigma_b, sigma_e, r_loc, b_prior
        )
        all_resid.append(y_all[mask_te] - preds)
    if len(all_resid) == 0:
        return np.inf
    all_resid =  np.concatenate(all_resid)
    return np.sqrt(np.mean(all_resid**2))


### ------------------------------------------------------------------ ###
### GP correction: tune and predict
### ------------------------------------------------------------------ ###

raw_pred_cols  =  ['yield_pred_rs', 'yield_pred_aef', 'yield_pred_mlp']
b_prior        =  np.array([1.0, 0.0])

sigma_grid     =  [0.5, 1.0, 2.0, 4.0]
sigma_b_grid   =  [0.1, 0.5, 1.0, 5.0]
sigma_e_grid   =  [0.1, 0.5, 1.0, 2.0]
r_loc_grid     =  [0.5, 1.0, 2.0, 5.0]

### Build municipality-level area-weighted predictions
for pc in raw_pred_cols:
    df['_wQ_' + pc] =  df[pc] * df['land_input']
    df['_wA_' + pc] =  df.apply(
        lambda x, col=pc: x['land_input'] if np.isfinite(x[col]) else 0, axis=1
    )

agg_tmp  =  ['_wQ_' + pc for pc in raw_pred_cols] + ['_wA_' + pc for pc in raw_pred_cols]
mun_gp   =  df.groupby('muncode')[agg_tmp].sum().reset_index()

for pc in raw_pred_cols:
    mun_gp[pc + '_mun'] =  mun_gp['_wQ_' + pc] / mun_gp['_wA_' + pc]
    mun_gp.loc[mun_gp['_wA_' + pc] == 0, pc + '_mun'] =  np.nan

mun_gp =  mun_gp[['muncode'] + [pc + '_mun' for pc in raw_pred_cols]]
mun_gp =  mun_gp.merge(siap_df, on='muncode', how='left')
mun_gp =  mun_gp.merge(mun_centroids, on='muncode', how='left')
mun_gp['CVE_ENT'] =  mun_gp['muncode'].str.slice(0, 2)

gp_best_params =  {}

for pc in raw_pred_cols:
    tag   =  pc.replace('yield_pred_', '')
    pcmun =  pc + '_mun'

    mun_tr =  mun_gp[[pcmun, 'yield_siap', 'cen_lat', 'cen_lon', 'CVE_ENT']].replace(
        [np.inf, -np.inf], np.nan).dropna().copy()
    print(f"\n{'='*70}")
    print(f"  GP tuning for {tag.upper()} ({len(mun_tr):,} municipalities)")
    print(f"{'='*70}")

    H_arr     =  np.column_stack([mun_tr[pcmun].values, np.ones(len(mun_tr))])
    y_arr     =  mun_tr['yield_siap'].values
    G_arr     =  mun_tr[['cen_lat', 'cen_lon']].values
    D2_full   =  cdist(G_arr, G_arr, metric='sqeuclidean')
    state_ids =  mun_tr['CVE_ENT'].values
    states    =  sorted(np.unique(state_ids))

    best_rmse   =  np.inf
    best_params =  None
    n_tested    =  0
    n_total     =  len(sigma_grid) * len(sigma_b_grid) * len(sigma_e_grid) * len(r_loc_grid)
    t_start     =  time.time()

    for sig, sb, se, rl in product(sigma_grid, sigma_b_grid, sigma_e_grid, r_loc_grid):
        rmse_cv =  gp_loso_cv_fast(H_arr, y_arr, D2_full, state_ids, states,
                                    sig, sb, se, rl, b_prior)
        n_tested += 1
        if rmse_cv < best_rmse:
            best_rmse   =  rmse_cv
            best_params =  (sig, sb, se, rl)
        if n_tested % 64 == 0:
            elapsed =  time.time() - t_start
            print(f"    {n_tested}/{n_total} tested, best RMSE={best_rmse:.4f} [{elapsed:.0f}s]")

    sig_best, sb_best, se_best, rl_best =  best_params
    gp_best_params[tag] =  best_params
    elapsed =  time.time() - t_start
    print(f"  Best LOSO RMSE = {best_rmse:.4f} [{elapsed:.0f}s]")
    print(f"  sigma={sig_best}, sigma_b={sb_best}, sigma_e={se_best}, r_loc={rl_best}")

    ### Fit final GP on all training municipalities, predict at ADC level
    adc_mask =  df[pc].notna() & df['cen_lat'].notna()
    adc_sub  =  df.loc[adc_mask].copy()

    H_adc       =  np.column_stack([adc_sub[pc].values, np.ones(len(adc_sub))])
    G_adc       =  adc_sub[['cen_lat', 'cen_lon']].values
    D2_adc_tr   =  cdist(G_adc, G_arr, metric='sqeuclidean')

    gp_preds =  deep_gp_predict(H_arr, y_arr, D2_full, H_adc, D2_adc_tr,
                                 sig_best, sb_best, se_best, rl_best, b_prior)

    col_gp       =  f'yield_pred_{tag}_gp'
    df[col_gp]   =  np.nan
    df.loc[adc_mask, col_gp] =  gp_preds

    n_gp =  df[col_gp].notna().sum()
    print(f"  {col_gp}: {n_gp:,} ADC predictions")


### ------------------------------------------------------------------ ###
### Save Results
### ------------------------------------------------------------------ ###

gp_cols  =  ['adc'] + [f'yield_pred_{pc.replace("yield_pred_", "")}_gp'
                        for pc in raw_pred_cols]
gp_out   =  df[gp_cols].copy()
gp_out.to_csv(gp_out_path, index=False)

print(f"\nWritten: {gp_out_path}")
print(f"  Rows: {len(gp_out):,}")
for c in gp_cols[1:]:
    print(f"  {c}: {gp_out[c].notna().sum():,} non-null")

print(f"\nBest hyperparameters:")
for tag, (sig, sb, se, rl) in gp_best_params.items():
    print(f"  {tag:4s}: sigma={sig}, sigma_b={sb}, sigma_e={se}, r_loc={rl}")

print("\nDone.")
