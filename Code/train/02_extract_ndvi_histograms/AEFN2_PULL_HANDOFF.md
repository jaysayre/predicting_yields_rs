# aefn2 (cropland-masked NDVI) extraction + pull — session handoff

# ✅ EXTRACTION COMPLETE 2026-08-15 — 4,784/4,784 batches pulled locally.
# Eval + accuracy tables re-run on the full panel. Remaining work is the
# muni-CV R² and the Overleaf prose (see "What's left" at the bottom).

**Last updated:** 2026-08-15
**What this tracks:** the `2_ls_cropland_features.py --tag aefn2` extraction — cropland-masked
NDVI features (AEF-parity recipe, 150 per-dim feats) that will become the paper's **new
NDVI baseline**, replacing the unmasked h3 2D-histogram rows.

---

## Final status (2026-08-15) — DONE

| Metric | Value |
|---|---|
| Target total | **4,784** batches (598/yr × 8 yrs, 2017–2024) |
| On Drive | **4,784** (100%) |
| Local CSVs pulled | **4,784** — every year at 598, all 32 states |
| Panel rows | 1,529,215 ADC-years × 150 features |

*Trace: 2026-08-12 4,158 / 2026-08-13 4,573 / 2026-08-15 4,784.*

### How the last 211 actually got done (the useful lesson)

The tail was NOT quota-bound and NOT simply slow — it was **queue contention**:

- **`listOperations` ages out old SUCCEEDED tasks**, so the EE task list *undercounts*
  badly (it read 4,516 when Drive held 4,774). **Never** measure progress from it —
  measure from Drive (`rclone lsf` map) or the local file count.
- 5 of 6 projects were quota-**restricted**; only **`<project-6>`** had quota, and
  its slots were being burned on **406 PENDING duplicates** of batches already on Drive,
  while the 10 genuinely-missing batches (state 32 / 2024) sat parked on the *locked*
  `<project-3>`. So the queue was busy doing nothing useful.
- Fix: cancel the duplicates, **resubmit the 10 on the unrestricted project**
  (`--states 32 --years 2024 --allow_dup_pending`). Done in ~40 min instead of
  waiting for the Sep 1 reset. Batching is **per-state**, so `--states 32` reproduces
  identical batch indices — safe to resubmit a slice.
- Reusable tooling: `scratchpad/aefn2_gap_and_cancel_plan.py` (dry-run by default)
  diffs Drive against the complete-2022 batch list, then cancels only
  confirmed-duplicate queued tasks, never a copy of a missing batch.

Note: the pull's `have / want` line can show local **ahead of** the task-list count —
same aging-out cause. Not an error.

Local dir: `~/Dropbox/Projects/Maize_prediction/Data/cropland_features/csvs_crop_aefn2/`

---

## Why it was slow (historical — superseded by the section above)

Original diagnosis: bottleneck is **per-USER Earth Engine concurrency (~1 slot)**, NOT
quota; a 40-min monitor showed fresh projects' tasks never getting scheduled. That held
while all queues were James's. It turned out to be **incomplete**: once
`<project-6>` (a different person's account, hence its own concurrency) was in play
it sustained 4–5 concurrent — but spent them on duplicate work. See "How the last 211
actually got done".

---

## Historical — commands used while the extraction was still running

### 1. Check task states + rate (one command)
```bash
~/miniforge3/envs/geo_env/bin/python - <<'PY'
import ee, warnings; warnings.filterwarnings('ignore')
from collections import Counter
projects = "<project-1> <project-2> <project-3> <project-4> <project-5> <project-6>".split()
succ=set(); running=0; pending=0; run_proj=Counter(); done_day=Counter()
for p in projects:
    try: ee.Initialize(project=p)
    except Exception: print(f"  {p}: init fail"); continue
    for o in ee.data.listOperations():
        md=o.get('metadata',{}); d=md.get('description',''); s=md.get('state')
        if not d.startswith('cropfeat_crop_aefn2_adc_'): continue
        if s=='SUCCEEDED':
            succ.add(d); end=md.get('endTime','')
            if end: done_day[end[:10]]+=1
        elif s=='RUNNING': running+=1; run_proj[p]+=1
        elif s=='PENDING': pending+=1
print(f"SUCCEEDED unique: {len(succ)} / 4784")
print(f"RUNNING: {running} by project: {dict(run_proj)}")
print(f"PENDING: {pending}")
print("completions/day (last 8):")
for day in sorted(done_day)[-8:]: print(f"   {day}: {done_day[day]}")
PY
```

### 2. Pull whatever newly SUCCEEDED (safe to re-run anytime)
```bash
cd "~/CalAg Dropbox/Jay Sayre/Github/predicting_yields_rs/Code/train/02_extract_ndvi_histograms"
# actual path: /home/jdesktop/CalAg\ Dropbox/Jay\ Sayre/Github/predicting_yields_rs/Code/train/02_extract_ndvi_histograms
TAG=crop_aefn2 ./3_pull_cropland_csvs.sh
```
- Traverses Drive duplicate folders (a few min) then fetches by file-ID. Resume-safe.
- Prints `have / want` at the end. Re-run to backfill rate-limited stragglers.
- Default PROJECTS list already includes `<project-5> <project-6>`.

### 3. Count local
```bash
find ~/Dropbox/Projects/Maize_prediction/Data/cropland_features/csvs_crop_aefn2 \
     -name 'cropfeat_crop_aefn2_adc_*.csv' | wc -l
```

---

## What's left

**Done 2026-08-15 on the full panel:**

1. ✅ **Masked eval re-run** — `partial_masked_mun_train_adc_eval.py` (the `partial_`
   prefix is now historical; rename when convenient). Wrote
   `Data/predictions/adc_aefn2_masked_preds.parquet` (190,932 ADCs).
   Final (after the item-6 anchor fix): **P-V Corr. R²=0.403, within −0.065,
   N=68,202**; **combined Corr. R²=0.517, within −0.077, N=70,760**. Raw rows:
   combined 0.232 / −0.082, P-V 0.244 / −0.062.
2. ✅ **Accuracy tables regenerated** — `accuracy_main_2022.py` wrote
   `accuracy_{combined,spring_summer,fall_winter}_2022.tex`,
   `common_sample_{combined,spring_summer}_2022.tex` (common N=57,421 / 55,892),
   and both scatter PDF/PNGs. Combined-season NDVI (masked): Raw −0.082 within,
   Corr. −0.077, Shrink **+0.085** — still below every AEF variant
   (AEF Hist Ens. Shrink R²=0.598, within **+0.225**).

3. ✅ **Muni-level CV R² recomputed = 0.662** (was the 0.686 placeholder).
   Random muni-year 5-fold, 17,692 muni-years / 2,267 munis, folds 0.638–0.681.
   Producer is now **`analysis/02_accuracy_maize/masked_muni_cv.py`** — kept in the
   repo, NOT a scratchpad (the original `scratchpad/masked_muni_cv.py` was lost when
   that ephemeral dir was cleared). It caches muni features to
   `Data/cropland_features/muni_aefn2_masked.parquet` and writes
   `mun_aefn2_masked_gb_kfold_preds.parquet`.
5. ✅ **Folded to a single "NDVI (masked)" row** in both tables:
   - `4_validation_mun_level.py` (Table 1) — one Landsat row from
     `mun_aefn2_masked_gb_holdout_preds.parquet`, produced by the new
     `train_ndvi_masked()` in `3_train_holdout_validation_models.py`
     (run `python3 3_train_holdout_validation_models.py masked` to redo just that row).
     Result: **N=3,571 R²=0.534 RMSE=1.490**, vs AEF mean 0.579 / AEF Hist 0.611 /
     AEF Hist Ens. 0.627 / Agg-NN 0.812 on the identical held-out sample.
   - `mun_survey_improvement.py` — agg. row now from `adc_aefn2_masked_preds.parquet`,
     mun-trained GB row from the masked K-fold preds. Panel A: agg. 0.613,
     GB (mun.) 0.677, vs AEF Hist Ens. 0.750 and SIAP 0.623.

**Still open:**

4. **Prose to Overleaf `new_body.tex`** — ⚠️ the parked draft
   `scratchpad/rs_baseline_rewrite_draft.tex` is **GONE** (ephemeral scratchpad cleared),
   so the methodology-subsec rewrite and the 3 results-prose updates (L292 five families,
   L300 core paragraph, L302 coverage caveat) must be **rewritten**, not copied.
   Numbers to use: **muni-CV 0.662** (not 0.686), coverage 68,604 / 70,195 / 58,022 →
   re-derive from the current tables. Also update abstract/conclusion 0.447.
6. ✅ **`correct()` discrepancy RESOLVED 2026-08-15 — and it was a real bug in the
   published tables, not just a script mismatch.**

   Cause: the **SIAP municipal anchor season did not match the census target**.
   `accuracy_main_2022.py` built ONE all-seasons anchor and used it for every table,
   so the P-V and O-I tables corrected against a combined-season municipal mean;
   the eval script did the mirror opposite (Spring-Summer anchor on its combined row).
   Verified by holding preds/weights/formula fixed and swapping only the anchor —
   it reproduces both numbers exactly (`scratchpad/anchor_diagnosis.py`):

   | target | anchor | N | R² |
   |---|---|---|---|
   | combined | Spring-Summer | 70,363 | 0.391 |
   | combined | all seasons | 70,760 | **0.517** ✓ |
   | P-V | Spring-Summer | 68,202 | **0.403** ✓ |
   | P-V | all seasons | 68,320 | 0.481 |

   Fix: anchors are now season-matched in BOTH scripts (combined → all seasons,
   P-V → Spring-Summer, O-I → Fall-Winter). In `accuracy_main_2022.py` the anchor
   also defines the SIAP benchmark row and the common-sample intersection, so all
   three follow the season now; captions state which anchor was used.

   **Published P-V corrected rows all moved DOWN** (the mismatch had been flattering
   every model): NDVI 0.481→0.403, AEF mean →0.451, AEF Hist →0.500,
   AEF Hist Ens. →0.498, Agg-NN →0.407. Combined-season rows are unchanged.
   Both scripts now agree exactly on every metric and N.

   **Full sweep of the anchor bug across the repo (2026-08-15/16).** Any script doing
   the additive ex-post correction was audited; the rule is *the SIAP anchor season
   must match the census target scored*:

   | Script | Status |
   |---|---|
   | `accuracy_main_2022.py` | ✅ fixed — per-table anchors (also drives SIAP benchmark row + common-sample) |
   | `partial_masked_mun_train_adc_eval.py` | ✅ fixed — `pred_corr` (all szn) + `pred_corr_pv` (S-S) |
   | `matched_mask_vs_unmasked.py` | ✅ fixed — combined target ⇒ all-seasons anchor; re-run done |
   | `harmonic_adc_eval.py` | ✅ fixed — two anchors threaded through `run_set`; summary regenerated |
   | `2_gb_aef_hist_ensemble.py` | ✅ fixed — adds `pred_corr_pv`; re-run done |
   | `gb_aef_hist_ensemble_dm.py` | ✅ fixed + re-run 2026-08-16 |
   | `gb_aef_hist_ensemble_qbin.py` | ✅ fixed + re-run 2026-08-16 |
   | `4_gb_aef_hist_ensemble_other_crops.py` | ✅ already correct — combined target + all-szn anchor, no P-V rows |
   | `census_thought_all_models.py`, `mun_agg_hist_ens.py` | ✅ already correct |

   Cross-check that the fix is right: `2_gb_aef_hist_ensemble.py` and
   `accuracy_main_2022.py` are independently-written correction paths and now agree —
   P-V Corr. 0.499 (N=57,701) vs 0.498 (N=57,731); combined Corr. 0.576 both.
   `gb_aef_hist_ensemble_dm.py`'s baseline comparison row reproduces 0.499 / N=57,701
   exactly. All three `adc_aef_hist_ens{,_dm,_qbin}_eval.parquet` now carry
   `pred_corr_pv` (Spring-Summer) alongside `pred_corr` (all seasons).

   ⚠️ **Gotcha when patching these scripts:** a per-script grep for the model's own
   `Corr.` row is NOT enough — `_dm` also has a *baseline comparison* block that reads
   `adc_aef_hist_ens_eval.parquet` and scored P-V with `pred_corr`. It was missed on the
   first pass and only caught because its 0.546 didn't match the expected 0.499. Grep
   `eval_row(.*yield_pv` across the whole file, not just the model's own rows.

   `matched_mask_vs_unmasked.py` re-run (full panel, matched N 29,008→57,638 because the
   old run predated completion): NDVI unmasked Corr. 0.506, **NDVI masked Corr. 0.531**,
   AEF Hist Ens. Corr. 0.576.

   ⚠️ Note on `harmonic_adc_eval_summary.csv`: the previous file held only `h2_quantile`
   (a partial `--features` run). Its **Raw** R² also differs from the new run
   (0.216 → 0.057) — Raw never touches the anchor, so something upstream changed since
   that old run (features regenerated / different labels). Do NOT read the old-vs-new
   delta as the anchor fix; the new all-5-set run is the reference.

8. ✅ **Separate bug found + fixed 2026-08-16: `4_gb_aef_hist_ensemble_other_crops.py`
   still had the pre-2026-07-27 `fillna(0)` behaviour.** ADCs with no cropland pixels
   under the WorldCover mask carry all-null features; `fillna(0)` turned them into
   all-zero vectors that got confident near-constant predictions. The main maize script
   fixed this on 2026-07-27; other_crops never got the port, so
   `accuracy_other_crops_adc_2022.tex` (last built 2026-06-10) was contaminated.
   Null-drop ported (27,145 ADCs dropped, same as the maize run) and re-run.

   **The two perennials took the hit** — their ADCs are poorly captured by the cropland
   mask, so far more were being zero-filled:

   | Crop | N old→new | Raw R² | Corr. R² | Shrink R² |
   |---|---|---|---|---|
   | Sorghum | 7,095→6,984 | 0.456→0.462 | 0.385→0.392 | 0.458→0.464 |
   | **Sugar** | 5,759→**5,064** | 0.374→**0.286** | 0.207→**−0.015** | 0.455→**0.379** |
   | Wheat | 3,368→3,344 | 0.393→0.397 | 0.524→0.524 | 0.461→0.463 |
   | **Avocados** | 6,601→**5,144** | −0.447→**−0.556** | −0.486→**−0.550** | −0.349→**−0.495** |

   Also changed: that script used to **overwrite the live Overleaf .tex on every run**.
   It now writes to `tables/` only and needs `--write_overleaf` to touch Overleaf
   (template is still read from there). ✅ **Overleaf PUSHED 2026-08-16 07:52** with
   `--write_overleaf` — all 12 AEF Hist Ens. rows replaced, surrounding table structure
   untouched; pre-write copy in scratchpad.

   ⚠️ No GEE re-pull was needed (asked and checked): the nulls are already in the local
   `alpha_earth_mex_adcs*.parquet`; this is purely how they're filtered.

   **Mask coverage by crop** (share of 2022 census ADCs with NO cropland pixels, i.e.
   all-null features and therefore dropped). The AEF extraction masks to WorldCover
   class **40 (Cropland) only** — `6_ee_alpha_earth_binned_hist.py`, `wc.eq(40)`:

   | Crop | census ADCs | w/ features | all-null | % lost |
   |---|---|---|---|---|
   | Avocados | 9,079 | 6,283 | 2,700 | **29.7%** |
   | Maize | 95,264 | 73,698 | 20,344 | 21.4% |
   | Sugar | 9,394 | 8,160 | 1,173 | **12.5%** |
   | Sorghum | 12,138 | 11,551 | 516 | 4.3% |
   | Wheat | 6,120 | 5,889 | 174 | 2.8% |

   The perennials lose most — avocado orchards are usually WorldCover class 10 (Tree
   cover), not 40. **DECISION 2026-08-16: not re-extracting** under a broadened
   (40 ∪ 10) or unmasked recipe — considered and declined; don't reopen without a
   reason. Consequence to state as a caveat if these rows are used: the avocado/sugar
   metrics are computed on the ~70% / ~87.5% of census ADCs that have cropland pixels,
   and that subset is not random.

7. ⏸️ **Fall-winter (O-I) corrected rows look pathological** — every model shows a
   NEGATIVE Between R² (≈ −0.64), and the SIAP benchmark itself scores R²=0.776 with
   Between −0.636 on only 1,150 anchor munis. **User decision 2026-08-16: ignore.**
   O-I is a secondary table. Noted only so it isn't rediscovered as new.

**Do NOT** re-optimize the extractor (tileScale/reducers) — feature values must stay
comparable across the completed panel.

## Related memory
- `aefn2-masked-extraction-state.md` — extraction recipe + project details
- `muni-cv-random-kfold.md` — the LOYO→random K-fold CV change (don't reintroduce LOYO)
- `phase-c-needs-all-years.md` — don't start Phase C retrain/tables until full panel extracted
