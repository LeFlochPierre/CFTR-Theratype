# Phase 14C — Completed rescue ablation / diagnostic caveat

All results below are the user's **training-only** 2×5 position-grouped cross-validation observations; position MAE in pp WT.

| Model | Overall MAE | Prior ETI-characterised MAE | VTI-only MAE |
|---|---:|---:|---:|
| Original VTI Forest | 17.798 | 20.395 | 17.323 |
| Forest + real Band C gain | 17.851 | 20.058 | 17.524 |
| Forest + shuffled Band C | 17.726 | 20.045 | 17.347 |

Real auxiliary versus shuffled-label control: overall improvement **−0.125 pp**, paired position-bootstrap CI [−0.407, +0.146]. No demonstrated rescue prediction benefit. This does **not** contradict the strong measured maturation–function association in Phase 14B.

**Implementation caveat:** the initial `maturation_prediction.py` exported the genuine Band C auxiliary prediction as NaN due to checking a stale loop variable `shuffle` instead of the current `name`. The rescue comparison above completed before the crash and is usable for this exploratory ablation. Auxiliary Band C prediction MAE and Spearman **must remain marked incomplete** until the patch is applied and the script rerun successfully. No NaN performance values should be reported as scientific results.

Frozen 2024 and 2026 validation/test outcomes remained sealed in this exploratory analysis. Avoid further tuning the same multi-output Forest.
