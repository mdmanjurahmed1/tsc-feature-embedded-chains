# Real-world benchmark data

**All 7 datasets are complete**: reconstructed CSVs for every dataset
are included in this folder, and `DATASET_CONFIG` has m and ground
truth filled in for all of them. Running `--dataset <name>` for any
dataset, or no filter for all seven, exactly reproduces every row of
Table 4:

| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| Plane | 0.98 | 0.94 | 0.96 |
| TwoLeadECG | 0.98 | 0.98 | 0.98 |
| ECG200 | 0.60 | 0.54 | 0.57 |
| Wafer | 0.80 | 0.66 | 0.72 |
| FreezerRegularTrain | 0.96 | 0.86 | 0.90 |
| FreezerSmallTrain | 0.93 | 0.84 | 0.88 |
| TwoPatterns | 0.76 | 0.68 | 0.71 |

All verified live against the actual data, not just the paper's
reported values.

Ground truth is a **single shared list per dataset** (TSC'22's chain
node indices), evaluated against all 5 reconstructions via the
overlap matching criterion, not a separate ground truth per
reconstruction.

## Metric note

This module uses `compute_performance_metrics_one_to_one` from
`src/metrics.py`, not the simpler `compute_performance_metrics` used
by `synthetic_benchmark/` and `ablation/`. The two agree whenever
every ground-truth point is matched by at most one detected point,
true for most reconstructions here, but diverge when two detected
points both independently overlap the same ground-truth point
(confirmed for two of FreezerSmallTrain's five reconstructions). The
one-to-one version, which treats a second detected point matching an
already-claimed ground-truth point as a false positive rather than an
extra true positive, was confirmed as the correct match against the
paper's published numbers for exactly those two cases. See the
function's docstring in `src/metrics.py` for the full explanation.

## Minor note on TwoLeadECG

Reconstructions 1-3 match the exact detected chain indices reported
in the working session; reconstructions 4-5 produce different raw
indices but identical derived P/R/F1 scores, suggesting a minor
transcription difference in the originally-reported index lists for
those two reconstructions rather than a data or pipeline issue (the
metrics that actually populate Table 4 match exactly either way).

## Source

UCR Time Series Archive (Dau et al., 2018); reconstructed series and
ground-truth chain nodes derived from TSC'22 (Zhang et al., 2022)
chain outputs, per Section 5.2 of the paper.
