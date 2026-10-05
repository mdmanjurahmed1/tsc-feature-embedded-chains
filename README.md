# Time Series Chain Discovery: A Feature-Based Embedded Approach

Code accompanying the paper *"Time Series Chain Discovery: A Feature-Based
Embedded Approach to Enhance Identifying Patterns in Time Series"*
(submitted to *Data Mining and Knowledge Discovery*).

## Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.12.

## Repository structure

```
src/                    Core framework (import as a package)
  preprocessing.py        Structural filter (Section 4.1)
  matrix_profile.py        Base chain discovery via STUMP/ALLC (Section 4.2)
  c22_features.py          C22 feature extraction and normalization (Section 4.3)
  kd_tree_augmentation.py  K-D Tree augmentation (Section 4.4)
  chain_discovery.py       Feature-based chain discovery + full pipeline (Section 4.5, Alg. 1)
  final_filtering.py       Final redundancy filtering (Section 4.6)
  plotting.py              Shared chain visualization utility

case_studies/            One folder per case study in Section 5.1
  web_query/                Web Query Volume (Fig. 10, 11)
  penguin/                  Penguin Activity (Fig. 12)
  tilt_table/                Tilt Table Data (Fig. 13–16)
  battery/                   Battery Discharge Capacity (new case study)
  constant_signals/          Constant / trivial signal robustness test (Fig. 18)

synthetic_benchmark/     Reproduces the synthetic benchmark evaluation (Section 5.2, Fig. 19, Table 2)
real_world_benchmark/    Reproduces the real-world UCR benchmark evaluation (Section 5.2, Table 4)
ablation/                Reproduces the ablation and efficiency study (Table 3)
figures/                  Scripts to regenerate all numbered paper figures
```

## Reproducing a case study

Each case study folder contains a `run_*.py` script that loads its
dataset (from a public URL or a cached local copy under `data/`), runs
the full pipeline via `src.chain_discovery.run_full_pipeline`, and saves
the resulting figure as a vector PDF.

```bash
cd case_studies/web_query
python run_web_query.py --m 25
python run_web_query.py --m 50
```

See each case study folder's own short README for dataset-specific
notes (source, license, and any caching details).

## Reproducing the synthetic benchmark evaluation (Fig. 19, Table 2)

```bash
cd synthetic_benchmark
python run_synthetic_benchmark.py
```

This runs the proposed method live across 101 noise levels (0% to
100%, 1% steps) on the synthetic ground-truth chain benchmark, and
verified to exactly reproduce the paper's published precision/recall/F1
values (cross-checked at 0%, 20%, and 100% noise against Table 2).

**Runtime note**: this involves 101 full pipeline runs (each including
22 separate per-feature Matrix Profile computations), so expect this to
take a substantial amount of time (tens of minutes) depending on your
machine.

TSC'17's F1 curve is an external baseline, not part of this codebase.
Its 101-point curve is cached in `data/tsc17_baseline_f1.csv`; all 11
values at 10%-increment noise levels were cross-checked against the
paper's published Table 2 and match exactly.

## Reproducing the real-world benchmark evaluation (Table 4)

```bash
cd real_world_benchmark
python run_real_world_benchmark.py                # all 7 datasets
python run_real_world_benchmark.py --dataset Wafer # a single dataset
```

**All 7 datasets are complete and live-verified**, exactly reproducing
every row of Table 4: Plane (P=0.98, R=0.94, F1=0.96), TwoLeadECG
(P=0.98, R=0.98, F1=0.98), ECG200 (P=0.60, R=0.54, F1=0.57), Wafer
(P=0.80, R=0.66, F1=0.72), FreezerRegularTrain (P=0.96, R=0.86,
F1=0.90), FreezerSmallTrain (P=0.93, R=0.84, F1=0.88), and TwoPatterns
(P=0.76, R=0.68, F1=0.71). See `data/README.md` for details,
including a note on the one-to-one matching metric
(`compute_performance_metrics_one_to_one` in `src/metrics.py`) this
module uses, which differs subtly from the metric used by
`synthetic_benchmark/` and `ablation/` and was required to exactly
match two of FreezerSmallTrain's five reconstructions.

## Reproducing the ablation study (Table 3)

```bash
cd ablation
python run_ablation.py                        # all 4 variants
python run_ablation.py --variant "w/o KD-Tree" # a single variant
```

This reproduces the F1-score, runtime, and peak-memory results across
101 noise levels (0% to 100%) for each ablation variant (Base only
(TSC'17-equivalent), w/o Feature-Based Chain, w/o K-D Tree, Full
Model), and was cross-checked against the paper's Table 3: the "Base
only" variant's F1 exactly reproduces TSC'17's own published values,
and "Full Model" exactly reproduces the proposed method's values from
Table 2.

All four variants use the same `discover_stump_chain` (with the
should_skip structural filter applied), matching the master ablation
notebook where all four variants are switchable blocks sharing one
`discover_stump_chain` definition. A separate standalone notebook
defines the Base-only variant without should_skip filtering, kept in
`src/matrix_profile.py` as `discover_stump_chain_raw` for fidelity to
that source; empirically the two are equivalent here, since
`should_skip` returns False at all 101 noise levels for the synthetic
benchmark series used throughout this repo.

**Runtime note**: `stumpy`'s Matrix Profile computation is
numba-JIT-compiled on its first call in a fresh process, which can
take tens of seconds. `run_ablation.py` runs an explicit warm-up call
before timing begins so this one-time cost doesn't contaminate the
first measurement; if you time individual pipeline calls yourself
outside this script, do the same.

## Parameters

All framework parameters (thresholds, k, delta, etc.) are documented in
Table 1 of the paper (Section 4.7) and set as function defaults
throughout `src/`, so no separate configuration file is required to
reproduce the reported results.

## Data availability

- UCR Time Series Archive subsets used in the real-world benchmark
  (Section 5.2, Table 4) are cached under `real_world_benchmark/data/`
  — see that folder's `README.md`.
- Penguin, tilt table, and Web Query datasets are loaded directly from
  their original public sources (see each case study's script); a
  cached local copy is also provided under `data/` in case the
  original host is unavailable.
- The battery discharge dataset (NASA PCoE, cell B0006) is cached
  under `case_studies/battery/data/`.

## Citation

If you use this code, please cite the paper (full citation to be added
upon acceptance).
