# dp-budget-accounting

Code and measurements for *Liquidating the Fund: Differential Privacy and the
Accounting of an Exhaustible Stock* (Pierrelouis). The paper classifies the
privacy-loss budget as an exhaustible stock and proposes a ledger, a written
allocation rule, and a reserve. This repository holds the clinical measurement
behind its Section 6 and Tables 2 and 3: a DP-SGD training sweep on PTB-XL with
the energy of every run metered in hardware, and a one-run canary audit of the
trained models.

The result the paper uses: training under the mechanism costs the same
electricity at every privacy budget (97.6% over the non-private baseline,
within one percentage point per unit ε), while test AUC rises from 0.551 at
ε = 0.25 to 0.787 at ε = 5.0. A physical account of the process therefore
records nothing of the budget drawn; the privacy accountant is the only record.

## Layout

```
preprocess_ptbxl.py      PTB-XL download -> data/ptbxl_raw_100hz.npz (run first)
paper_utils_dpcnn.py     PTB-XL loading, patient-disjoint folds, InceptionTime1D (GroupNorm),
                         DP-SGD via Opacus with the RDP accountant, training loop
paper_utils_energy.py    energy meter: NVML cumulative energy counter, RAPL package counter
                         with wraparound, idle baseline and subtraction, empirical-ε bounds
measure_idle.py          measures the idle power the sweep subtracts (writes idle_baseline.json)
run_full_sweep.py        the 33-model sweep (10 ε × 3 seeds + 3 baseline), shadow models,
                         loss-threshold / RMIA / LiRA attacks, assembled tables
run_canary_audit.py      one-run canary audit (Steinke, Nasr and Jagielski, 2023)
tools/run_paper_numbers.py   recomputes every number in the paper from results/sweep and
                         checks each against the printed value
tools/make_figures.py    Figures 1 and 2
results/sweep/           the measurement CSVs every number comes from (canonical run),
                         with idle_baseline.json and the audit provenance JSONs
results/paper_numbers/   Table 2, Table 3 and numbers.json as recomputed
results/figures/         the figures as submitted (600 dpi PNG and vector PDF)
```

## Configuration

| | |
|---|---|
| Dataset | PTB-XL v1.0.1, 12-lead ECG at 100 Hz, binary atrial fibrillation |
| Splits | Patient-disjoint `strat_fold`: folds 1–8 train, 9 validation, 10 test |
| Records per model | 8,709, a seeded half of the 17,418-record audit pool |
| Model | InceptionTime1D with GroupNorm (no BatchNorm, for per-sample gradients) |
| Training | 30 epochs, batch 256, learning rate 1e-3, class-weighted loss |
| Mechanism | DP-SGD (Opacus), clipping norm 1.0, RDP accountant, δ = 4.6e-5 |
| ε grid | 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, plus a non-private baseline |
| Seeds | 3 per setting, 33 target models |
| Energy | NVML `nvmlDeviceGetTotalEnergyConsumption` (GPU) plus RAPL `intel-rapl:0` (CPU package), polled at 1 s with wraparound accumulation; idle draw measured over 5 minutes and subtracted as power × wall time |
| Boundary | GPU and CPU package only. Memory, power-supply losses, cooling and embodied energy are outside it, so every figure is a lower bound |

Energy figures are specific to one machine. The claim is the flatness across ε
and the ratios between runs, not the absolute joules.

## Reproduce the paper's numbers from the included CSVs

No GPU needed.

```
pip install -r requirements-analysis.txt
python3 tools/run_paper_numbers.py
```

This recomputes Table 2, Table 3, the regression of overhead on ε (all thirty
runs and the ten grid means), the paired seed differences, the realized ε
totals, and the audit bounds, and checks each against the value printed in the
paper. Rounded values in the paper are computed from unrounded means, as the
table notes state.

## Re-run the measurement

Needs a CUDA GPU with the NVML energy counter (Volta or newer) and readable
RAPL counters (`/sys/class/powercap/intel-rapl:0/energy_uj`, usually via a
passwordless sudoers rule for `cat` on that path).

1. `pip install -r requirements.txt`
2. Download PTB-XL from PhysioNet (https://physionet.org/content/ptb-xl/) into
   `data/ptbxl/` and run `python3 preprocess_ptbxl.py`. It writes
   `data/ptbxl_raw_100hz.npz` and checks the record count, AF prevalence and
   patient disjointness on the way.
3. `python3 measure_idle.py` with the machine otherwise idle. Writes
   `idle_baseline.json`; the sweep refuses to run without it.
4. `python3 run_full_sweep.py --out-dir results/sweep_new` runs all four
   stages. `--stages targets` runs only the 33 metered target models. The
   stage list is one comma-separated token.
5. `python3 run_canary_audit.py --sweep results/sweep_new --canary-kind flipped --epochs 30`.
   Point `--sweep` at the directory that holds `target_mask.npy`. Without it
   the audit trains on the full 17,418-record pool, which is how the
   deposited `canary_summary_ep30.csv` was produced; the paper reports that
   arm's cost per record. The 100-epoch arm in `canary_summary_ep100.csv`
   was size-matched to the sweep's 8,709-record half.

A sweep must go into a fresh directory: training skips existing checkpoints
and writes no energy for cached rows.

## Data

PTB-XL is public under its PhysioNet license and is not redistributed here:
https://physionet.org/content/ptb-xl/ (doi:10.13026/kfzx-aw45). The audit
split is regenerated from its seed by `run_full_sweep.py` and saved as
`target_mask.npy` in the sweep directory.

## Cite

See `CITATION.cff`. Archived at Zenodo: DOI to be added on release.
