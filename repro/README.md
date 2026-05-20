# Reproducibility package

This folder documents the commands used for the Symmetry manuscript draft.

## Environment

Create the conda environment from:

```bash
conda env create -f repro/environment.yml
conda activate pta
```

The run scripts use the active `python` by default. To force a specific
interpreter, set `PYTHON=/path/to/python` before invoking the script.

## Public data products

- Full NANOGrav 15-year timing data:
  `dataset/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0`
- Public HD free-spectrum KDE:
  `data_sources/NANOGrav15yr_KDE-FreeSpectra/30f_fs{hd}_ceffyl`

## Reproduce reported manuscript tables

```bash
bash repro/run_kde_model_comparison.sh
```

This regenerates:

- `analysis_outputs/kde_model_comparison/model_comparison.json`
- `analysis_outputs/kde_model_comparison/prior_sensitivity.json`
- per-model `evidence.json`
- per-model `posterior_summary.json`
- per-model posterior sample archives

The current manuscript tables are populated from those JSON files.

## Reproduce manuscript figures

```bash
cd symmetry_hk_special_issue/mdpi_formatted_full
python figures/make_figures.py
latexmk -pdf main.tex
```

The figure script reads the KDE comparison summaries and writes PDFs under
`symmetry_hk_special_issue/mdpi_formatted_full/figures/`.

## Full timing-likelihood pipeline

The full enterprise production pipeline is documented but was not completed in
this draft because the run is computationally expensive. A two-pulsar smoke test
has already been archived at `analysis_outputs/test_run_small/smbhb_pl`.

Run the smoke test:

```bash
bash repro/run_enterprise_smoke.sh
```

Run the production PL/SBPL analysis:

```bash
bash repro/run_enterprise_production.sh
```

Production outputs should be deposited with the public GitHub release and
Zenodo archive before journal submission if the manuscript is to claim full
timing-likelihood evidences.
