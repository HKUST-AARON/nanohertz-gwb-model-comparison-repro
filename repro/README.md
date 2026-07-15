# Reproducibility workflow

This repository contains the public data inputs, numerical outputs, and scripts
needed to reproduce the Bayesian model-comparison tables reported in the
published Symmetry article (DOI: `10.3390/sym18071169`). The published Version
of Record is available separately at
[`paper/Xu_Zhang_Guo_2026_Symmetry_VOR.pdf`](../paper/Xu_Zhang_Guo_2026_Symmetry_VOR.pdf).

## Public archive

- GitHub repository:
  <https://github.com/HKUST-AARON/nanohertz-gwb-model-comparison-repro>
- Published article:
  <https://doi.org/10.3390/sym18071169>
- Latest data-only GitHub release:
  <https://github.com/HKUST-AARON/nanohertz-gwb-model-comparison-repro/releases/tag/v1.0.5-data-only>
- Zenodo all-versions DOI:
  <https://doi.org/10.5281/zenodo.20319210>

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
  Zenodo DOI `10.5281/zenodo.16051178`
- Public HD free-spectrum KDE:
  `data_sources/NANOGrav15yr_KDE-FreeSpectra_v1.1.0/ceffyl_data/30f_fs{hd}_ceffyl`

The KDE input is the corrected NANOGrav `NANOGrav15yr_KDE-FreeSpectra_v1.1.0`
Zenodo archive (DOI: `10.5281/zenodo.10344086`). The timing-data release is
available from NANOGrav/Zenodo.

## Reproduce reported analysis outputs

```bash
bash repro/run_kde_model_comparison.sh
python analysis/smbhb_env_density.py
```

This regenerates:

- `analysis_outputs/kde_model_comparison/model_comparison.json`
- `analysis_outputs/kde_model_comparison/prior_sensitivity.json`
- per-model `evidence.json`
- per-model `posterior_summary.json`
- per-model posterior sample archives
- `analysis_outputs/smbhb_env/sbpl_density_summary.json`
- `analysis_outputs/smbhb_env/sbpl_density_samples.npz`

The published article tables are populated from those JSON files.

## Direct timing-likelihood extension

The evidence values reported in the article are the public-KDE values above. The direct
enterprise pipeline is included as the timing-residual extension of the same
source models, and a two-pulsar smoke test has already been archived at
`analysis_outputs/test_run_small/smbhb_pl`.

Run the smoke test:

```bash
bash repro/run_enterprise_smoke.sh
```

Run the production PL/SBPL analysis:

```bash
bash repro/run_enterprise_production.sh
```

Production outputs should be deposited with the public GitHub release and
Zenodo archive before they are cited as full timing-likelihood evidences.
