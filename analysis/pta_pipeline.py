#!/usr/bin/env python3
"""Utility helpers for building PTA models used in the unified analysis."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("PINT_LOG_LEVEL", "CRITICAL")
os.environ.setdefault("PINT_LOG_FILE", "false")

import numpy as np
from enterprise.pulsar import Pulsar
from enterprise.signals import deterministic_signals
from enterprise.signals.signal_base import PTA
from enterprise_extensions import chromatic as chrom
from enterprise_extensions import models


LOG = logging.getLogger(__name__)

DEFAULT_EPHEMERIS = "DE440"


def configure_runtime_verbosity() -> None:
    """Minimize verbose logging from third-party libraries."""
    os.environ["LOGURU_LEVEL"] = "ERROR"
    os.environ["PINT_LOG_LEVEL"] = "CRITICAL"
    os.environ["PINT_LOG_FILE"] = "false"


DMX_KEY_RE = re.compile(r"DMX(R[12])?_(\d+)")


def _parse_dmx_from_par(par_path: Path) -> Dict[str, Dict[str, float]]:
    """Extract DMX interval metadata from a TEMPO par file."""

    dmx_entries: Dict[str, Dict[str, float]] = defaultdict(dict)
    try:
        lines = par_path.read_text().splitlines()
    except OSError:
        return {}

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        key = tokens[0]
        match = DMX_KEY_RE.match(key)
        if not match:
            continue
        suffix = match.group(1)
        idx = match.group(2)
        entry = dmx_entries[idx]
        try:
            value = float(tokens[1])
        except ValueError:
            continue
        if suffix is None:
            entry["DMX_VAL"] = value
            if len(tokens) >= 4:
                try:
                    entry["DMX_ERR"] = abs(float(tokens[3]))
                except ValueError:
                    pass
        elif suffix == "R1":
            entry["DMX_R1"] = value
        elif suffix == "R2":
            entry["DMX_R2"] = value

    result: Dict[str, Dict[str, float]] = {}
    for idx, data in dmx_entries.items():
        if not data:
            continue
        key = f"DMX_{idx}"
        if "DMX_ERR" not in data or data["DMX_ERR"] <= 0.0:
            data["DMX_ERR"] = 1e-4
        result[key] = data
    return result


def load_pulsars(
    dataset_root: Path,
    *,
    max_pulsars: Optional[int] = None,
    timing_package: str = "pint",
    drop_backend_duplicates: bool = True,
    return_dmx: bool = False,
) -> List[Pulsar] | Tuple[List[Pulsar], Dict[str, Dict[str, Dict[str, float]]]]:
    """Load par/tim files from the given dataset root."""
    configure_runtime_verbosity()
    dataset_root = Path(dataset_root)
    par_dir = dataset_root / "wideband" / "par"
    tim_dir = dataset_root / "wideband" / "tim"
    if not par_dir.exists() or not tim_dir.exists():
        raise FileNotFoundError(f"Could not locate wideband par/tim in {dataset_root}")

    par_files = sorted(par_dir.glob("*.par"))
    if drop_backend_duplicates:
        orig_count = len(par_files)
        par_files = [
            path
            for path in par_files
            if ("ao" not in path.stem.lower()) and ("gbt" not in path.stem.lower())
        ]
        if len(par_files) != orig_count:
            LOG.info(
                "Filtered duplicate backend par files: %d -> %d",
                orig_count,
                len(par_files),
            )
    if max_pulsars is not None:
        par_files = par_files[:max_pulsars]

    dmx_master: Dict[str, Dict[str, Dict[str, float]]] = {}

    pulsars: List[Pulsar] = []
    for par_path in par_files:
        tim_path = tim_dir / (par_path.stem + ".tim")
        if not tim_path.exists():
            base = par_path.stem.split("_PINT")[0]
            candidates = sorted(tim_dir.glob(f"{base}*.tim"))
            if not candidates:
                LOG.warning("Missing .tim for %s; skipping", par_path.name)
                continue
            tim_path = candidates[0]
            LOG.debug("Using fallback tim %s for %s", tim_path.name, par_path.name)
        LOG.info("Loading %s", par_path.stem)
        psr = Pulsar(
            str(par_path),
            str(tim_path),
            ephem=DEFAULT_EPHEMERIS,
            timing_package=timing_package,
            drop_pint_litepint_warning=True,
        )
        pulsars.append(psr)

        if return_dmx:
            dmx_master[psr.name] = _parse_dmx_from_par(par_path)

    if not pulsars:
        raise RuntimeError("No pulsars loaded; check dataset path and files")
    LOG.info("Loaded %d pulsars", len(pulsars))
    if return_dmx:
        for psr in pulsars:
            dmx_master.setdefault(psr.name, {})
        return pulsars, dmx_master
    return pulsars


def _set_uniform_prior(parameter, low: float, high: float) -> None:
    parameter.prior._defaults["pmin"] = low
    parameter.prior._defaults["pmax"] = high


def _enforce_common_amplitude_prior(pta, low: float, high: float) -> None:
    for param in pta.params:
        if "gw_log10_A" in param.name and param.prior._func.__name__ == "UniformPrior":
            _set_uniform_prior(param, low, high)


def _enforce_sbpl_priors(pta, *, log10_fb_range: Iterable[float], delta_range: Iterable[float]) -> None:
    fb_min, fb_max = log10_fb_range
    delta_min, delta_max = delta_range
    for param in pta.params:
        if param.name.endswith("gw_log10_fbend") and param.prior._func.__name__ == "UniformPrior":
            _set_uniform_prior(param, fb_min, fb_max)
        if param.name.endswith("gw_kappa") and param.prior._func.__name__ == "UniformPrior":
            _set_uniform_prior(param, delta_min, delta_max)


def _build_smbhb_pta(
    pulsars: List[Pulsar],
    *,
    psd: str,
    n_rnfreqs: int,
    n_gwbfreqs: int,
    gamma_common: Optional[float],
    bayesephem: bool,
    white_vary: bool,
    use_dmdata: bool,
    is_wideband: bool,
    tm_marg: bool,
    noise_dict: Optional[Dict[str, float]],
    dmx_data: Dict[str, Dict[str, Dict[str, float]]],
) -> PTA:
    models_list = []
    for psr in pulsars:
        extra_sigs = None
        if bayesephem:
            extra_sigs = deterministic_signals.PhysicalEphemerisSignal(
                use_epoch_toas=True, model="setIII"
            )

        builder = models.model_singlepsr_noise(
            psr,
            psd=psd,
            gamma_val=gamma_common,
            components=n_rnfreqs,
            gw_components=n_gwbfreqs,
            is_wideband=is_wideband,
            use_dmdata=False,
            white_vary=white_vary,
            tm_marg=tm_marg,
            dm_var=True,
            dm_type="dmx",
            dmx_data=dmx_data,
            extra_sigs=extra_sigs,
            psr_model=True,
        )
        models_list.append(builder(psr))

    pta = PTA(models_list)
    if noise_dict:
        pta.set_default_params(noise_dict)
    return pta


def build_smbhb_pl(
    pulsars: List[Pulsar],
    *,
    n_rnfreqs: int = 30,
    n_gwbfreqs: int = 30,
    gamma_common: Optional[float] = 13.0 / 3.0,
    bayesephem: bool = True,
    white_vary: bool = False,
    use_dmdata: bool = True,
    is_wideband: bool = True,
    tm_marg: bool = True,
    noise_dict: Optional[Dict[str, float]] = None,
    dmx_data: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> PTA:
    """Construct the SMBHB power-law PTA model."""
    if dmx_data is None:
        dmx_data = {psr.name: {} for psr in pulsars}
    pta = _build_smbhb_pta(
        pulsars,
        psd="powerlaw",
        n_rnfreqs=n_rnfreqs,
        n_gwbfreqs=n_gwbfreqs,
        gamma_common=gamma_common,
        bayesephem=bayesephem,
        white_vary=white_vary,
        use_dmdata=use_dmdata,
        is_wideband=is_wideband,
        tm_marg=tm_marg,
        noise_dict=noise_dict,
        dmx_data=dmx_data,
    )
    _enforce_common_amplitude_prior(pta, -17.0, -14.0)
    return pta


def build_smbhb_sbpl(
    pulsars: List[Pulsar],
    *,
    n_rnfreqs: int = 30,
    n_gwbfreqs: int = 30,
    gamma_common: Optional[float] = 13.0 / 3.0,
    bayesephem: bool = True,
    white_vary: bool = False,
    use_dmdata: bool = True,
    is_wideband: bool = True,
    tm_marg: bool = True,
    log10_fb_range: Iterable[float] = (-9.4, -8.0),
    delta_range: Iterable[float] = (0.0, 4.0),
    noise_dict: Optional[Dict[str, float]] = None,
    dmx_data: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> PTA:
    """Construct the SMBHB SBPL PTA model with turnover priors."""
    if dmx_data is None:
        dmx_data = {psr.name: {} for psr in pulsars}
    pta = _build_smbhb_pta(
        pulsars,
        psd="turnover",
        n_rnfreqs=n_rnfreqs,
        n_gwbfreqs=n_gwbfreqs,
        gamma_common=gamma_common,
        bayesephem=bayesephem,
        white_vary=white_vary,
        use_dmdata=use_dmdata,
        is_wideband=is_wideband,
        tm_marg=tm_marg,
        noise_dict=noise_dict,
        dmx_data=dmx_data,
    )
    _enforce_common_amplitude_prior(pta, -17.0, -14.0)
    _enforce_sbpl_priors(pta, log10_fb_range=log10_fb_range, delta_range=delta_range)
    return pta


def get_tspan_years(pulsars: List[Pulsar]) -> float:
    """Return the maximum observation span in years."""
    tsecs = [psr.toas.max() - psr.toas.min() for psr in pulsars]
    return max(tsecs) / (365.25 * 24 * 3600)
def _coerce_float(value: float) -> float:
    """Return a plain Python float (for JSON serialization)."""

    return float(np.asarray(value))


def _parse_dmjump_params(par_dir: Path, psr_base: str) -> Dict[str, float]:
    """Extract DMJUMP values from the corresponding wideband timing model."""

    par_candidates = sorted(par_dir.glob(f"{psr_base}*.wb.par"))
    if not par_candidates:
        LOG.warning("No wideband par file found for %s in %s", psr_base, par_dir)
        return {}
    if len(par_candidates) > 1:
        LOG.debug(
            "Multiple par files match %s; using %s", psr_base, par_candidates[0].name
        )

    par_path = par_candidates[0]
    dm_values: Dict[str, float] = {}
    with par_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = shlex.split(stripped, comments=True)
            if not tokens or tokens[0].upper() != "DMJUMP":
                continue
            fe_label: Optional[str] = None
            value: Optional[float] = None
            idx = 1
            while idx < len(tokens):
                token = tokens[idx]
                if token == "-fe" and idx + 1 < len(tokens):
                    fe_label = tokens[idx + 1]
                    idx += 2
                    continue
                if token.startswith("-"):
                    idx += 1
                    continue
                try:
                    value = float(token)
                    break
                except ValueError:
                    idx += 1
            if fe_label is None:
                LOG.debug("Skipping DMJUMP without -fe flag in %s", par_path)
                continue
            if value is None:
                LOG.debug("Skipping DMJUMP without numeric value in %s", par_path)
                continue
            key = f"{psr_base}_{fe_label}_dmjump"
            dm_values[key] = _coerce_float(value)

    return dm_values


def _load_noise_chains(noise_dir: Path, burn_fraction: float = 0.25) -> Dict[str, float]:
    """Parse wideband noise chain files into a dictionary."""

    if not noise_dir.exists():
        raise FileNotFoundError(f"Noise directory not found: {noise_dir}")

    noise_dict: Dict[str, float] = {}
    pars_files = sorted(noise_dir.glob("*.pars.txt"))
    if not pars_files:
        raise RuntimeError(f"No *.pars.txt files in {noise_dir}")

    par_dir = noise_dir.parent / "par"

    for pars_path in pars_files:
        param_names = [line.strip() for line in pars_path.read_text().splitlines() if line.strip()]
        if not param_names:
            continue

        chain_glob = pars_path.name.replace(".pars.txt", ".chain_*.txt")
        chain_paths = sorted(noise_dir.glob(chain_glob))
        if not chain_paths:
            raise FileNotFoundError(f"No chain files matching {chain_glob} in {noise_dir}")

        samples_list: List[np.ndarray] = []
        for chain_path in chain_paths:
            try:
                chain_data = np.loadtxt(chain_path, usecols=range(len(param_names)))
            except OSError as exc:  # pragma: no cover - pass-through for missing files
                raise FileNotFoundError(f"Failed to read {chain_path}") from exc

            if chain_data.ndim == 1:
                chain_data = chain_data[np.newaxis, :]

            if chain_data.size == 0:
                continue

            burn = int(chain_data.shape[0] * burn_fraction)
            trimmed = chain_data[burn:] if burn < chain_data.shape[0] else chain_data
            trimmed = trimmed[np.all(np.isfinite(trimmed), axis=1)]
            if trimmed.size == 0:
                continue
            samples_list.append(trimmed)

        if not samples_list:
            raise RuntimeError(f"No valid samples found for {pars_path.name}")

        combined = np.vstack(samples_list)
        medians = np.median(combined, axis=0)
        for name, value in zip(param_names, medians):
            noise_dict[name] = _coerce_float(value)

        psr_base = pars_path.name.split(".wb.pars.txt")[0]
        dm_values = _parse_dmjump_params(par_dir, psr_base)
        for key, value in dm_values.items():
            noise_dict.setdefault(key, value)

    LOG.info("Parsed %d noise parameters from %s", len(noise_dict), noise_dir)
    return noise_dict


def load_noise_dict(
    dataset_root: Path,
    *,
    noise_json: Optional[Path] = None,
    burn_fraction: float = 0.25,
) -> Dict[str, float]:
    """Load or construct the wideband noise dictionary for the dataset."""

    dataset_root = Path(dataset_root)
    if noise_json is None:
        noise_json = dataset_root / "wideband_noise.json"

    if noise_json.exists():
        LOG.info("Loading noise dictionary from %s", noise_json)
        return json.loads(noise_json.read_text())

    noise_dir = dataset_root / "wideband" / "noise"
    noise_dict = _load_noise_chains(noise_dir, burn_fraction=burn_fraction)

    try:
        noise_json.write_text(json.dumps(noise_dict, indent=2))
        LOG.info("Wrote noise dictionary to %s", noise_json)
    except OSError:
        LOG.warning("Failed to write noise dictionary to %s; continuing with in-memory copy", noise_json)

    return noise_dict
