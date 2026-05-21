#!/usr/bin/env python3
"""Run staged HD full-timing likelihood evidence tests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from dynesty import NestedSampler
from scipy.stats import norm
from sksparse.cholmod import CholmodNotPositiveDefiniteError

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pta_pipeline import get_tspan_years, load_noise_dict, load_pulsars  # noqa: E402
from enterprise_extensions import models  # noqa: E402


LOG = logging.getLogger(__name__)


MODEL_CHOICES = ("hd_powerlaw", "hd_turnover", "hd_phase_bpl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--max-pulsars", type=int, default=8)
    parser.add_argument("--common-components", type=int, default=10)
    parser.add_argument("--red-components", type=int, default=10)
    parser.add_argument("--nlive", type=int, default=256)
    parser.add_argument("--dlogz", type=float, default=2.0)
    parser.add_argument("--walks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(outdir / "run.log", mode="a")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def configure_local_timing_files(dataset_root: Path) -> None:
    clock_dir = dataset_root / "clock"
    if clock_dir.exists():
        os.environ["PINT_CLOCK_OVERRIDE"] = str(clock_dir.resolve())
        de440 = clock_dir / "de440.bsp"
        if de440.exists():
            from pint.solar_system_ephemerides import load_kernel

            load_kernel("de440", path=str(de440))


def set_uniform_prior(pta, suffix: str, low: float, high: float) -> None:
    for param in pta.params:
        if param.name.endswith(suffix) and param.prior._func.__name__ == "UniformPrior":
            param.prior._defaults["pmin"] = low
            param.prior._defaults["pmax"] = high


def build_hd_model(model_name: str, pulsars, noise_dict: Dict[str, float], args: argparse.Namespace):
    common_kwargs = dict(
        psrs=pulsars,
        tm_marg=False,
        is_wideband=True,
        use_dmdata=False,
        dm_var=False,
        bayesephem=False,
        white_vary=False,
        noisedict=noise_dict,
        orf="hd",
        red_var=True,
        red_psd="powerlaw",
        red_components=args.red_components,
        common_components=args.common_components,
        common_logmin=-17.0,
        common_logmax=-14.0,
    )
    if model_name == "hd_powerlaw":
        pta = models.model_general(
            common_psd="powerlaw",
            gamma_common=13.0 / 3.0,
            **common_kwargs,
        )
    elif model_name == "hd_turnover":
        pta = models.model_general(
            common_psd="turnover",
            gamma_common=13.0 / 3.0,
            **common_kwargs,
        )
        set_uniform_prior(pta, "gw_hd_log10_fbend", -9.6, -7.2)
        set_uniform_prior(pta, "gw_hd_kappa", 0.0, 7.0)
    elif model_name == "hd_phase_bpl":
        pta = models.model_general(
            common_psd="broken_powerlaw",
            gamma_common=2.0,
            **common_kwargs,
        )
        set_uniform_prior(pta, "gw_hd_log10_fb", -9.6, -7.2)
        set_uniform_prior(pta, "gw_hd_delta", 7.0, 9.0)
        set_uniform_prior(pta, "gw_hd_kappa", 0.01, 0.5)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return pta


def prior_transform_factory(params: List, names: List[str]):
    def transform(unit_cube: np.ndarray) -> np.ndarray:
        theta = np.zeros_like(unit_cube)
        for idx, (u, param) in enumerate(zip(unit_cube, params)):
            prior_name = param.prior._func.__name__
            if prior_name == "UniformPrior":
                bounds = param.prior._defaults
                theta[idx] = bounds["pmin"] + (bounds["pmax"] - bounds["pmin"]) * u
            elif prior_name == "NormalPrior":
                defaults = param.prior._defaults
                theta[idx] = defaults["mu"] + defaults["sigma"] * norm.ppf(
                    np.clip(u, 1e-12, 1 - 1e-12)
                )
            else:
                raise ValueError(f"Unsupported prior {prior_name} for {names[idx]}")
        return theta

    return transform


def loglike_factory(pta):
    names = pta.param_names

    def loglike(theta: np.ndarray) -> float:
        params = dict(zip(names, theta))
        try:
            return pta.get_lnlikelihood(params)
        except (CholmodNotPositiveDefiniteError, np.linalg.LinAlgError, ValueError):
            return -np.inf

    return loglike


def initialize_live_points(transform, loglike, ndim: int, nlive: int, seed: int):
    rng = np.random.default_rng(seed)
    live_u: List[np.ndarray] = []
    live_v: List[np.ndarray] = []
    live_logl: List[float] = []

    def try_add(u: np.ndarray) -> bool:
        theta = transform(u)
        logl = loglike(theta)
        if np.isfinite(logl):
            live_u.append(u)
            live_v.append(theta)
            live_logl.append(logl)
            return True
        return False

    try_add(np.full(ndim, 0.5))
    attempts = 0
    while len(live_u) < nlive and attempts < 100000:
        scale = 0.03 if len(live_u) < nlive // 2 else None
        u = np.clip(rng.normal(0.5, scale, size=ndim), 0.0, 1.0) if scale else rng.uniform(0.0, 1.0, size=ndim)
        try_add(u)
        attempts += 1
    if len(live_u) < nlive:
        raise RuntimeError(f"Only initialized {len(live_u)} of {nlive} live points")
    return np.vstack(live_u), np.vstack(live_v), np.asarray(live_logl)


def run_dynesty(pta, outdir: Path, *, nlive: int, dlogz: float, walks: int, seed: int) -> Dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    ndim = len(pta.param_names)
    transform = prior_transform_factory(pta.params, pta.param_names)
    loglike = loglike_factory(pta)
    live_points = initialize_live_points(transform, loglike, ndim, nlive, seed)
    LOG.info("Initialized %d live points for %d parameters", nlive, ndim)

    sampler = NestedSampler(
        loglike,
        transform,
        ndim,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
        walks=walks,
        update_interval=10,
        first_update={"min_ncall": 500, "min_eff": 5.0},
        live_points=live_points,
    )
    start = time.time()
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    runtime = time.time() - start
    results = sampler.results
    weights = np.exp(results.logwt - results.logz[-1])

    np.savez_compressed(
        outdir / "dynesty_results.npz",
        samples=results.samples,
        logwt=results.logwt,
        weights=weights,
        logz=results.logz,
        logzerr=results.logzerr,
        information=results.information,
        param_names=np.asarray(pta.param_names),
    )
    summary = {
        "logz": float(results.logz[-1]),
        "logzerr": float(results.logzerr[-1]),
        "information": float(results.information[-1]),
        "nlive": nlive,
        "dlogz": dlogz,
        "walks": walks,
        "runtime_seconds": float(runtime),
        "ncall": int(np.sum(results.ncall)),
        "nparams": ndim,
    }
    (outdir / "evidence.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_metadata(outdir: Path, pulsars, args: argparse.Namespace) -> None:
    metadata = {
        "pulsars": [psr.name for psr in pulsars],
        "toas": [int(len(psr.toas)) for psr in pulsars],
        "Tspan_years": float(get_tspan_years(pulsars)),
        "common_components": args.common_components,
        "red_components": args.red_components,
        "orf": "hd",
        "tm_marg": False,
        "dm_var": False,
        "bayesephem": False,
        "models": args.models,
        "note": "Staged real-TOA HD full-timing evidence using enterprise_extensions.models.model_general.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    setup_logging(args.outdir)
    configure_local_timing_files(args.dataset_root)
    LOG.info("Loading NANOGrav pulsars from %s", args.dataset_root)
    pulsars = load_pulsars(args.dataset_root, max_pulsars=args.max_pulsars)
    noise_dict = load_noise_dict(args.dataset_root)
    write_metadata(args.outdir, pulsars, args)

    results: Dict[str, Dict[str, float]] = {}
    for idx, model_name in enumerate(args.models):
        model_out = args.outdir / model_name
        evidence_path = model_out / "evidence.json"
        if args.skip_existing and evidence_path.exists():
            LOG.info("Skipping existing %s", model_name)
            results[model_name] = json.loads(evidence_path.read_text())
            continue
        LOG.info("Building %s", model_name)
        pta = build_hd_model(model_name, pulsars, noise_dict, args)
        (model_out / "params.json").parent.mkdir(parents=True, exist_ok=True)
        (model_out / "params.json").write_text(json.dumps(pta.param_names, indent=2))
        LOG.info("Running %s with params: %s", model_name, pta.param_names)
        results[model_name] = run_dynesty(
            pta,
            model_out,
            nlive=args.nlive,
            dlogz=args.dlogz,
            walks=args.walks,
            seed=args.seed + idx,
        )
        LOG.info("SUMMARY %s %s", model_name, json.dumps(results[model_name], sort_keys=True))

    comparison = {"results": results}
    if "hd_powerlaw" in results:
        base = results["hd_powerlaw"]["logz"]
        for model_name, summary in results.items():
            if model_name != "hd_powerlaw":
                comparison[f"delta_logz_{model_name}_minus_hd_powerlaw"] = summary["logz"] - base
    (args.outdir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    LOG.info("Wrote %s", args.outdir / "comparison.json")


if __name__ == "__main__":
    main()
