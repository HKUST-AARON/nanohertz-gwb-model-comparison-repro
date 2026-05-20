#!/usr/bin/env python3
"""Run unified SMBHB PL and SBPL analyses with the direct PTA likelihood."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
from enterprise_extensions import sampler as ee_sampler
from scipy.stats import norm
from sksparse.cholmod import CholmodNotPositiveDefiniteError

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pta_pipeline import (  # noqa: E402
    build_smbhb_pl,
    build_smbhb_sbpl,
    get_tspan_years,
    load_noise_dict,
    load_pulsars,
)


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0"),
        help="Path to the unpacked NANOGrav dataset (directory containing wideband/).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_outputs/smbhb_unified"),
        help="Output directory for samples and summaries.",
    )
    parser.add_argument(
        "--model",
        choices=["pl", "sbpl", "both"],
        default="both",
        help="Which model(s) to run.",
    )
    parser.add_argument(
        "--max-pulsars",
        type=int,
        default=None,
        help="Optional cap on the number of pulsars (useful for quick tests).",
    )
    parser.add_argument(
        "--nlive",
        type=int,
        default=1500,
        help="Number of live points for dynesty nested sampling.",
    )
    parser.add_argument(
        "--dlogz",
        type=float,
        default=0.1,
        help="Stopping criterion for dynesty (evidence tolerance).",
    )
    parser.add_argument(
        "--dynesty-walks",
        type=int,
        default=30,
        help="Number of random-walk steps per proposal in dynesty.",
    )
    parser.add_argument(
        "--ptmcmc-steps",
        type=int,
        default=500_000,
        help="Number of PTMCMC iterations per cold chain.",
    )
    parser.add_argument(
        "--burn",
        type=int,
        default=50_000,
        help="Burn-in iterations to discard from each PTMCMC chain.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=50,
        help="Thinning applied when writing PTMCMC chains to disk.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous runs if outputs exist.",
    )
    parser.add_argument(
        "--disable-dmdata",
        action="store_true",
        help="Disable DM data modelling (useful for debugging).",
    )
    return parser.parse_args()


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="a")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def prior_transform_factory(params: List, names: List[str]):
    """Return a callable that maps unit-cube samples to parameter space."""

    def transform(unit_cube: np.ndarray) -> np.ndarray:
        theta = np.zeros_like(unit_cube)
        for idx, (u, param) in enumerate(zip(unit_cube, params)):
            func_name = param.prior._func.__name__
            if func_name == "UniformPrior":
                bounds = param.prior._defaults
                theta[idx] = bounds["pmin"] + (bounds["pmax"] - bounds["pmin"]) * u
            elif func_name == "NormalPrior":
                defaults = param.prior._defaults
                theta[idx] = defaults["mu"] + defaults["sigma"] * norm.ppf(
                    np.clip(u, 1e-12, 1 - 1e-12)
                )
            else:
                raise ValueError(
                    f"Unsupported prior {func_name} for parameter {names[idx]}"
                )
        return theta

    return transform


def loglike_factory(pta) -> callable:
    names = pta.param_names

    def loglike(theta: np.ndarray) -> float:
        params = {name: val for name, val in zip(names, theta)}
        try:
            return pta.get_lnlikelihood(params)
        except (CholmodNotPositiveDefiniteError, np.linalg.LinAlgError, ValueError):
            return -np.inf

    return loglike


def run_dynesty(
    pta,
    outdir: Path,
    *,
    nlive: int,
    dlogz: float,
    walks: int,
    resume: bool,
) -> Dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    evidence_path = outdir / "evidence.json"
    results_path = outdir / "dynesty_results.npz"

    if evidence_path.exists() and not resume:
        raise FileExistsError(
            f"{evidence_path} exists. Use --resume to overwrite or continue."
        )

    ndim = len(pta.param_names)
    transform = prior_transform_factory(pta.params, pta.param_names)
    loglike = loglike_factory(pta)

    LOG.info("Generating initial live points")
    rng = np.random.default_rng(202503)
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

    center_u = np.full(ndim, 0.5)
    try_add(center_u.copy())

    attempts = 0
    max_attempts = 50000
    while len(live_u) < nlive and attempts < max_attempts:
        if len(live_u) < nlive // 4:
            u = np.clip(rng.normal(0.5, 0.02, size=ndim), 0.0, 1.0)
        elif len(live_u) < nlive // 2:
            u = np.clip(rng.normal(0.5, 0.05, size=ndim), 0.0, 1.0)
        else:
            u = rng.uniform(0.0, 1.0, size=ndim)
        try_add(u)
        attempts += 1

    if len(live_u) < nlive:
        raise RuntimeError(
            f"Failed to initialize live points: obtained {len(live_u)} of {nlive}"
        )

    live_u_arr = np.vstack(live_u)
    live_v_arr = np.vstack(live_v)
    live_logl_arr = np.array(live_logl)

    LOG.info(
        "Initialized %d/%d live points after %d attempts", len(live_u), nlive, attempts
    )

    LOG.info("Starting dynesty with nlive=%d, dlogz=%.3f", nlive, dlogz)
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
        live_points=(live_u_arr, live_v_arr, live_logl_arr),
    )
    start = time.time()
    sampler.run_nested(dlogz=dlogz, print_progress=False, resume=resume)
    runtime = time.time() - start
    results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])

    np.savez_compressed(
        results_path,
        samples=results.samples,
        logwt=results.logwt,
        weights=weights,
        logz=results.logz,
        logzerr=results.logzerr,
        information=results.information,
    )

    summary = {
        "logz": float(results.logz[-1]),
        "logzerr": float(results.logzerr[-1]),
        "information": float(results.information[-1]),
        "nlive": nlive,
        "dlogz": dlogz,
        "walks": walks,
        "runtime_seconds": runtime,
        "ncall": int(np.sum(np.atleast_1d(results.ncall))),
    }
    evidence_path.write_text(json.dumps(summary, indent=2))
    LOG.info(
        "Dynesty finished: logZ = %.3f ± %.3f (runtime %.1f min)",
        summary["logz"],
        summary["logzerr"],
        runtime / 60.0,
    )

    equal_samples = dyfunc.resample_equal(results.samples, weights)
    np.savez_compressed(
        outdir / "posterior_equal_samples.npz",
        samples=equal_samples,
        param_names=np.array(pta.param_names),
    )

    return summary


def run_ptmcmc(
    pta,
    outdir: Path,
    *,
    steps: int,
    burn: int,
    thin: int,
    resume: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sampler = ee_sampler.setup_sampler(
        pta,
        outdir=str(outdir),
        resume=resume,
        empirical_distr=None,
    )
    ndim = len(pta.param_names)

    def _draw_initial_vector():
        values: List[float] = []
        for param in pta.params:
            draw = param.sample()
            arr = np.atleast_1d(np.array(draw, dtype=float))
            values.extend(arr.flatten().tolist())
        return np.array(values)

    x0 = _draw_initial_vector()
    if x0.shape[0] != ndim:
        raise RuntimeError(
            f"Initial position mismatch: expected {ndim} elements, drew {x0.shape[0]}"
        )

    LOG.info(
        "Launching PTMCMC: steps=%d burn=%d thin=%d dim=%d",
        steps,
        burn,
        thin,
        ndim,
    )
    sampler.sample(
        x0,
        steps,
        SCAMweight=30,
        AMweight=30,
        DEweight=50,
        NUTSweight=0,
        MALAweight=0,
        HMCweight=0,
        covUpdate=1000,
        burn=burn,
        thin=thin,
    )
    LOG.info("PTMCMC complete; chains written to %s", outdir)


def summarise_posterior(equal_samples: np.ndarray, names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    quantiles = [0.16, 0.5, 0.84]
    for idx, name in enumerate(names):
        q16, q50, q84 = np.quantile(equal_samples[:, idx], quantiles)
        summary[name] = {
            "p16": float(q16),
            "median": float(q50),
            "p84": float(q84),
        }
    return summary


def run_model(
    label: str,
    pta,
    outdir: Path,
    *,
    nlive: int,
    dlogz: float,
    walks: int,
    ptmcmc_steps: int,
    burn: int,
    thin: int,
    resume: bool,
) -> None:
    model_dir = outdir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("===== Running model %s =====", label)

    dynesty_summary = run_dynesty(
        pta,
        model_dir,
        nlive=nlive,
        dlogz=dlogz,
        walks=walks,
        resume=resume,
    )

    equal_samples = np.load(model_dir / "posterior_equal_samples.npz")["samples"]
    param_names = pta.param_names
    posterior_summary = summarise_posterior(equal_samples, param_names)
    (model_dir / "posterior_summary.json").write_text(
        json.dumps(posterior_summary, indent=2)
    )

    if ptmcmc_steps > 0:
        run_ptmcmc(
            pta,
            model_dir / "ptmcmc",
            steps=ptmcmc_steps,
            burn=burn,
            thin=thin,
            resume=resume,
        )

    metadata = {
        "param_names": param_names,
        "nlive": nlive,
        "dynesty": dynesty_summary,
        "ptmcmc_steps": ptmcmc_steps,
        "burn": burn,
        "thin": thin,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    LOG.info("Model %s finished; outputs stored at %s", label, model_dir)


def main() -> None:
    args = parse_args()
    setup_logging(args.outdir)

    LOG.info("Loading pulsars from %s", args.dataset_root)
    pulsars, dmx_data = load_pulsars(
        args.dataset_root,
        max_pulsars=args.max_pulsars,
        return_dmx=True,
    )
    LOG.info("Effective Tspan = %.2f years", get_tspan_years(pulsars))

    LOG.info("Loading noise dictionary")
    noise_dict = load_noise_dict(args.dataset_root)

    models_to_run: List[Tuple[str, callable]] = []
    if args.model in {"pl", "both"}:
        models_to_run.append(("smbhb_pl", build_smbhb_pl))
    if args.model in {"sbpl", "both"}:
        models_to_run.append(("smbhb_sbpl", build_smbhb_sbpl))

    use_dmdata = not args.disable_dmdata

    for label, builder in models_to_run:
        pta = builder(
            pulsars,
            use_dmdata=use_dmdata,
            noise_dict=noise_dict,
            dmx_data=dmx_data,
        )
        run_model(
            label,
            pta,
            args.outdir,
            nlive=args.nlive,
            dlogz=args.dlogz,
            walks=args.dynesty_walks,
            ptmcmc_steps=args.ptmcmc_steps,
            burn=args.burn,
            thin=args.thin,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
