#!/usr/bin/env python3
"""Fit source spectra to the NANOGrav 15-year free-spectrum KDE summaries.

This is a reproducible spectral-model comparison, not a replacement for the
full enterprise timing-residual likelihood.  It uses the public HD free-spectrum
KDE products as a compact likelihood for the strain spectrum and writes all
numbers used by the manuscript tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

FYR = 1.0 / (365.25 * 24.0 * 3600.0)
H0 = 67.4 * 1000.0 / 3.085677581491367e22
LOG10_12PI2 = math.log10(12.0 * math.pi**2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kde-dir",
        type=Path,
        default=Path("data_sources/NANOGrav15yr_KDE-FreeSpectra/30f_fs{hd}_ceffyl"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_outputs/kde_model_comparison"),
    )
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--posterior-samples", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def load_kde(kde_dir: Path) -> Dict[str, np.ndarray]:
    freqs = np.load(kde_dir / "freqs.npy")
    grid = np.load(kde_dir / "log10rhogrid.npy")
    log_density = np.load(kde_dir / "density.npy")[0]
    floor = float(np.nanmin(log_density))
    return {
        "freqs": freqs,
        "grid": grid,
        "log_density": log_density,
        "floor": np.array(floor),
    }


def log10_rho_from_hc(freqs: np.ndarray, hc: np.ndarray) -> np.ndarray:
    df = np.min(np.atleast_1d(freqs))
    log_phi = 2.0 * np.log10(hc) - LOG10_12PI2 - 3.0 * np.log10(freqs) + math.log10(df)
    return 0.5 * log_phi


def log10_rho_powerlaw(freqs: np.ndarray, log10_a: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    df = float(np.min(freqs))
    log_phi = (
        2.0 * log10_a[:, None]
        - LOG10_12PI2
        + (gamma[:, None] - 3.0) * math.log10(FYR)
        - gamma[:, None] * np.log10(freqs[None, :])
        + math.log10(df)
    )
    return 0.5 * log_phi


def log10_rho_sbpl(
    freqs: np.ndarray,
    log10_a: np.ndarray,
    log10_fb: np.ndarray,
    delta_gamma: np.ndarray,
    zeta: float = 1.0,
) -> np.ndarray:
    base = log10_rho_powerlaw(freqs, log10_a, np.full_like(log10_a, 13.0 / 3.0))
    fb = 10.0 ** log10_fb
    turn = -0.5 * zeta * np.log10(1.0 + (fb[:, None] / freqs[None, :]) ** (delta_gamma[:, None] / zeta))
    return base + turn


def log10_rho_phase_transition(
    freqs: np.ndarray,
    log10_omega_peak: np.ndarray,
    log10_f_peak: np.ndarray,
    b_high: np.ndarray,
    a_low: float = 3.0,
) -> np.ndarray:
    x = freqs[None, :] / (10.0 ** log10_f_peak[:, None])
    omega_peak = 10.0 ** log10_omega_peak[:, None]
    shape = ((a_low + b_high[:, None]) * x**a_low) / (
        b_high[:, None] + a_low * x ** (a_low + b_high[:, None])
    )
    omega = omega_peak * shape
    hc = np.sqrt(3.0 * H0**2 / (2.0 * math.pi**2)) * np.sqrt(omega) / freqs[None, :]
    return log10_rho_from_hc(freqs[None, :], hc)


def evaluate_loglike(kde: Dict[str, np.ndarray], model_log10rho: np.ndarray) -> np.ndarray:
    grid = kde["grid"]
    density = kde["log_density"]
    floor = float(kde["floor"])
    out = np.zeros(model_log10rho.shape[0])
    for idx in range(model_log10rho.shape[1]):
        out += np.interp(
            model_log10rho[:, idx],
            grid,
            density[idx],
            left=floor,
            right=floor,
        )
    return out


def logmeanexp(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + math.log(float(np.mean(np.exp(values - vmax))))


def block_logz_error(values: np.ndarray, blocks: int = 20) -> float:
    chunks = np.array_split(values, blocks)
    estimates = np.array([logmeanexp(chunk) for chunk in chunks if len(chunk)])
    if estimates.size <= 1:
        return float("nan")
    return float(np.std(estimates, ddof=1) / math.sqrt(estimates.size))


def weighted_resample(
    rng: np.random.Generator,
    params: np.ndarray,
    loglike: np.ndarray,
    size: int,
) -> np.ndarray:
    shifted = loglike - np.max(loglike)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    take = rng.choice(params.shape[0], size=min(size, params.shape[0]), replace=True, p=weights)
    return params[take]


def summarize(samples: np.ndarray, names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, idx], [16.0, 50.0, 84.0])
        summary[name] = {
            "p16": float(q16),
            "median": float(q50),
            "p84": float(q84),
            "minus": float(q50 - q16),
            "plus": float(q84 - q50),
        }
    return summary


def run_model(
    label: str,
    rng: np.random.Generator,
    kde: Dict[str, np.ndarray],
    outdir: Path,
    prior: Callable[[np.random.Generator, int], Tuple[np.ndarray, Tuple[str, ...]]],
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    nsamples: int,
    posterior_size: int,
) -> Dict[str, float]:
    model_dir = outdir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    params, names = prior(rng, nsamples)
    log10rho = model(kde["freqs"], params)
    loglike = evaluate_loglike(kde, log10rho)
    logz = logmeanexp(loglike)
    logzerr = block_logz_error(loglike)
    posterior = weighted_resample(rng, params, loglike, posterior_size)

    np.savez_compressed(
        model_dir / "posterior_samples.npz",
        samples=posterior,
        param_names=np.array(names),
    )
    evidence = {
        "logz": float(logz),
        "logzerr": float(logzerr),
        "n_prior_samples": int(nsamples),
        "likelihood": "sum of per-frequency public HD free-spectrum KDE log densities",
    }
    (model_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))
    (model_dir / "posterior_summary.json").write_text(
        json.dumps(summarize(posterior, names), indent=2)
    )
    return evidence


def make_priors(overrides: Dict[str, Tuple[float, float]] | None = None):
    overrides = overrides or {}

    def pl_fixed(rng: np.random.Generator, n: int):
        bounds = overrides.get("log10_a", (-17.0, -14.0))
        p = rng.uniform(bounds[0], bounds[1], size=(n, 1))
        return p, ("log10_A",)

    def sbpl(rng: np.random.Generator, n: int):
        a = rng.uniform(-17.0, -14.0, size=n)
        fb_min, fb_max = overrides.get("log10_fb", (-9.4, -8.0))
        fb = rng.uniform(fb_min, fb_max, size=n)
        dg = rng.uniform(0.0, 4.0, size=n)
        return np.column_stack([a, fb, dg]), ("log10_A", "log10_fb", "Delta_gamma")

    def cosmic_strings(rng: np.random.Generator, n: int):
        a = rng.uniform(-17.0, -14.0, size=n)
        beta_min, beta_max = overrides.get("beta", (-1.0, -0.5))
        beta = rng.uniform(beta_min, beta_max, size=n)
        return np.column_stack([a, beta]), ("log10_A", "beta_hc")

    def phase_transition(rng: np.random.Generator, n: int):
        omega = rng.uniform(-12.0, -5.0, size=n)
        fp_min, fp_max = overrides.get("log10_f_peak", (-9.0, -7.0))
        fp = rng.uniform(fp_min, fp_max, size=n)
        b = rng.uniform(2.0, 4.0, size=n)
        return np.column_stack([omega, fp, b]), ("log10_Omega_peak", "log10_f_peak", "b_high")

    return pl_fixed, sbpl, cosmic_strings, phase_transition


def models_from_priors(priors):
    pl_fixed, sbpl, cosmic_strings, phase_transition = priors
    return {
        "smbhb_pl": (
            pl_fixed,
            lambda freqs, p: log10_rho_powerlaw(freqs, p[:, 0], np.full(p.shape[0], 13.0 / 3.0)),
        ),
        "smbhb_env": (
            sbpl,
            lambda freqs, p: log10_rho_sbpl(freqs, p[:, 0], p[:, 1], p[:, 2]),
        ),
        "cosmic_strings": (
            cosmic_strings,
            lambda freqs, p: log10_rho_powerlaw(freqs, p[:, 0], 3.0 - 2.0 * p[:, 1]),
        ),
        "phase_transition": (
            phase_transition,
            lambda freqs, p: log10_rho_phase_transition(freqs, p[:, 0], p[:, 1], p[:, 2]),
        ),
    }


def write_comparisons(outdir: Path, evidences: Dict[str, Dict[str, float]]) -> None:
    base = evidences["smbhb_pl"]["logz"]
    base_err = evidences["smbhb_pl"]["logzerr"]
    rows = {}
    for label, ev in evidences.items():
        delta = ev["logz"] - base
        err = math.sqrt(ev["logzerr"] ** 2 + base_err**2)
        rows[label] = {
            "delta_logz_vs_smbhb_pl": float(delta),
            "error": float(err),
            "bayes_factor_vs_smbhb_pl": float(math.exp(delta)) if delta < 700 else float("inf"),
        }
    (outdir / "model_comparison.json").write_text(json.dumps(rows, indent=2))


def run_prior_sensitivity(args: argparse.Namespace, kde: Dict[str, np.ndarray], outdir: Path) -> None:
    scans = {
        "smbhb_env_log10_fb_-10.0_-8.0": {"log10_fb": (-10.0, -8.0)},
        "smbhb_env_log10_fb_-9.0_-8.3": {"log10_fb": (-9.0, -8.3)},
        "cosmic_strings_beta_-1.2_-0.3": {"beta": (-1.2, -0.3)},
        "phase_transition_log10_f_peak_-10.0_-6.0": {"log10_f_peak": (-10.0, -6.0)},
    }
    baseline = json.loads((outdir / "model_comparison.json").read_text())
    rng = np.random.default_rng(args.seed + 1000)
    results = {}
    for label, override in scans.items():
        priors = make_priors(override)
        model_map = models_from_priors(priors)
        if label.startswith("smbhb_env"):
            model_label = "smbhb_env"
        elif label.startswith("cosmic_strings"):
            model_label = "cosmic_strings"
        else:
            model_label = "phase_transition"
        prior, model = model_map[model_label]
        ev = run_model(
            label,
            rng,
            kde,
            outdir / "prior_sensitivity_runs",
            prior,
            model,
            max(args.samples // 2, 50_000),
            min(args.posterior_samples, 10_000),
        )
        base_logz = json.loads((outdir / "smbhb_pl" / "evidence.json").read_text())["logz"]
        results[label] = {
            "delta_logz_vs_smbhb_pl": float(ev["logz"] - base_logz),
            "baseline_delta_logz": baseline[model_label]["delta_logz_vs_smbhb_pl"],
            "override": override,
        }
    (outdir / "prior_sensitivity.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    kde = load_kde(args.kde_dir)
    rng = np.random.default_rng(args.seed)
    evidences: Dict[str, Dict[str, float]] = {}
    for label, (prior, model) in models_from_priors(make_priors()).items():
        evidences[label] = run_model(
            label,
            rng,
            kde,
            args.outdir,
            prior,
            model,
            args.samples,
            args.posterior_samples,
        )
    write_comparisons(args.outdir, evidences)
    run_prior_sensitivity(args, kde, args.outdir)
    manifest = {
        "kde_dir": str(args.kde_dir),
        "seed": args.seed,
        "samples": args.samples,
        "posterior_samples": args.posterior_samples,
        "frequency_count": int(kde["freqs"].size),
        "grid_min": float(kde["grid"].min()),
        "grid_max": float(kde["grid"].max()),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
