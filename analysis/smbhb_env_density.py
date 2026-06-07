#!/usr/bin/env python3
"""
Propagate SMBHB-env posterior samples to nuclear-density estimates.

Given draws of the bend frequency f_b from the SBPL posterior, we sample
plausible binary parameters and compute the stellar density rho_* implied by
equating stellar hardening and GW-driven inspiral timescales.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = REPO_ROOT / "analysis_outputs" / "kde_model_comparison" / "smbhb_env" / "posterior_samples.npz"
OUTPUT_DIR = REPO_ROOT / "analysis_outputs" / "smbhb_env"

# Physical constants (SI units)
G = 6.67430e-11
C = 2.99792458e8
MSUN = 1.98847e30
PC = 3.085677581491367e16
MSUN_PER_PC3_TO_SI = MSUN / (PC**3)


def draw_astrophysical_hyperparams(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample total mass, individual masses, velocity dispersion, and H."""
    log10_mtot = rng.normal(loc=9.3, scale=0.3, size=size)
    m_tot = (10.0 ** log10_mtot) * MSUN

    q = rng.uniform(0.25, 1.0, size=size)
    m1 = m_tot / (1.0 + q)
    m2 = m_tot - m1

    sigma = rng.normal(loc=200e3, scale=30e3, size=size)  # m/s
    sigma = np.clip(sigma, 50e3, None)

    H = rng.normal(loc=15.0, scale=5.0, size=size)
    H = np.clip(H, 5.0, None)

    return m1, m2, sigma, H


def compute_density(f_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = f_b.size
    m1, m2, sigma, H = draw_astrophysical_hyperparams(size, rng)
    m_tot = m1 + m2

    a = (G * m_tot / (np.pi**2 * f_b**2)) ** (1.0 / 3.0)

    numerator = 64.0 * (G**2) * m1 * m2 * m_tot * sigma
    denominator = 5.0 * (C**5) * H * (a**5)
    rho_si = numerator / denominator

    rho_msun_pc3 = rho_si / MSUN_PER_PC3_TO_SI
    return rho_msun_pc3


def summarize(samples: np.ndarray) -> dict[str, float]:
    median, lo, hi = np.percentile(samples, [50.0, 16.0, 84.0])
    log_samples = np.log10(samples)
    log_med, log_lo, log_hi = np.percentile(log_samples, [50.0, 16.0, 84.0])

    return {
        "median": float(median),
        "minus": float(median - lo),
        "plus": float(hi - median),
        "log10_median": float(log_med),
        "log10_minus": float(log_med - log_lo),
        "log10_plus": float(log_hi - log_med),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(SAMPLES_PATH)
    samples = data["samples"]
    if "param_names" in data:
        param_names = [name.decode() if isinstance(name, bytes) else str(name) for name in data["param_names"]]
        try:
            fb_index = param_names.index("log10_fb")
        except ValueError as exc:
            raise RuntimeError("log10_fb not found in posterior samples") from exc
    else:
        fb_index = 1

    rng = np.random.default_rng(20250215)

    if samples.shape[0] > 40000:
        idx = rng.choice(samples.shape[0], size=40000, replace=False)
        samples = samples[idx]

    log10_fb = samples[:, fb_index]
    f_b = 10.0 ** log10_fb

    rho_samples = compute_density(f_b, rng)

    np.savez(OUTPUT_DIR / "sbpl_density_samples.npz", rho=rho_samples)

    summary = summarize(rho_samples)
    summary["posterior_source"] = str(SAMPLES_PATH.relative_to(REPO_ROOT))
    with open(OUTPUT_DIR / "sbpl_density_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Saved density posterior samples to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
