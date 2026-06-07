"""Appendix-only demonstrator for a simplified Hellings--Downs cross-check.

This script reproduces a pedagogical correlation plot using the public
NANOGrav 15-year wideband data set. It is intended for sanity checks and
appendix-style figures only; no significance statements in the main text
rely on its outputs.
"""
# Appendix-only demonstrator; not used for any HD significance claims.
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pint.models import get_model
import pint.toa as toa
from pint.residuals import Residuals
import astropy.units as u
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
from loguru import logger
import sys

# Reduce noisy logging output from PINT/Loguru.
os.environ.setdefault("PINT_LOG_LEVEL", "ERROR")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
logger.remove()
logger.add(sys.stderr, level="ERROR")

BIN_WIDTH_DAYS = 30.0
POLY_ORDER = 2
MIN_POINTS_PER_BIN = 2
MIN_OVERLAP = 5


@dataclass
class PulsarSeries:
    name: str
    mjd: np.ndarray
    residual: np.ndarray
    toa_err: np.ndarray
    ra_rad: float
    dec_rad: float
    binned_residual: np.ndarray | None = None
    binned_error: np.ndarray | None = None


@dataclass
class PairCorrelation:
    pulsar_i: str
    pulsar_j: str
    angle_deg: float
    hd_gamma: float
    corr: float
    sigma: float
    n_overlap: int


def hellings_downs(theta_rad: np.ndarray) -> np.ndarray:
    """Return the Hellings--Downs correlation for angle theta."""
    cos_theta = np.cos(theta_rad)
    x = (1.0 - cos_theta) / 2.0
    # Avoid log(0) at zero separation.
    x = np.clip(x, 1e-12, None)
    hd = 1.5 * x * np.log(x) - 0.25 * x + 0.5
    # For zero separation the limiting value is 1.
    if np.isscalar(theta_rad):
        return float(hd)
    return hd


def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Return angular separation in radians between two sky positions."""
    cos_sep = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cos_sep = min(1.0, max(-1.0, cos_sep))
    return math.acos(cos_sep)


def bin_residuals(
    series: PulsarSeries, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bin residuals using inverse-variance weighting."""
    nbins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    binned = np.full(nbins, np.nan)
    err = np.full(nbins, np.nan)
    indices = np.digitize(series.mjd, bin_edges) - 1
    for k in range(nbins):
        mask = indices == k
        if mask.sum() < MIN_POINTS_PER_BIN:
            continue
        w = 1.0 / np.square(series.toa_err[mask])
        wsum = w.sum()
        if wsum <= 0:
            continue
        mean = np.sum(w * series.residual[mask]) / wsum
        binned[k] = mean
        err[k] = math.sqrt(1.0 / wsum)
    return binned, err


def detrend_binned(
    binned: np.ndarray, errors: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Remove a low-order polynomial trend from the binned residuals."""
    cleaned = binned.copy()
    mask = (~np.isnan(binned)) & (~np.isnan(errors)) & (errors > 0)
    if mask.sum() <= POLY_ORDER + 2:
        return cleaned
    t = centers[mask]
    t0 = np.mean(t)
    x = t - t0
    w = 1.0 / np.square(errors[mask])
    coeffs = np.polynomial.polynomial.polyfit(x, binned[mask], POLY_ORDER, w=w)
    trend = np.polynomial.polynomial.polyval(x, coeffs)
    cleaned[mask] = binned[mask] - trend
    return cleaned


def compute_pair_correlation(
    series_i: PulsarSeries,
    series_j: PulsarSeries,
    centers: np.ndarray,
) -> PairCorrelation | None:
    ri = series_i.binned_residual
    rj = series_j.binned_residual
    ei = series_i.binned_error
    ej = series_j.binned_error
    if ri is None or rj is None or ei is None or ej is None:
        return None
    mask = (
        (~np.isnan(ri))
        & (~np.isnan(rj))
        & (~np.isnan(ei))
        & (~np.isnan(ej))
        & (ei > 0)
        & (ej > 0)
    )
    if mask.sum() < MIN_OVERLAP:
        return None
    ri_sel = ri[mask]
    rj_sel = rj[mask]
    wi = 1.0 / np.square(ei[mask])
    wj = 1.0 / np.square(ej[mask])
    w = np.sqrt(wi * wj)
    # Subtract weighted means to avoid bias from offsets.
    ri_sel = ri_sel - np.average(ri_sel, weights=w)
    rj_sel = rj_sel - np.average(rj_sel, weights=w)
    wsum = w.sum()
    cov = np.sum(w * ri_sel * rj_sel) / wsum
    vi = np.sum(w * ri_sel**2) / wsum
    vj = np.sum(w * rj_sel**2) / wsum
    if vi <= 0 or vj <= 0:
        return None
    corr = cov / math.sqrt(vi * vj)
    n_eff = mask.sum()
    sigma = (1.0 - corr**2) / math.sqrt(max(n_eff - 3, 1))
    theta = angular_separation(
        series_i.ra_rad, series_i.dec_rad, series_j.ra_rad, series_j.dec_rad
    )
    hd = float(hellings_downs(theta))
    return PairCorrelation(
        pulsar_i=series_i.name,
        pulsar_j=series_j.name,
        angle_deg=math.degrees(theta),
        hd_gamma=hd,
        corr=corr,
        sigma=sigma,
        n_overlap=n_eff,
    )


def load_series(par_path: Path, tim_path: Path) -> PulsarSeries:
    model = get_model(str(par_path))
    toas = toa.get_TOAs(str(tim_path), planets=True, usepickle=True)
    res = Residuals(toas, model).time_resids.to(u.s).value
    errors = toas.get_errors().to(u.s).value
    mjd = toas.get_mjds().value
    if hasattr(model, "RAJ"):
        ra = model.RAJ.quantity.to(u.rad).value
        dec = model.DECJ.quantity.to(u.rad).value
    elif hasattr(model, "ELONG"):
        coord = SkyCoord(
            lon=model.ELONG.quantity,
            lat=model.ELAT.quantity,
            frame=BarycentricTrueEcliptic(),
        )
        icrs = coord.icrs
        ra = icrs.ra.to(u.rad).value
        dec = icrs.dec.to(u.rad).value
    else:
        raise AttributeError("Astrometric coordinates not found in par file")
    name = par_path.name.split("_", 1)[0]
    return PulsarSeries(name=name, mjd=mjd, residual=res, toa_err=errors, ra_rad=ra, dec_rad=dec)


def select_pulsars(par_dir: Path, limit: int | None = None) -> List[Path]:
    pars = sorted(par_dir.glob("*.par"))
    if limit is not None:
        pars = pars[:limit]
    return pars


def run(dataset_root: Path, outdir: Path, max_pulsars: int | None) -> None:
    logging.info("Loading wideband timing models from %s", dataset_root)
    par_dir = dataset_root / "wideband" / "par"
    tim_dir = dataset_root / "wideband" / "tim"
    selected = select_pulsars(par_dir, max_pulsars)
    pulsars: List[PulsarSeries] = []
    for par_path in selected:
        tim_path = tim_dir / (par_path.stem + ".tim")
        if not tim_path.exists():
            logging.warning("Missing tim file for %s", par_path.name)
            continue
        try:
            series = load_series(par_path, tim_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping %s due to %s", par_path.name, exc)
            continue
        pulsars.append(series)
        logging.info("Loaded %s with %d TOAs", series.name, len(series.mjd))
    if len(pulsars) < 4:
        raise RuntimeError("Not enough pulsars to form correlations")

    min_mjd = min(float(p.mjd.min()) for p in pulsars)
    max_mjd = max(float(p.mjd.max()) for p in pulsars)
    bin_edges = np.arange(math.floor(min_mjd), math.ceil(max_mjd) + BIN_WIDTH_DAYS, BIN_WIDTH_DAYS)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    for p in pulsars:
        binned, err = bin_residuals(p, bin_edges)
        p.binned_error = err
        p.binned_residual = detrend_binned(binned, err, centers)

    pairs: List[PairCorrelation] = []
    for i in range(len(pulsars)):
        for j in range(i + 1, len(pulsars)):
            pcorr = compute_pair_correlation(pulsars[i], pulsars[j], centers)
            if pcorr:
                pairs.append(pcorr)
    if not pairs:
        raise RuntimeError("No pulsar pairs passed quality cuts")

    df = pd.DataFrame([p.__dict__ for p in pairs])
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "hd_pair_correlations.csv"
    df.to_csv(csv_path, index=False)

    weights = df["n_overlap"].values.astype(float)
    gamma = df["hd_gamma"].values
    corr = df["corr"].values
    amp_den = np.sum(weights * gamma * gamma)
    amp = np.sum(weights * gamma * corr) / amp_den
    resid = corr - amp * gamma
    amp_var = np.sum(weights * resid * resid) / ((len(df) - 1) * amp_den)
    amp_err = math.sqrt(max(amp_var, 0.0))

    summary = {
        "n_pulsars": len(pulsars),
        "n_pairs": len(pairs),
        "hd_scale": amp,
        "hd_scale_unc": amp_err,
        "bin_width_days": BIN_WIDTH_DAYS,
        "poly_order": POLY_ORDER,
    }
    with open(outdir / "hd_summary.json", "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sc = ax.scatter(
        df["angle_deg"],
        df["corr"],
        c=df["n_overlap"],
        s=20 + 3 * df["n_overlap"],
        cmap="viridis",
        alpha=0.75,
        linewidths=0.0,
    )
    cbar = fig.colorbar(sc, ax=ax, label="Overlapping bins")

    # Angle-binned averages for readability.
    angle_bins = np.linspace(0, 180, 13)
    bin_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    mean_vals = []
    err_vals = []
    for lo, hi in zip(angle_bins[:-1], angle_bins[1:], strict=False):
        sel = (df["angle_deg"] >= lo) & (df["angle_deg"] < hi)
        if not np.any(sel):
            mean_vals.append(np.nan)
            err_vals.append(np.nan)
            continue
        w = weights[sel]
        vals = corr[sel]
        mean = np.average(vals, weights=w)
        variance = np.average((vals - mean) ** 2, weights=w)
        err = math.sqrt(variance) / math.sqrt(w.size)
        mean_vals.append(mean)
        err_vals.append(err)
    ax.errorbar(bin_centers, mean_vals, yerr=err_vals, fmt="o", color="black", ms=4, label="Binned mean")

    theta = np.linspace(0, math.pi, 256)
    ax.plot(np.degrees(theta), amp * hellings_downs(theta), color="crimson", lw=2, label=f"HD × {amp:.2f}")
    ax.set_xlabel("Angular separation (deg)")
    ax.set_ylabel("Weighted correlation coefficient")
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.6, 1.1)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = outdir / "hd_reanalysis.pdf"
    fig.savefig(plot_path)
    logging.info("Saved pair table to %s", csv_path)
    logging.info("Saved summary to %s", outdir / "hd_summary.json")
    logging.info("Saved plot to %s", plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="dataset/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0",
        type=Path,
        help="Path to the unpacked NANOGrav data set",
    )
    parser.add_argument(
        "--outdir",
        default=Path("analysis_outputs/hd_check"),
        type=Path,
        help="Directory where outputs will be written",
    )
    parser.add_argument(
        "--max-pulsars",
        type=int,
        default=24,
        help="Maximum number of pulsars to process from the wideband release",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(args.dataset_root, args.outdir, args.max_pulsars)


if __name__ == "__main__":
    main()
