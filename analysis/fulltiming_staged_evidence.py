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
from scipy import linalg as sl
from scipy import sparse as sps
from scipy.stats import norm
from sksparse.cholmod import CholmodNotPositiveDefiniteError

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pta_pipeline import get_tspan_years, load_noise_dict, load_pulsars  # noqa: E402
from enterprise_extensions import models  # noqa: E402


LOG = logging.getLogger(__name__)


MODEL_CHOICES = ("hd_powerlaw", "hd_turnover", "hd_phase_bpl")
EXCLUDED_WIDEBAND_DMX_PULSARS = ("J1713+0747",)


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
    parser.add_argument("--fix-red-noise", action="store_true")
    parser.add_argument("--nlive", type=int, default=256)
    parser.add_argument("--dlogz", type=float, default=2.0)
    parser.add_argument("--walks", type=int, default=8)
    parser.add_argument("--sampler", choices=("rwalk", "rslice", "slice"), default="rwalk")
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--phiinv-method", choices=("cliques", "partition", "sparse"), default="cliques")
    parser.add_argument(
        "--likelihood-backend",
        choices=("cpu", "cuda-dense", "cuda-sparse", "cuda-partial-schur", "cuda-hybrid-cholmod"),
        default="cpu",
    )
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
        use_dmdata=True,
        dm_var=False,
        bayesephem=False,
        white_vary=False,
        noisedict=noise_dict,
        orf="hd",
        red_var=not args.fix_red_noise,
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


def cuda_dense_loglike_factory(pta, phiinv_method: str):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular

    names = pta.param_names
    reference_params = {}
    for name, param in zip(names, pta.params):
        prior_name = param.prior._func.__name__
        if prior_name == "UniformPrior":
            bounds = param.prior._defaults
            reference_params[name] = 0.5 * (bounds["pmin"] + bounds["pmax"])
        elif prior_name == "NormalPrior":
            reference_params[name] = param.prior._defaults["mu"]
        else:
            reference_params[name] = 0.0

    TNrs = pta.get_TNr(reference_params)
    TNTs = pta.get_TNT(reference_params)
    rNr_logdet = pta.get_rNr_logdet(reference_params)
    loglike_constant = -0.5 * np.sum([ell for pair in rNr_logdet for ell in pair])

    if pta._commonsignals:
        TNT_gpu = cp.asarray(sl.block_diag(*TNTs))
        TNr_gpu = cp.asarray(np.concatenate(TNrs))
        cached_blocks = None
    else:
        TNT_gpu = None
        TNr_gpu = None
        cached_blocks = [
            (cp.asarray(TNT), cp.asarray(TNr) if TNr is not None else None)
            for TNr, TNT in zip(TNrs, TNTs)
        ]

    def solve_cholesky(cf, rhs):
        y = solve_triangular(cf, rhs, lower=True)
        return solve_triangular(cf, y, lower=True, trans="T")

    def loglike(theta: np.ndarray) -> float:
        params = dict(zip(names, theta))
        try:
            phiinvs = pta.get_phiinv(params, logdet=True, method=phiinv_method)
            loglike_value = loglike_constant
            loglike_value += sum(pta.get_logsignalprior(params))

            if pta._commonsignals:
                phiinv, logdet_phi = phiinvs
                if hasattr(phiinv, "toarray"):
                    phiinv = phiinv.toarray()
                Sigma_gpu = TNT_gpu + cp.asarray(phiinv)
                cf = cp.linalg.cholesky(Sigma_gpu)
                expval = solve_cholesky(cf, TNr_gpu)
                logdet_sigma = 2 * cp.sum(cp.log(cp.diag(cf)))
                return float(
                    loglike_value
                    + 0.5 * (cp.dot(TNr_gpu, expval) - logdet_sigma - float(logdet_phi)).get()
                )

            for TNT_block, TNr_block, pl in zip((block[0] for block in cached_blocks), (block[1] for block in cached_blocks), phiinvs):
                if TNr_block is None:
                    continue
                phiinv, logdet_phi = pl
                if hasattr(phiinv, "toarray"):
                    phiinv = phiinv.toarray()
                Sigma_gpu = TNT_block + cp.asarray(np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
                cf = cp.linalg.cholesky(Sigma_gpu)
                expval = solve_cholesky(cf, TNr_block)
                logdet_sigma = 2 * cp.sum(cp.log(cp.diag(cf)))
                loglike_value += 0.5 * float(
                    (cp.dot(TNr_block, expval) - logdet_sigma - float(logdet_phi)).get()
                )
            return float(loglike_value)
        except (CholmodNotPositiveDefiniteError, np.linalg.LinAlgError, ValueError, cp.linalg.LinAlgError):
            return -np.inf

    return loglike


def csr_positions(template: sps.csr_matrix, matrix: sps.csr_matrix) -> np.ndarray:
    row_maps = []
    for row in range(template.shape[0]):
        start, end = template.indptr[row], template.indptr[row + 1]
        row_maps.append({int(col): pos for pos, col in enumerate(template.indices[start:end], start)})
    positions = np.empty(matrix.nnz, dtype=np.int64)
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_map = row_maps[row]
        for pos in range(start, end):
            positions[pos] = row_map[int(matrix.indices[pos])]
    return positions


def csc_positions(template: sps.csc_matrix, matrix: sps.csc_matrix) -> np.ndarray:
    return csr_positions(template.T.tocsr(), matrix.T.tocsr())


def partition_sparse_phiinv_data_factory(pta, reference_params: Dict[str, float], cp):
    phivecs0 = [signalcollection.get_phi(reference_params) for signalcollection in pta._signalcollections]
    if np.any([phi.ndim == 2 for phi in phivecs0 if phi is not None]):
        raise NotImplementedError("cuda-sparse requires diagonal per-pulsar Phi blocks")
    if len(pta._commonsignals) != 1:
        raise NotImplementedError("cuda-sparse requires one common signal")

    phis0 = [phi for phi in phivecs0 if phi is not None]
    size = int(sum(len(phi) for phi in phis0))
    diag_matrix = sps.identity(size, format="csr", dtype=np.float64)
    slices = pta._get_slices(phivecs0)
    csclass, csdict = next(iter(pta._commonsignals.items()))
    common_items = list(csdict.items())
    npsr = len(common_items)

    common_indices = np.asarray(
        [slices[csc].start + csc._idx[cs] for cs, csc in common_items],
        dtype=np.int64,
    )
    nfreq = common_indices.shape[1]
    pair_i, pair_j = np.tril_indices(npsr)
    common_rows = common_indices[pair_i, :].T.ravel()
    common_cols = common_indices[pair_j, :].T.ravel()
    common_pattern = sps.coo_matrix(
        (np.ones_like(common_rows, dtype=np.float64), (common_rows, common_cols)),
        shape=(size, size),
    ).tocsr()
    common_order = sps.coo_matrix(
        (np.arange(len(common_rows), dtype=np.int64), (common_rows, common_cols)),
        shape=(size, size),
    ).tocsr().data.astype(np.int64, copy=False)

    phi_template = (diag_matrix + common_pattern).tocsr()
    phi_template.data = np.zeros_like(phi_template.data, dtype=np.float64)
    phi_template.sort_indices()
    diag_positions = csr_positions(phi_template, diag_matrix)
    common_positions = csr_positions(phi_template, common_pattern)

    orf = np.empty((npsr, npsr), dtype=np.float64)
    for i, (cs1, _) in enumerate(common_items):
        for j, (cs2, _) in enumerate(common_items):
            orf[i, j] = float(np.asarray(csclass._orf(cs1._psrpos, cs2._psrpos, params=reference_params)))

    common_signal = common_items[0][0]
    diag_positions_gpu = cp.asarray(diag_positions)
    common_positions_gpu = cp.asarray(common_positions)
    common_order_gpu = cp.asarray(common_order)
    common_indices_gpu = cp.asarray(common_indices)
    pair_i_gpu = cp.asarray(pair_i)
    pair_j_gpu = cp.asarray(pair_j)
    diag_idx_gpu = cp.arange(npsr)
    orf_gpu = cp.asarray(orf)
    eye_gpu = cp.eye(npsr, dtype=cp.float64)
    phi_data_gpu = cp.empty(phi_template.nnz, dtype=cp.float64)

    def build(params: Dict[str, float]):
        phivecs = [signalcollection.get_phi(params) for signalcollection in pta._signalcollections]
        phidiag = cp.asarray(np.concatenate([phi for phi in phivecs if phi is not None]).astype(np.float64))
        phi_data_gpu[diag_positions_gpu] = 1.0 / phidiag
        logdet_phi = cp.sum(cp.log(phidiag))

        common_diag = phidiag[common_indices_gpu].T
        logdet_phi -= cp.sum(cp.log(common_diag))
        prior = cp.asarray(csclass._prior(common_signal._labels, params=params))
        blocks = prior[:, None, None] * orf_gpu[None, :, :]
        blocks[:, diag_idx_gpu, diag_idx_gpu] = common_diag

        cf = cp.linalg.cholesky(blocks)
        rhs = cp.broadcast_to(eye_gpu, blocks.shape)
        y = cp.linalg.solve(cf, rhs)
        inv_blocks = cp.linalg.solve(cp.swapaxes(cf, -1, -2), y)
        values = inv_blocks[:, pair_i_gpu, pair_j_gpu]
        logdet_phi += 2.0 * cp.sum(cp.log(cp.diagonal(cf, axis1=1, axis2=2)))

        phi_data_gpu[common_positions_gpu] = values.ravel()[common_order_gpu]
        return phi_data_gpu, float(logdet_phi.get())

    return phi_template, build


def cuda_sparse_loglike_factory(pta):
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import nvmath.sparse.advanced as nvs

    names = pta.param_names
    reference_params = {}
    for name, param in zip(names, pta.params):
        prior_name = param.prior._func.__name__
        if prior_name == "UniformPrior":
            bounds = param.prior._defaults
            reference_params[name] = 0.5 * (bounds["pmin"] + bounds["pmax"])
        elif prior_name == "NormalPrior":
            reference_params[name] = param.prior._defaults["mu"]
        else:
            reference_params[name] = 0.0

    TNrs = pta.get_TNr(reference_params)
    TNT_sp = sps.block_diag(pta.get_TNT(reference_params), "csr")
    TNr_gpu = cp.asarray(np.concatenate(TNrs).astype(np.float64))
    rNr_logdet = pta.get_rNr_logdet(reference_params)
    loglike_constant = -0.5 * np.sum([ell for pair in rNr_logdet for ell in pair])
    solver_logger = logging.getLogger("fulltiming.nvmath")
    solver_logger.setLevel(logging.WARNING)
    solver_options = nvs.DirectSolverOptions(
        sparse_system_type=nvs.DirectSolverMatrixType.SPD,
        sparse_system_view=nvs.DirectSolverMatrixViewType.FULL,
        logger=solver_logger,
    )
    phi_template, build_phiinv_data = partition_sparse_phiinv_data_factory(pta, reference_params, cp)
    TNT_lower = sps.tril(TNT_sp, format="csr")
    TNT_sym = (TNT_lower + TNT_lower.T - sps.diags(TNT_lower.diagonal(), format="csr")).tocsr()
    phi_coo = phi_template.tocoo()
    phi_order_data = np.arange(phi_template.nnz, dtype=np.float64)
    phi_offdiag = phi_coo.row != phi_coo.col
    phi_rows = np.concatenate([phi_coo.row, phi_coo.col[phi_offdiag]])
    phi_cols = np.concatenate([phi_coo.col, phi_coo.row[phi_offdiag]])
    phi_order_values = np.concatenate([phi_order_data, phi_order_data[phi_offdiag]])
    phi_order = sps.coo_matrix(
        (phi_order_values, (phi_rows, phi_cols)),
        shape=phi_template.shape,
    ).tocsr()
    phi_pattern = phi_order.copy()
    phi_pattern.data = np.ones_like(phi_pattern.data)

    pattern = TNT_sym.copy()
    pattern.data = np.ones_like(pattern.data)
    Sigma0 = (pattern + phi_pattern).tocsr()
    Sigma0.data = np.zeros_like(Sigma0.data, dtype=np.float64)
    Sigma0.sort_indices()

    base_data = Sigma0.data.copy()
    base_data[csr_positions(Sigma0, TNT_sym)] += TNT_sym.data.astype(np.float64, copy=False)
    phi_positions_gpu = cp.asarray(csr_positions(Sigma0, phi_pattern))
    phi_order_gpu = cp.asarray(phi_order.data.astype(np.int64, copy=False))
    Sigma_gpu = cpsp.csr_matrix(Sigma0)
    base_data_gpu = cp.asarray(base_data)
    solver = nvs.DirectSolver(Sigma_gpu, TNr_gpu, options=solver_options)
    solver.plan()
    shadow_calls = int(os.environ.get("CUDA_SPARSE_SHADOW_CALLS", "3"))
    shadow_interval = int(os.environ.get("CUDA_SPARSE_SHADOW_INTERVAL", "0"))
    shadow_tol = float(os.environ.get("CUDA_SPARSE_TOL", "1e-3"))
    state = {"remaining": shadow_calls, "calls": 0, "enabled": True}

    def loglike(theta: np.ndarray) -> float:
        params = dict(zip(names, theta))
        if not state["enabled"]:
            raise RuntimeError("cuda-sparse disabled by failed shadow check")
        try:
            phi_data, logdet_phi = build_phiinv_data(params)
            Sigma_gpu.data[:] = base_data_gpu
            Sigma_gpu.data[phi_positions_gpu] += phi_data[phi_order_gpu]
            factor_info = solver.factorize()
            expval = solver.solve()
            quad = cp.dot(TNr_gpu, expval).get()
            logdet_sigma = 2.0 * np.sum(np.log(np.abs(factor_info.diag)))
            logsignal_prior = sum(pta.get_logsignalprior(params))
            value = float(loglike_constant + logsignal_prior + 0.5 * (quad - logdet_sigma - float(logdet_phi)))
            state["calls"] += 1
            should_shadow = state["remaining"] > 0 or (
                shadow_interval > 0 and state["calls"] % shadow_interval == 0
            )
            if should_shadow:
                exact_value = pta.get_lnlikelihood(params, phiinv_method="partition")
                diff = value - exact_value
                LOG.info(
                    "cuda-sparse shadow diff %.6g (sparse %.12g, exact %.12g)",
                    diff,
                    value,
                    exact_value,
                )
                if state["remaining"] > 0:
                    state["remaining"] -= 1
                if not np.isfinite(diff) or abs(diff) > shadow_tol:
                    state["enabled"] = False
                    raise RuntimeError(
                        f"cuda-sparse disabled: shadow diff {diff:.6g} exceeds tolerance {shadow_tol:.6g}"
                    )
            return value
        except RuntimeError as exc:
            if "shadow diff" in str(exc) or "failed shadow" in str(exc):
                raise
            return -np.inf
        except (CholmodNotPositiveDefiniteError, np.linalg.LinAlgError, ValueError, cp.linalg.LinAlgError):
            return -np.inf

    return loglike


def cuda_hybrid_cholmod_loglike_factory(pta):
    import cupy as cp
    from sksparse.cholmod import cholesky

    names = pta.param_names
    reference_params = {}
    for name, param in zip(names, pta.params):
        prior_name = param.prior._func.__name__
        if prior_name == "UniformPrior":
            bounds = param.prior._defaults
            reference_params[name] = 0.5 * (bounds["pmin"] + bounds["pmax"])
        elif prior_name == "NormalPrior":
            reference_params[name] = param.prior._defaults["mu"]
        else:
            reference_params[name] = 0.0

    TNrs = pta.get_TNr(reference_params)
    TNT_sp = sps.block_diag(pta.get_TNT(reference_params), "csr")
    TNr = np.concatenate(TNrs).astype(np.float64)
    rNr_logdet = pta.get_rNr_logdet(reference_params)
    loglike_constant = -0.5 * np.sum([ell for pair in rNr_logdet for ell in pair])
    phi_template, build_phiinv_data = partition_sparse_phiinv_data_factory(pta, reference_params, cp)

    TNT_lower = sps.tril(TNT_sp, format="csr")
    TNT_sym = (TNT_lower + TNT_lower.T - sps.diags(TNT_lower.diagonal(), format="csr")).tocsc()
    phi_coo = phi_template.tocoo()
    phi_order_data = np.arange(phi_template.nnz, dtype=np.float64)
    phi_offdiag = phi_coo.row != phi_coo.col
    phi_rows = np.concatenate([phi_coo.row, phi_coo.col[phi_offdiag]])
    phi_cols = np.concatenate([phi_coo.col, phi_coo.row[phi_offdiag]])
    phi_order_values = np.concatenate([phi_order_data, phi_order_data[phi_offdiag]])
    phi_order = sps.coo_matrix(
        (phi_order_values, (phi_rows, phi_cols)),
        shape=phi_template.shape,
    ).tocsc()
    phi_pattern = phi_order.copy()
    phi_pattern.data = np.ones_like(phi_pattern.data)
    pattern = TNT_sym.copy()
    pattern.data = np.ones_like(pattern.data)
    Sigma_cpu = (pattern + phi_pattern).tocsc()
    Sigma_cpu.data = np.zeros_like(Sigma_cpu.data, dtype=np.float64)
    Sigma_cpu.sort_indices()

    base_data = Sigma_cpu.data.copy()
    base_data[csc_positions(Sigma_cpu, TNT_sym)] += TNT_sym.data.astype(np.float64, copy=False)
    phi_positions = csc_positions(Sigma_cpu, phi_pattern)
    phi_csc_order = phi_order.data.astype(np.int64, copy=False)
    phi_data0, _ = build_phiinv_data(reference_params)
    Sigma_cpu.data[:] = base_data
    Sigma_cpu.data[phi_positions] += cp.asnumpy(phi_data0)[phi_csc_order]
    cholmod_mode = os.environ.get("CUDA_HYBRID_CHOLMOD_MODE", "auto")
    cholmod_ordering = os.environ.get("CUDA_HYBRID_CHOLMOD_ORDERING", "default")
    factor = cholesky(Sigma_cpu, mode=cholmod_mode, ordering_method=cholmod_ordering)
    LOG.info(
        "Initialized cuda-hybrid-cholmod with dim %d, nnz %d, mode %s, ordering %s",
        Sigma_cpu.shape[0],
        Sigma_cpu.nnz,
        cholmod_mode,
        cholmod_ordering,
    )
    shadow_calls = int(os.environ.get("CUDA_HYBRID_SHADOW_CALLS", "3"))
    shadow_interval = int(os.environ.get("CUDA_HYBRID_SHADOW_INTERVAL", "0"))
    shadow_tol = float(os.environ.get("CUDA_HYBRID_TOL", "1e-3"))
    state = {"remaining": shadow_calls, "calls": 0, "enabled": True}

    def loglike(theta: np.ndarray) -> float:
        params = dict(zip(names, theta))
        if not state["enabled"]:
            raise RuntimeError("cuda-hybrid-cholmod disabled by failed shadow check")
        try:
            phi_data, logdet_phi = build_phiinv_data(params)
            Sigma_cpu.data[:] = base_data
            Sigma_cpu.data[phi_positions] += cp.asnumpy(phi_data)[phi_csc_order]
            factor.cholesky_inplace(Sigma_cpu)
            expval = factor.solve_A(TNr)
            quad = float(TNr @ expval)
            logdet_sigma = float(factor.logdet())
            logsignal_prior = sum(pta.get_logsignalprior(params))
            value = float(loglike_constant + logsignal_prior + 0.5 * (quad - logdet_sigma - float(logdet_phi)))
            state["calls"] += 1
            should_shadow = state["remaining"] > 0 or (
                shadow_interval > 0 and state["calls"] % shadow_interval == 0
            )
            if should_shadow:
                exact_value = pta.get_lnlikelihood(params, phiinv_method="partition")
                diff = value - exact_value
                LOG.info(
                    "cuda-hybrid-cholmod shadow diff %.6g (hybrid %.12g, exact %.12g)",
                    diff,
                    value,
                    exact_value,
                )
                if state["remaining"] > 0:
                    state["remaining"] -= 1
                if not np.isfinite(diff) or abs(diff) > shadow_tol:
                    state["enabled"] = False
                    raise RuntimeError(
                        f"cuda-hybrid-cholmod disabled: shadow diff {diff:.6g} exceeds tolerance {shadow_tol:.6g}"
                    )
            return value
        except RuntimeError as exc:
            if "shadow diff" in str(exc) or "failed shadow" in str(exc):
                raise
            return -np.inf
        except (CholmodNotPositiveDefiniteError, np.linalg.LinAlgError, ValueError, cp.linalg.LinAlgError):
            return -np.inf

    return loglike


def cuda_partial_schur_loglike_factory(pta):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular

    names = pta.param_names
    reference_params = {}
    for name, param in zip(names, pta.params):
        prior_name = param.prior._func.__name__
        if prior_name == "UniformPrior":
            bounds = param.prior._defaults
            reference_params[name] = 0.5 * (bounds["pmin"] + bounds["pmax"])
        elif prior_name == "NormalPrior":
            reference_params[name] = param.prior._defaults["mu"]
        else:
            reference_params[name] = 0.0

    TNr = np.concatenate(pta.get_TNr(reference_params)).astype(np.float64)
    TNT_sp = sps.block_diag(pta.get_TNT(reference_params), "csr")
    rNr_logdet = pta.get_rNr_logdet(reference_params)
    loglike_constant = -0.5 * np.sum([ell for pair in rNr_logdet for ell in pair])

    phivecs0 = [signalcollection.get_phi(reference_params) for signalcollection in pta._signalcollections]
    if np.any([phi.ndim == 2 for phi in phivecs0 if phi is not None]):
        raise NotImplementedError("cuda-partial-schur requires diagonal per-pulsar Phi blocks")
    if len(pta._commonsignals) != 1:
        raise NotImplementedError("cuda-partial-schur requires one common signal")

    slices = pta._get_slices(phivecs0)
    _, csdict = next(iter(pta._commonsignals.items()))
    common_items = list(csdict.items())
    common_indices = np.asarray(
        [slices[csc].start + csc._idx[cs] for cs, csc in common_items],
        dtype=np.int64,
    )
    common_flat = common_indices.T.ravel()
    local_mask = np.ones(TNT_sp.shape[0], dtype=bool)
    local_mask[common_flat] = False
    keep_idx = np.arange(TNT_sp.shape[0], dtype=np.int64)[~local_mask]

    if np.all(~local_mask):
        raise ValueError("cuda-partial-schur requires at least one eliminated coefficient")

    phi_template, build_phiinv_data = partition_sparse_phiinv_data_factory(pta, reference_params, cp)
    TNT_lower = sps.tril(TNT_sp, format="csr")
    TNT_sym = (TNT_lower + TNT_lower.T - sps.diags(TNT_lower.diagonal(), format="csr")).tocsr()
    schur_base = TNT_sym[keep_idx, :][:, keep_idx].toarray().astype(np.float64, copy=False)
    rhs_base = TNr[keep_idx]
    orig_to_keep = np.full(TNT_sp.shape[0], -1, dtype=np.int64)
    orig_to_keep[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    diag_positions = csr_positions(phi_template, sps.identity(TNT_sp.shape[0], format="csr", dtype=np.float64))
    local_blocks = []
    for cs, csc in common_items:
        span = np.arange(slices[csc].start, slices[csc].stop, dtype=np.int64)
        common_idx = np.asarray(slices[csc].start + csc._idx[cs], dtype=np.int64).ravel()
        local_idx = span[~np.isin(span, common_idx)]
        if local_idx.size == 0:
            continue
        common_pos = orig_to_keep[common_idx]
        if np.any(common_pos < 0):
            raise ValueError("cuda-partial-schur common index mapping failed")
        local_blocks.append(
            {
                "base": cp.asarray(
                    TNT_sym[local_idx, :][:, local_idx].toarray().astype(np.float64, copy=False)
                ),
                "B": cp.asarray(TNT_sym[local_idx, :][:, common_idx].toarray().astype(np.float64, copy=False)),
                "rhs": cp.asarray(TNr[local_idx].astype(np.float64, copy=False)),
                "diag_positions": cp.asarray(diag_positions[local_idx]),
                "diag_idx": cp.arange(local_idx.size),
                "common_pos": cp.asarray(common_pos),
            }
        )

    phi_position_template = phi_template.copy()
    phi_position_template.data = np.arange(phi_template.nnz, dtype=np.float64)
    phi_reduced_coo = phi_template[keep_idx, :][:, keep_idx].tocoo()
    phi_reduced_positions = phi_position_template[keep_idx, :][:, keep_idx].tocoo().data.astype(
        np.int64,
        copy=False,
    )
    phi_rows_gpu = cp.asarray(phi_reduced_coo.row.astype(np.int64, copy=False))
    phi_cols_gpu = cp.asarray(phi_reduced_coo.col.astype(np.int64, copy=False))
    phi_offdiag = phi_reduced_coo.row != phi_reduced_coo.col
    phi_offdiag_gpu = cp.asarray(phi_offdiag)
    phi_reduced_positions_gpu = cp.asarray(phi_reduced_positions)
    Sigma_gpu = cp.empty_like(cp.asarray(schur_base.astype(np.float64, copy=False)))
    schur_base_gpu = cp.asarray(schur_base.astype(np.float64, copy=False))
    rhs_base_gpu = cp.asarray(rhs_base.astype(np.float64, copy=False))

    shadow_calls = int(os.environ.get("CUDA_PARTIAL_SCHUR_SHADOW_CALLS", "3"))
    shadow_interval = int(os.environ.get("CUDA_PARTIAL_SCHUR_SHADOW_INTERVAL", "0"))
    shadow_tol = float(os.environ.get("CUDA_PARTIAL_SCHUR_TOL", "1e-3"))
    state = {"remaining": shadow_calls, "calls": 0, "enabled": True}
    LOG.info(
        "Initialized cuda-partial-schur with dim %d -> %d, eliminated %d local coefficients",
        TNT_sp.shape[0],
        keep_idx.size,
        int(np.sum(local_mask)),
    )

    def partial_value(params: Dict[str, float]) -> float:
        phi_data, logdet_phi = build_phiinv_data(params)
        Sigma_gpu[:] = schur_base_gpu
        rhs_gpu = rhs_base_gpu.copy()
        local_quad = cp.array(0.0, dtype=cp.float64)
        local_logdet = cp.array(0.0, dtype=cp.float64)
        for block in local_blocks:
            A = block["base"].copy()
            A[block["diag_idx"], block["diag_idx"]] += phi_data[block["diag_positions"]]
            cfA = cp.linalg.cholesky(A)
            rhs = cp.concatenate((block["B"], block["rhs"][:, None]), axis=1)
            y = solve_triangular(cfA, rhs, lower=True)
            solved = solve_triangular(cfA, y, lower=True, trans="T")
            AinvB = solved[:, :-1]
            Ainvr = solved[:, -1]
            pos = block["common_pos"]
            Sigma_gpu[pos[:, None], pos[None, :]] -= block["B"].T @ AinvB
            rhs_gpu[pos] -= block["B"].T @ Ainvr
            local_quad += block["rhs"] @ Ainvr
            local_logdet += 2.0 * cp.sum(cp.log(cp.diag(cfA)))
        phi_values = phi_data[phi_reduced_positions_gpu]
        Sigma_gpu[phi_rows_gpu, phi_cols_gpu] += phi_values
        Sigma_gpu[phi_cols_gpu[phi_offdiag_gpu], phi_rows_gpu[phi_offdiag_gpu]] += phi_values[
            phi_offdiag_gpu
        ]
        cf = cp.linalg.cholesky(Sigma_gpu)
        y = solve_triangular(cf, rhs_gpu, lower=True)
        expval = solve_triangular(cf, y, lower=True, trans="T")
        quad = float((local_quad + cp.dot(rhs_gpu, expval)).get())
        logdet_sigma = float((local_logdet + 2.0 * cp.sum(cp.log(cp.diag(cf)))).get())
        logsignal_prior = sum(pta.get_logsignalprior(params))
        return float(loglike_constant + logsignal_prior + 0.5 * (quad - logdet_sigma - float(logdet_phi)))

    def loglike(theta: np.ndarray) -> float:
        params = dict(zip(names, theta))
        if not state["enabled"]:
            raise RuntimeError("cuda-partial-schur disabled by failed shadow check")
        try:
            value = partial_value(params)
        except (CholmodNotPositiveDefiniteError, np.linalg.LinAlgError, ValueError, RuntimeError, cp.linalg.LinAlgError):
            state["enabled"] = False
            raise
        state["calls"] += 1
        should_shadow = state["remaining"] > 0 or (
            shadow_interval > 0 and state["calls"] % shadow_interval == 0
        )
        if should_shadow:
            exact_value = pta.get_lnlikelihood(params, phiinv_method="partition")
            diff = value - exact_value
            LOG.info(
                "cuda-partial-schur shadow diff %.6g (partial %.12g, exact %.12g)",
                diff,
                value,
                exact_value,
            )
            if state["remaining"] > 0:
                state["remaining"] -= 1
            if not np.isfinite(diff) or abs(diff) > shadow_tol:
                state["enabled"] = False
                raise RuntimeError(
                    f"cuda-partial-schur disabled: shadow diff {diff:.6g} exceeds tolerance {shadow_tol:.6g}"
                )
        return value

    return loglike


def loglike_factory(pta, phiinv_method: str, likelihood_backend: str):
    if likelihood_backend == "cuda-dense":
        return cuda_dense_loglike_factory(pta, phiinv_method)
    if likelihood_backend == "cuda-sparse":
        return cuda_sparse_loglike_factory(pta)
    if likelihood_backend == "cuda-hybrid-cholmod":
        return cuda_hybrid_cholmod_loglike_factory(pta)
    if likelihood_backend == "cuda-partial-schur":
        return cuda_partial_schur_loglike_factory(pta)

    names = pta.param_names

    def loglike(theta: np.ndarray) -> float:
        params = dict(zip(names, theta))
        try:
            return pta.get_lnlikelihood(params, phiinv_method=phiinv_method)
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

    attempts = 0
    while len(live_u) < nlive and attempts < 100000:
        u = rng.uniform(0.0, 1.0, size=ndim)
        try_add(u)
        attempts += 1
    if len(live_u) < nlive:
        raise RuntimeError(f"Only initialized {len(live_u)} of {nlive} live points")
    return np.vstack(live_u), np.vstack(live_v), np.asarray(live_logl)


def run_dynesty(
    pta,
    outdir: Path,
    *,
    nlive: int,
    dlogz: float,
    walks: int,
    sampler_name: str,
    maxiter: int | None,
    maxcall: int | None,
    seed: int,
    phiinv_method: str,
    likelihood_backend: str,
) -> Dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    ndim = len(pta.param_names)
    transform = prior_transform_factory(pta.params, pta.param_names)
    loglike = loglike_factory(pta, phiinv_method, likelihood_backend)
    live_points = initialize_live_points(transform, loglike, ndim, nlive, seed)
    LOG.info("Initialized %d live points for %d parameters", nlive, ndim)

    sampler = NestedSampler(
        loglike,
        transform,
        ndim,
        nlive=nlive,
        bound="multi",
        sample=sampler_name,
        walks=walks,
        update_interval=10,
        first_update={"min_ncall": 500, "min_eff": 5.0},
        live_points=live_points,
    )
    start = time.time()
    progress_path = outdir / "progress.jsonl"
    next_progress_at = {"time": start}
    last_status = {"delta_logz": float("nan")}

    def log_progress(results, iteration: int, ncall: int, **_: object) -> None:
        now = time.time()
        should_log = now >= next_progress_at["time"]
        if should_log:
            next_progress_at["time"] = now + 300.0
        logzerr = float(np.sqrt(results.logzvar)) if results.logzvar >= 0 else float("nan")
        record = {
            "elapsed_seconds": float(now - start),
            "iteration": int(iteration),
            "ncall": int(ncall),
            "logz": float(results.logz),
            "logzerr": logzerr,
            "delta_logz": float(results.delta_logz),
            "eff": float(results.eff),
        }
        last_status["delta_logz"] = record["delta_logz"]
        if not should_log:
            return
        with progress_path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        LOG.info("PROGRESS %s", json.dumps(record, sort_keys=True))

    sampler.run_nested(
        maxiter=maxiter,
        maxcall=maxcall,
        dlogz=dlogz,
        print_progress=True,
        print_func=log_progress,
    )
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
        "final_delta_logz": float(last_status["delta_logz"]),
        "converged": bool(last_status["delta_logz"] <= dlogz),
        "information": float(results.information[-1]),
        "nlive": nlive,
        "dlogz": dlogz,
        "walks": walks,
        "sampler": sampler_name,
        "maxiter": maxiter,
        "maxcall": maxcall,
        "phiinv_method": phiinv_method,
        "likelihood_backend": likelihood_backend,
        "runtime_seconds": float(runtime),
        "ncall": int(np.sum(results.ncall)),
        "nparams": ndim,
    }
    (outdir / "evidence.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_metadata(outdir: Path, pulsars, args: argparse.Namespace) -> None:
    metadata = {
        "pulsars": [psr.name for psr in pulsars],
        "excluded_pulsars": list(EXCLUDED_WIDEBAND_DMX_PULSARS),
        "exclusion_reason": "WidebandTimingModel cannot account for all TOAs in DMX intervals for these pulsars.",
        "toas": [int(len(psr.toas)) for psr in pulsars],
        "Tspan_years": float(get_tspan_years(pulsars)),
        "common_components": args.common_components,
        "red_components": args.red_components,
        "fix_red_noise": args.fix_red_noise,
        "orf": "hd",
        "tm_marg": False,
        "dm_var": False,
        "bayesephem": False,
        "models": args.models,
        "phiinv_method": args.phiinv_method,
        "likelihood_backend": args.likelihood_backend,
        "note": "Staged real-TOA HD full-timing evidence using enterprise_extensions.models.model_general.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    setup_logging(args.outdir)
    configure_local_timing_files(args.dataset_root)
    LOG.info("Loading NANOGrav pulsars from %s", args.dataset_root)
    pulsars = load_pulsars(args.dataset_root, max_pulsars=args.max_pulsars)
    pulsars = [psr for psr in pulsars if psr.name not in EXCLUDED_WIDEBAND_DMX_PULSARS]
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
            sampler_name=args.sampler,
            maxiter=args.maxiter,
            maxcall=args.maxcall,
            seed=args.seed + idx,
            phiinv_method=args.phiinv_method,
            likelihood_backend=args.likelihood_backend,
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
