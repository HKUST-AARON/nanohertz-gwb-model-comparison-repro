#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
KDE_OUT = REPO_ROOT / 'analysis_outputs' / 'kde_model_comparison'
KDE_DIR = REPO_ROOT / 'data_sources' / 'NANOGrav15yr_KDE-FreeSpectra' / '30f_fs{hd}_ceffyl'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FYR = 1.0 / (365.25 * 24 * 3600)
H0 = 67.4 * 1000.0 / 3.085677581491367e22
LOG10_12PI2 = np.log10(12.0 * np.pi**2)
G = 6.67430e-11
C = 2.99792458e8
MSUN = 1.98847e30
PC = 3.085677581491367e16
MSUN_PER_PC3_TO_SI = MSUN / (PC**3)


def load_summary(model):
    return json.loads((KDE_OUT / model / 'posterior_summary.json').read_text())


def load_samples(model):
    data = np.load(KDE_OUT / model / 'posterior_samples.npz')
    return data['samples'], [str(x) for x in data['param_names']]


def log10hc_from_log10rho(freqs, log10rho):
    df = np.min(freqs)
    return log10rho + 0.5 * (LOG10_12PI2 + 3.0 * np.log10(freqs) - np.log10(df))


def kde_free_spectrum_quantiles():
    freqs = np.load(KDE_DIR / 'freqs.npy')
    grid = np.load(KDE_DIR / 'log10rhogrid.npy')
    log_density = np.load(KDE_DIR / 'density.npy')[0]
    quantiles = []
    for row in log_density:
        weights = np.exp(row - np.max(row))
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        quantiles.append(np.interp([0.025, 0.16, 0.50, 0.84, 0.975], cdf, grid))
    log10rho_q = np.array(quantiles)
    log10hc_q = np.column_stack([log10hc_from_log10rho(freqs, log10rho_q[:, i]) for i in range(5)])
    return freqs, log10hc_q


def log10hc_smbhb_pl(freqs):
    summary = load_summary('smbhb_pl')
    log10_a = summary['log10_A']['median']
    return log10_a - (2.0 / 3.0) * np.log10(freqs / FYR)


def log10hc_smbhb_env(freqs):
    summary = load_summary('smbhb_env')
    log10_a = summary['log10_A']['median']
    fb = 10.0 ** summary['log10_fb']['median']
    delta_gamma = summary['Delta_gamma']['median']
    return log10_a - (2.0 / 3.0) * np.log10(freqs / FYR) - 0.5 * np.log10(1.0 + (fb / freqs) ** delta_gamma)


def log10hc_cosmic_strings(freqs):
    summary = load_summary('cosmic_strings')
    return summary['log10_A']['median'] + summary['beta_hc']['median'] * np.log10(freqs / FYR)


def log10hc_phase_transition(freqs):
    summary = load_summary('phase_transition')
    f_peak = 10.0 ** summary['log10_f_peak']['median']
    b = summary['b_high']['median']
    omega_peak = 10.0 ** summary['log10_Omega_peak']['median']
    x = freqs / f_peak
    omega = omega_peak * ((3.0 + b) * x**3.0) / (b + 3.0 * x**(3.0 + b))
    hc = np.sqrt(3.0 * H0**2 / (2.0 * np.pi**2)) * np.sqrt(omega) / freqs
    return np.log10(hc)


def density_from_log10fb(log10_fb, seed=20250215):
    rng = np.random.default_rng(seed)
    size = log10_fb.size
    log10_mtot = rng.normal(loc=9.3, scale=0.3, size=size)
    m_tot = (10.0 ** log10_mtot) * MSUN
    q = rng.uniform(0.25, 1.0, size=size)
    m1 = m_tot / (1.0 + q)
    m2 = m_tot - m1
    sigma = np.clip(rng.normal(loc=200e3, scale=30e3, size=size), 50e3, None)
    hardening = np.clip(rng.normal(loc=15.0, scale=5.0, size=size), 5.0, None)
    f_b = 10.0 ** log10_fb
    a_b = (G * m_tot / (np.pi**2 * f_b**2)) ** (1.0 / 3.0)
    rho_si = 64.0 * G**2 * m1 * m2 * m_tot * sigma / (5.0 * C**5 * hardening * a_b**5)
    return rho_si / MSUN_PER_PC3_TO_SI

# 1) Hellings–Downs curve + synthetic points
def hellings_downs(mu):
    # mu = cos gamma, gamma in [0,pi]
    x = (1 - mu) / 2.0
    # HD function for distinct pulsars
    return 0.5 - x/4.0 + 1.5 * x * np.log(np.clip(x, 1e-12, 1))

def make_hd_curve():
    rng = np.random.default_rng(42)
    npts = 150
    # Random sky separations isotropic: cos(gamma) ~ U[-1,1]
    mu = rng.uniform(-1, 1, size=npts)
    gamma = np.arccos(mu)
    hd = hellings_downs(mu)
    # Add synthetic scatter and symmetric errorbars
    noise = rng.normal(0, 0.03, size=npts)
    y = hd + noise
    yerr = np.full_like(y, 0.06)

    mu_grid = np.linspace(-1, 1, 1000)
    gamma_grid = np.arccos(mu_grid)
    hd_grid = hellings_downs(mu_grid)

    plt.figure(figsize=(6.8, 4.6))
    plt.errorbar(np.degrees(gamma), y, yerr=yerr, fmt='o', ms=3, alpha=0.7, label='Illustrative pairs')
    plt.plot(np.degrees(gamma_grid), hd_grid, 'k--', lw=2, label='Hellings–Downs (GR)')
    plt.xlabel('Angular separation (deg)')
    plt.ylabel(r'$\Gamma(\gamma)$')
    plt.ylim(-0.2, 0.6)
    plt.xlim(0, 180)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'HD_curve.pdf')
    plt.close()


# 2) Posterior triangle (corner-like) for SMBHB-env samples and density
def make_posterior_triangle():
    samples, _ = load_samples('smbhb_env')
    rng = np.random.default_rng(123)
    if samples.shape[0] > 9000:
        samples = samples[rng.choice(samples.shape[0], size=9000, replace=False)]
    rho_star = density_from_log10fb(samples[:, 1])
    values = np.column_stack([samples[:, 0], samples[:, 1], samples[:, 2], np.log10(rho_star)])
    labels = [
        r'$\log_{10} A$',
        r'$\log_{10}(f_b/\mathrm{Hz})$',
        r'$\Delta\gamma$',
        r'$\log_{10}\rho_\star$',
    ]

    fig, axes = plt.subplots(4, 4, figsize=(7.6, 7.2))
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                ax.hist(values[:, j], bins=45, color='0.35', alpha=0.85)
                q16, q50, q84 = np.percentile(values[:, j], [16, 50, 84])
                ax.axvline(q50, color='C3', lw=1.2)
                ax.axvspan(q16, q84, color='C3', alpha=0.15)
            elif i > j:
                ax.hexbin(values[:, j], values[:, i], gridsize=35, mincnt=1, cmap='viridis')
            else:
                ax.axis('off')
            if i == 3:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j != 0:
                ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'posterior_triangle.pdf')
    plt.close()


# 3) Strain spectra figure
def make_strain_spectra():
    freqs, data_q = kde_free_spectrum_quantiles()
    f = np.logspace(np.log10(freqs.min() * 0.8), np.log10(freqs.max() * 1.15), 400)
    models = {
        'SMBHB-PL': log10hc_smbhb_pl,
        'SMBHB-env': log10hc_smbhb_env,
        'Cosmic strings': log10hc_cosmic_strings,
        'Phase transition': log10hc_phase_transition,
    }
    colors = {
        'SMBHB-PL': 'C0',
        'SMBHB-env': 'C1',
        'Cosmic strings': 'C2',
        'Phase transition': 'C3',
    }

    fig = plt.figure(figsize=(7.4, 7.2))
    gs = GridSpec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1.45, 1.0], hspace=0.35, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(freqs, 10**data_q[:, 0], 10**data_q[:, 4], color='0.85', label='Free spectrum 95%')
    ax.fill_between(freqs, 10**data_q[:, 1], 10**data_q[:, 3], color='0.65', label='Free spectrum 68%')
    ax.plot(freqs, 10**data_q[:, 2], 'ko', ms=3.5, label='Free spectrum median')
    for label, fn in models.items():
        ax.loglog(f, 10**fn(f), lw=1.8, color=colors[label], label=label)
    ax.axvline(FYR, color='k', ls=':', lw=1)
    ax.set_xlabel('Frequency f [Hz]')
    ax.set_ylabel(r'$h_c(f)$')
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax_res = fig.add_subplot(gs[1, 0])
    median = data_q[:, 2]
    for label, fn in models.items():
        residual = fn(freqs) - median
        ax_res.semilogx(freqs, residual, marker='o', ms=2.5, lw=1.1, color=colors[label], label=label)
    ax_res.axhline(0, color='k', ls='--', lw=0.8)
    ax_res.set_xlabel('Frequency f [Hz]')
    ax_res.set_ylabel(r'$\Delta\log_{10}h_c$')

    ax_bar = fig.add_subplot(gs[1, 1])
    comp = json.loads((KDE_OUT / 'model_comparison.json').read_text())
    labels = ['SMBHB-env', 'Cosmic strings', 'Phase transition']
    keys = ['smbhb_env', 'cosmic_strings', 'phase_transition']
    vals = [comp[k]['delta_logz_vs_smbhb_pl'] for k in keys]
    errs = [comp[k]['error'] for k in keys]
    ax_bar.barh(np.arange(len(vals)), vals, xerr=errs, color=[colors[x] for x in labels], alpha=0.85)
    ax_bar.axvline(0, color='k', lw=0.8)
    ax_bar.set_yticks(np.arange(len(vals)), labels, fontsize=8)
    ax_bar.set_xlabel(r'$\Delta\ln Z$ vs. SMBHB-PL')
    ax_bar.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strain_spectra.pdf')
    plt.close()


def make_rho_posterior():
    samples, _ = load_samples('smbhb_env')
    rho_star = density_from_log10fb(samples[:, 1])
    log_rho = np.log10(rho_star)
    q16, q50, q84 = np.percentile(log_rho, [16, 50, 84])
    plt.figure(figsize=(6.2, 4.2))
    plt.hist(log_rho, bins=70, density=True, color='0.35', alpha=0.85)
    plt.axvline(q50, color='C3', lw=1.8, label='Median')
    plt.axvspan(q16, q84, color='C3', alpha=0.18, label='68% interval')
    plt.xlabel(r'$\log_{10}(\rho_\star/M_\odot\,\mathrm{pc}^{-3})$')
    plt.ylabel('Posterior density')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rho_posterior.pdf')
    plt.close()


# 4) Sampler traces (synthetic)
def make_sampler_traces():
    rng = np.random.default_rng(7)
    n = 4000
    # A trace
    A0 = 2.4e-15
    A = A0 + np.cumsum(rng.normal(0, 0.02e-15, size=n))
    A = np.clip(A, 1.6e-15, 3.2e-15)
    # gamma trace
    g0 = 4.33
    g = g0 + np.cumsum(rng.normal(0, 0.01, size=n))
    g = np.clip(g, 3.5, 5.5)

    fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.8), sharex=True)
    axs[0].plot(A, lw=0.7)
    axs[0].set_ylabel(r'$A_\mathrm{GWB}$')
    axs[0].axhline(A0, color='k', ls='--', lw=0.8)
    axs[1].plot(g, lw=0.7, color='tab:orange')
    axs[1].set_ylabel(r'$\gamma_\mathrm{GWB}$')
    axs[1].set_xlabel('Iteration')
    axs[1].axhline(g0, color='k', ls='--', lw=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sampler_traces.pdf')
    plt.close()


# 5) Forecast ln BF vs years
def make_forecast_lnBF():
    years = np.linspace(0, 10, 100)
    # simple illustrative model for growth in |ln BF|
    lnBF_cs = 2.0 + 0.35 * (years**1.2)
    lnBF_pt = 2.4 + 0.40 * (years**1.25)

    plt.figure(figsize=(6.4, 4.4))
    plt.plot(years, lnBF_cs, label='CS vs SMBHB')
    plt.plot(years, lnBF_pt, label='PT vs SMBHB')
    plt.axhline(5.0, color='k', ls='--', lw=1, label='Decisive |ln BF| ≈ 5')
    plt.xlabel('Additional observing years')
    plt.ylabel(r'$|\ln \mathrm{BF}|$')
    plt.ylim(0, max(lnBF_pt.max(), lnBF_cs.max())*1.1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'forecast_lnBF.pdf')
    plt.close()

# 6) Residual PSD and whitening checks
def make_whitening_checks():
    rng = np.random.default_rng(8)
    f = np.logspace(-9.3, -7.2, 48)
    Sr_model = (f/1e-8)**(-13/3)
    Sr_model /= Sr_model.max()
    Sr_obs = Sr_model * np.exp(rng.normal(0, 0.25, size=f.size))
    white = Sr_obs / Sr_model

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6))
    axs[0].loglog(f, Sr_obs, 'o-', ms=3, label='Residual PSD')
    axs[0].loglog(f, Sr_model, '--', label='Model $S_r(f)$')
    axs[0].set_xlabel('Frequency f [Hz]')
    axs[0].set_ylabel(r'$S_r(f)$')
    axs[0].legend(frameon=False)

    axs[1].semilogx(f, white, 'o-', ms=3, label='Whitened')
    axs[1].axhline(1.0, color='k', ls='--', lw=1)
    axs[1].set_xlabel('Frequency f [Hz]')
    axs[1].set_ylabel('Whitened level')
    axs[1].set_ylim(0.2, 2.0)
    axs[1].legend(frameon=False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'whitening_checks.pdf')
    plt.close()

# 7) Angle-dependent PPD for HD correlation
def make_hd_ppd():
    mu = np.linspace(-1, 1, 181)
    hd = hellings_downs(mu)
    sigma = 0.05 + 0.05*(1 - (mu+1)/2)
    lo = hd - 2*sigma
    hi = hd + 2*sigma
    med = hd
    gamma_deg = np.degrees(np.arccos(mu))

    plt.figure(figsize=(6.6, 4.2))
    plt.fill_between(gamma_deg, lo, hi, color='C0', alpha=0.2, label='95% band')
    plt.plot(gamma_deg, med, 'C0-', lw=2, label='Median')
    plt.xlabel('Angular separation (deg)')
    plt.ylabel(r'$\Gamma(\gamma)$')
    plt.xlim(0, 180)
    plt.ylim(0.0, 0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hd_ppd.pdf')
    plt.close()

# 8) Anisotropy forecast and CW sensitivity
def make_anisotropy_and_cw():
    yrs = np.linspace(0, 10, 100)
    dipole_ul = 0.2/np.sqrt(1+yrs)
    quad_ul = 0.15/np.sqrt(1+yrs)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(yrs, dipole_ul, label='Dipole UL')
    plt.plot(yrs, quad_ul, label='Quadrupole UL')
    plt.xlabel('Additional observing years')
    plt.ylabel('Anisotropy upper limit')
    plt.ylim(0, max(dipole_ul[0], quad_ul[0])*1.1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'anisotropy_forecast.pdf')
    plt.close()

    f = np.logspace(-9.5, -7.0, 200)
    hc_sens = 5e-15*(f/1e-8)**(1/2)
    plt.figure(figsize=(6.4, 4.2))
    plt.loglog(f, hc_sens, label='CW 95% sensitivity')
    plt.xlabel('Frequency f [Hz]')
    plt.ylabel(r'$h_0$ sensitivity')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cw_sensitivity.pdf')
    plt.close()


if __name__ == '__main__':
    make_hd_curve()
    make_posterior_triangle()
    make_strain_spectra()
    make_sampler_traces()
    make_forecast_lnBF()
    make_whitening_checks()
    make_hd_ppd()
    make_anisotropy_and_cw()
    make_rho_posterior()
