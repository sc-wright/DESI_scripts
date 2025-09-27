import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from import_custom_catalog import CC
from sample_masks import (BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK,
                          BGS_SFR_MASK, BGS_MASS_MASK,
                          BGS_SNR_MASK, LO_Z_MASK, HI_Z_MASK,
                          Z50, Z90, M50, M90, SFR50, SFR90)
from sample_masks import bgs_ne_snr_cut




# New fit using scipy.optimize that adds 2 more parameters
def sfrsd_ne_mass_model(mass, sfrsd, b0, b1, b2, b3, b4, b5, a, b):
    """
    Model: log ne = (b0 + b1*m) + (b2 + b3*m)*x' + (b4 + b5*m)*x'^2
    where x' = sfrsd + a + b*(m - 10).
    """
    x_shifted = sfrsd + (a + b * (mass - 10.0))

    return ((b0 + b1 * mass)
            + (b2 + b3 * mass) * x_shifted
            + (b4 + b5 * mass) * x_shifted**2)

def sfrsd_ne_mass_model_fitting(mass, sfrsd, ne, sigma=None, p0=None, quiet=False):
    """
    Nonlinear fit to the model above using curve_fit.

    Parameters
    ----------
    mass, sfrsd, ne : 1D arrays (raw data)
    sigma : optional 1D array of y-errors for weighting (per-point); if provided,
            absolute_sigma=True is used so uncertainties are in data units
    p0 : optional initial guess of length 8 [b0,b1,b2,b3,b4,b5,a,b]
    quiet : if False, pretty-print coefficients with 1σ uncertainties

    Returns
    -------
    coeffs : array([b0,b1,b2,b3,b4,b5,a,b])
    perr   : 1σ uncertainties for the coefficients
    cov    : covariance matrix
    """
    mass  = np.ravel(mass).astype(float)
    sfrsd = np.ravel(sfrsd).astype(float)
    ne    = np.ravel(ne).astype(float)

    if p0 is None:
        # Simple, robust defaults; tweak if convergence is slow
        #p0 = [np.nanmedian(ne), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p0 = [0, 0, 0, 0, 0, 0, 0, 0]

    coeffs, cov = curve_fit(
        # pack (mass, sfrsd) into a single xdata and unpack inside lambda
        lambda xdata, b0, b1, b2, b3, b4, b5, a, b: sfrsd_ne_mass_model(
            xdata[0], xdata[1], b0, b1, b2, b3, b4, b5, a, b
        ),
        (mass, sfrsd),
        ne,
        p0=p0,
        sigma=sigma,
        absolute_sigma=(sigma is not None),
        maxfev=20000
    )

    # 1σ uncertainties from covariance
    perr = np.sqrt(np.diag(cov))

    if not quiet:
        names = ["b0","b1","b2","b3","b4","b5","a","b"]
        print("Fit coefficients with 1σ uncertainties:")
        for n, v, e in zip(names, coeffs, perr):
            print(f"  {n:>2} = {v:.6g} ± {e:.6g}")

    return coeffs, cov


def predict_ne(mass, sfrsd, coeffs):
    """
    Predict log ne given mass, sfrsd, and coeffs = (b0..b5, a, b).
    """
    b0, b1, b2, b3, b4, b5, a, b = coeffs
    pivot = np.average(mass)
    #print(pivot)
    x_shifted = sfrsd + a + b * (mass - pivot)

    return ((b0 + b1 * mass)
            + (b2 + b3 * mass) * x_shifted
            + (b4 + b5 * mass) * x_shifted**2)


def bin_sfrsd_ne_mass_data(mass, sfrsd, ne, mass_edges, sfrsd_edges):
    # This is for creating bins for the raw data and calculating median and error
    # For the purposes of plotting/display - these bins are not actually used for the fit
    centers_m = []
    centers_sfr = []
    means = []
    errs = []
    counts = []

    for i in range(len(mass_edges)-1):
        for j in range(len(sfrsd_edges)-1):
            mask = ((mass >= mass_edges[i]) & (mass < mass_edges[i+1]) &
                    (sfrsd >= sfrsd_edges[j]) & (sfrsd < sfrsd_edges[j+1]))
            values = ne[mask]

            centers_m.append(0.5*(mass_edges[i] + mass_edges[i+1]))
            centers_sfr.append(0.5*(sfrsd_edges[j] + sfrsd_edges[j+1]))
            counts.append(len(values))

            if len(values) > 0:
                mean = np.average(values)
                err = (np.std(values, ddof=1) / np.sqrt(len(values))
                       if len(values) >= 10 else np.nan)
            else:
                mean = np.nan
                err = np.nan

            means.append(mean)
            errs.append(err)

    return (np.array(centers_m), np.array(centers_sfr),
            np.array(means), np.array(errs), np.array(counts))


def sfrsd_ne_mass_model_plotting(sample_mask=BGS_SNR_MASK):

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length

    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = np.array((BGS_SNR_MASK) & (mass >= M50) & (z <= Z50))
        zmax = Z50
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = np.array((BGS_SNR_MASK) & (mass >= M90) & (z <= Z90))
        zmax = Z90

    mass = mass[sample_mask]
    sfrsd = sfr_sd[sample_mask]
    ne = ne[sample_mask]

    fs = 20

    mass_bin = 0.5
    sfrsd_bin = 0.25

    mass_edges = np.arange(9.0, 11.5 + mass_bin, mass_bin)  # 0.5 dex bins
    sfrsd_edges = np.arange(-2.0, 0.0 + sfrsd_bin, sfrsd_bin)  # 0.25 dex bins

    # Set plot values using bin_data_full function
    centers_m, centers_sfr, medians, errs, counts = bin_sfrsd_ne_mass_data(
        mass, sfrsd, ne, mass_edges, sfrsd_edges
    )

    # Print out bins if desired
    #for m, s, n, c in zip(centers_m, centers_sfr, means, counts):
    #    print(f"Mass bin center={m:.2f}, SFR bin center={s:.2f}, "
    #          f"ne={n:.3f}, count={c}")
    #for m, s, n, c in zip(centers_m, centers_sfr, medians, counts):
    #    print(f"logM={m:.2f}, logSFRsd={s:.2f}, count={c}, ne={n}")

    coeffs, cov = sfrsd_ne_mass_model_fitting(mass, sfrsd, ne, quiet=True)

    # Predicted values at each data point
    y_pred = predict_ne(mass, sfrsd, coeffs)

    # Residuals
    residuals = ne - y_pred

    # Define colors for bins
    colors = plt.cm.winter(np.linspace(0, 1, len(mass_edges) - 1))

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(len(mass_edges) - 1):
        m_mid = 0.5 * (mass_edges[i] + mass_edges[i + 1])
        in_bin = np.isclose(centers_m, m_mid)  # select only this bin’s data

        # Extract data for this bin
        xvals = centers_sfr[in_bin] + 0.01 * (i - (len(mass_edges) - 1) / 2)
        yvals = medians[in_bin]
        yerrs = errs[in_bin]
        cts = counts[in_bin]
        print(cts)

        # Separate good/bad inside this bin
        good = cts >= 10
        bad = cts < 10

        # Plot good points with errorbars
        ax.errorbar(xvals[good], yvals[good], yerr=yerrs[good],
                    fmt='o', color=colors[i], ecolor=colors[i],
                    elinewidth=1, capsize=0)

        # Plot bad points as open circles
        ax.plot(xvals[bad], yvals[bad], 'o',
                mfc='none', mec=colors[i], mew=1)

        # Plot model curve - old version
        #sfr_grid = np.linspace(sfrsd_edges[0], sfrsd_edges[-1], 100)
        #model_curve = predict_ne(m_mid, sfr_grid, coeffs)
        #ax.plot(sfr_grid, model_curve, color=colors[i],
        #        label=f"{mass_edges[i]}–{mass_edges[i + 1]}")

        # Plot model curve
        sfr_grid = np.linspace(sfrsd_edges[0], sfrsd_edges[-1], 100)
        model_curve = predict_ne(m_mid, sfr_grid, coeffs)  # coeffs now has 8 parameters
        if sum(cts) > 9:
            ax.plot(sfr_grid, model_curve, color=colors[i],
                    label=f"{mass_edges[i]}–{mass_edges[i + 1]}")

    ax.set_xlabel(r"$\log \, \Sigma_{\rm SFR}/M_\odot/yr/kpc^2$", fontsize=fs)
    ax.set_ylabel(r"$\log \, n_e / cm^{-3}$", fontsize=fs)
    ax.legend(title=r"$\log{m}$ bins (dex)")
    if sample == 2:
        tit = "low-z sample"
    elif sample == 3:
        tit = "all-z sample"
    elif sample == 1:
        tit = "all galaxies"
    else:
        tit = ""
    plt.title(tit)
    plt.xlim(-2.1, 0.1)
    plt.ylim(1.78, 2.63)
    plt.savefig(f'paper_figures/paper_sfrsd_ne_mass_fits_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    # Plot residuals
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, s=5, alpha=0.5)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Predicted $\log{n_e}$")
    plt.ylabel("Residuals (observed - predicted)")
    plt.title(f"Residuals vs predicted {tit}")
    plt.savefig(f'paper_figures/paper_sfrsd_ne_mass_residual1_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(mass, residuals, s=5, alpha=0.5)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xlabel("$\log{M_\star}$")
    axes[0].set_ylabel("Residuals")

    axes[1].scatter(sfrsd, residuals, s=5, alpha=0.5)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("$\log{\Sigma_{SFR}}$")
    axes[1].set_ylabel("Residuals")

    plt.suptitle(f"Residuals vs predictors {tit}")
    plt.savefig(f'paper_figures/paper_sfrsd_ne_mass_residual2_{sample}.png', dpi=PLOT_DPI)

    plt.show()

def fit_to_binned_data(sample_mask=BGS_SNR_MASK):

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length

    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = np.array((BGS_SNR_MASK) & (mass >= M50) & (z <= Z50))
        zmax = Z50
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = np.array((BGS_SNR_MASK) & (mass >= M90) & (z <= Z90))
        zmax = Z90
    else:
        sample = 0

    mass = mass[sample_mask]
    sfrsd = sfr_sd[sample_mask]
    ne = ne[sample_mask]

    fs = 20

    mass_bin = 0.5
    sfrsd_bin = 0.25

    mass_edges = np.arange(9.0, 11.5 + mass_bin, mass_bin)  # 0.5 dex bins
    sfrsd_edges = np.arange(-2.0, 0.0 + sfrsd_bin, sfrsd_bin)  # 0.25 dex bins

    # Set plot values using bin_data_full function
    centers_m, centers_sfr, medians, errs, counts = bin_sfrsd_ne_mass_data(
        mass, sfrsd, ne, mass_edges, sfrsd_edges
    )

    # Try fit to binned data as diagnostic
    #########################################################
    # bin_data_full gives centers_m, centers_sfr, medians, errs, counts
    # keep only bins with count>0 (or >=N)
    mask = ~np.isnan(medians)
    mass_bin = centers_m[mask]
    sfrsd_bin = centers_sfr[mask]
    ne_bin = medians[mask]
    sigma_bin = errs[mask]  # may have NaNs for small counts -> handle appropriately

    # remove bins without error estimates or use a fixed small sigma for them
    valid = ~np.isnan(sigma_bin)
    mass_b = mass_bin[valid]
    sfr_b = sfrsd_bin[valid]
    ne_b = ne_bin[valid]
    sig_b = sigma_bin[valid]

    coeffs, cov = sfrsd_ne_mass_model_fitting(mass_b, sfr_b, ne_b, quiet=True)

    # Predicted values at each data point
    y_pred = predict_ne(mass, sfrsd, coeffs)

    # Residuals
    residuals = ne - y_pred

    # Define colors for bins
    colors = plt.cm.winter(np.linspace(0, 1, len(mass_edges) - 1))

    fig, ax = plt.subplots(figsize=(6, 6))

    # This is no longer needed
    #sfr_grid = np.linspace(sfrsd_edges[0], sfrsd_edges[-1], 100)

    for i in range(len(mass_edges) - 1):
        m_mid = 0.5 * (mass_edges[i] + mass_edges[i + 1])
        in_bin = np.isclose(centers_m, m_mid)  # select only this bin’s data

        # Extract data for this bin
        xvals = centers_sfr[in_bin] + 0.01 * (i - (len(mass_edges) - 1) / 2)
        yvals = medians[in_bin]
        yerrs = errs[in_bin]
        cts = counts[in_bin]

        # Separate good/bad inside this bin
        good = cts >= 10
        bad = cts < 10

        # Plot good points with errorbars
        ax.errorbar(xvals[good], yvals[good], yerr=yerrs[good],
                    fmt='o', color=colors[i], ecolor=colors[i],
                    elinewidth=1, capsize=0)

        # Plot bad points as open circles
        ax.plot(xvals[bad], yvals[bad], 'o',
                mfc='none', mec=colors[i], mew=1)

        # Plot model curve - old version
        #sfr_grid = np.linspace(sfrsd_edges[0], sfrsd_edges[-1], 100)
        #model_curve = predict_ne(m_mid, sfr_grid, coeffs)
        #ax.plot(sfr_grid, model_curve, color=colors[i],
        #        label=f"{mass_edges[i]}–{mass_edges[i + 1]}")

        # Plot model curve
        sfr_grid = np.linspace(sfrsd_edges[0], sfrsd_edges[-1], 100)
        model_curve = predict_ne(m_mid, sfr_grid, coeffs)  # coeffs now has 8 parameters
        if sum(cts) > 9:
            ax.plot(sfr_grid, model_curve, color=colors[i],
                        label=f"{mass_edges[i]}–{mass_edges[i + 1]}")

    ax.set_xlabel(r"$\log \, \Sigma_{\rm SFR}/M_\odot/yr/kpc^2$", fontsize=fs)
    ax.set_ylabel(r"$\log \, n_e / cm^{-3}$", fontsize=fs)
    if sample == 2:
        tit = "low-z sample"
    elif sample == 3:
        tit = "all-z sample"
    elif sample == 1:
        tit = "all bgs snr > 5 galaxies "
    else:
        tit = ""
    plt.title(f"{tit} (fit to binned)")
    ax.legend(title=r"$\log{m}$ bins (dex)")
    plt.savefig(f'paper_figures/paper_sfrsd_ne_mass_binned_fits_{sample}.png', dpi=PLOT_DPI)

    plt.show()

    # Plot residuals
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, s=5, alpha=0.5)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Predicted $\log{n_e}$")
    plt.ylabel("Residuals (observed - predicted)")
    plt.title(f"Residuals vs predicted {tit} (fit to binned)")
    plt.savefig(f'paper_figures/paper_sfrsd_ne_mass_binned_residual1_{sample}.png', dpi=PLOT_DPI)

    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(mass, residuals, s=5, alpha=0.5)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xlabel("$\log{M_\star}$")
    axes[0].set_ylabel("Residuals")

    axes[1].scatter(sfrsd, residuals, s=5, alpha=0.5)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("$\log{\Sigma_{SFR}}$")
    axes[1].set_ylabel("Residuals")

    plt.suptitle(f"Residuals vs predictors {tit} (fit to binned)")
    plt.savefig(f'paper_figures/paper_sfrsd_ne_mass_binned_residual2_{sample}.png', dpi=PLOT_DPI)
    plt.show()


if __name__ == "__main__":
    PLOT_DPI = 500
    sfrsd_ne_mass_model_plotting(LO_Z_MASK)
    sfrsd_ne_mass_model_plotting(HI_Z_MASK)
    #fit_to_binned_data(HI_Z_MASK)
    #sfrsd_ne_mass_model_plotting(BGS_SNR_MASK)