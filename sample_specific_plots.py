import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects
from pride_colors_plotly.pride_colors import pride_colors_matplotlib

import numpy as np

from scipy.stats import binned_statistic_2d

from import_custom_catalog import CC
from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from calculation_scripts import sfr_ms, distance_from_ms
from sample_masks import (BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK,
                          BGS_SFR_MASK, BGS_MASS_MASK,
                          BGS_SNR_MASK, LO_Z_MASK, HI_Z_MASK,
                          Z50, Z90, M50, M90, SFR50, SFR90)
from sample_masks import bgs_ne_snr_cut

def plot_ne_distribution(sample_mask=BGS_SNR_MASK):
    """
    This plots a histogram of the ne distribution for the given sample mask
    :param snr_mask: default mask is the full snr>5 sample at all redshifts
    :return: none
    """

    ne, _ = bgs_ne_snr_cut()

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    # Statistics:
    #print("Median:", np.median(ne[sample_mask]))
    #print("Mean:", np.average(ne[sample_mask]))
    #print("Stdev:", np.std(ne[sample_mask]))

    plt.hist(ne[sample_mask], bins=100)
    plt.xlim(0, 3.5)
    if sample == 2:
        plt.title(f"Electron density distribution (low-z, {len(ne[sample_mask])} galaxies)")
    elif sample == 3:
        plt.title(f"Electron density distribution (all-z, {len(ne[sample_mask])} galaxies)")
    else:
        plt.title(f"Electron density distribution ({len(ne[sample_mask])} galaxies)")
    plt.xlabel(r"$\log{n_e} ~[\mathrm{cm}^{-3}]$")
    plt.show()


def plot_sfrsd_distribution(sample_mask=BGS_SNR_MASK):
    sfrsd = CC.catalog['SFR_SD'][BGS_MASK]

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    sfrsd = sfrsd[sample_mask]

    plt.hist(sfrsd, bins=100)
    if sample == 2:
        plt.title(r"$\Sigma_{SFR}$" + f" (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        plt.title(r"$\Sigma_{SFR}$" + f" (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(r"$\Sigma_{SFR}$" + f" ({sum(sample_mask)} galaxies)")
    plt.xlabel(r"$\log{M_\odot/yr/kpc^2}$")
    plt.xlim(-2.4, 0.1)
    plt.show()


def plot_redshift_vs_ne_mass_sfr(sample_mask=BGS_SNR_MASK):
    """
    Plots electron density vs redshift
    :return: none
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    redshift = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    ne = ne[sample_mask]
    redshift = redshift[sample_mask]

    #m, b = np.polyfit(redshift, ne, 1)
    #fit_x = np.linspace(0,.5,3)
    #fit_y = m * fit_x + b

    plt.hist2d(redshift, ne, bins=100, norm=mpl.colors.LogNorm())
    #plt.plot(fit_x, fit_y, color='r', label='linear fit (slope = {:.3f})'.format(m))
    if sample == 2:
        plt.vlines(Z50, 0, 3.5, color='b')
        plt.title(f'Electron density vs redshift (low-z, {sum(sample_mask)} galaxies)')
    elif sample == 3:
        plt.vlines(Z90, 0, 3.5, color='r')
        plt.title(f'Electron density vs redshift (all-z, {sum(sample_mask)} galaxies)')
    else:
        plt.title(f'Electron density vs redshift ({sum(sample_mask)} galaxies)')
    plt.xlabel("z")
    plt.ylabel(r'$n_e$ ($\log{}$cm$^{-3}$)')
    #plt.legend(loc='lower right')
    plt.xlim(0, 0.4)
    plt.ylim(0, 3.5)
    #plt.savefig('figures/ne_plots/ne_redshift.png')
    plt.show()

    plt.hist2d(redshift, mass, bins=100, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(Z50, M50, 13, color='b')
        plt.hlines(M50, 0, Z50, color='b')
        plt.title(f'Stellar mass vs redshift (low-z, {sum(sample_mask)} galaxies)')
    elif sample == 3:
        plt.vlines(Z90, M90, 13, color='r')
        plt.hlines(M90, 0, Z90, color='r')
        plt.title(f'Stellar mass vs redshift (all-z, {sum(sample_mask)} galaxies)')
    else:
        plt.title(f'Stellar mass vs redshift ({sum(sample_mask)} galaxies)')
    plt.xlabel("z")
    plt.ylabel(r'$\log{M_\star/M_\odot}$')
    plt.xlim(0, 0.4)
    plt.ylim(7, 11.5)
    #plt.savefig('figures/ne_plots/mass_redshift.png')
    plt.show()

    plt.hist2d(redshift, sfr, bins=100, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(Z50, SFR50, 10, color='b')
        plt.hlines(SFR50, 0, Z50, color='b')
        plt.title(f'SFR vs redshift (low-z, {sum(sample_mask)} galaxies)')
    elif sample == 3:
        plt.vlines(Z90, SFR90, 10, color='r')
        plt.hlines(SFR90, 0, Z90, color='r')
        plt.title(f'SFR vs redshift (all-z, {sum(sample_mask)} galaxies)')
    else:
        plt.title(f'SFR vs redshift ({sum(sample_mask)} galaxies)')
    plt.xlabel("z")
    plt.ylabel(r'$\log{M_\star/yr}$')
    plt.xlim(0, 0.4)
    plt.ylim(-2, 2)
    #plt.savefig('figures/ne_plots/sfr_redshift.png')
    plt.show()


def plot_mass_sfr_sfrsd_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Plot mass, sfr, sfrsd vs ne with percentile trendlines
    :return: none
    """
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK & (z < Z50)
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK & (z < Z90)

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    sfr_sd = sfr_sd[sample_mask]
    z = z[sample_mask]
    ne = ne[sample_mask]

    # set percentile line color
    colr = 'magenta'

    # Plot ne vs mass

    massmin = 8.0
    massmax = 11.5

    # Calculate 25/50/75th percentiles
    ne_75 = []
    ne_50 = []
    ne_25 = []
    mrange = []

    b = 0.1
    for i in np.arange(massmin, massmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(mass >= i, mass < i + b)],
                                          (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            mrange.append(i + b * 0.5)
        except IndexError:
            pass

    plt.hist2d(mass, ne, bins=200, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(M50, 0, 3.5, color='b')
        plt.title(f'$n_e$ vs $M_\star$ (low-z, {sum(sample_mask)} galaxies)')
    elif sample == 3:
        plt.vlines(M90, 0, 3.5, color='r')
        plt.title(f'$n_e$ vs $M_\star$ (all-z, {sum(sample_mask)} galaxies)')
    else:
        plt.title(f'$n_e$ vs $M_\star$ ({sum(sample_mask)} galaxies)')
    plt.plot(mrange, ne_25, color='white', linestyle='dashed', linewidth=3)
    plt.plot(mrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(mrange, ne_50, color='white', linewidth=3)
    plt.plot(mrange, ne_50, color=colr)
    plt.plot(mrange, ne_75, color='white', linestyle='dashed', linewidth=3)
    plt.plot(mrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{M_\star/M_\odot}$')
    plt.ylabel(r'$n_e$ ($\log{}$cm$^{-3}$)')
    plt.xlim(massmin, massmax)
    plt.ylim(1, 3)
    plt.show()

    # Plot ne vs sfr

    sfrmin = -1.5
    sfrmax = 2.0

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrrange = []

    for i in np.arange(sfrmin, sfrmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(sfr >= i, sfr < i + b)], (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            sfrrange.append(i + b * 0.5)
        except IndexError:
            pass

    plt.hist2d(sfr, ne, bins=200, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(SFR50, 0, 3.5, color='b')
        plt.title(f'$n_e$ vs SFR (low-z, {sum(sample_mask)} galaxies)')
    elif sample == 3:
        plt.vlines(SFR90, 0, 3.5, color='r')
        plt.title(f'$n_e$ vs SFR (all-z, {sum(sample_mask)} galaxies)')
    else:
        plt.title(f'$n_e$ vs SFR ({sum(sample_mask)} galaxies)')
    plt.plot(sfrrange, ne_25, color='white', linestyle='dashed', linewidth=3)
    plt.plot(sfrrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrrange, ne_50, color='white', linewidth=3)
    plt.plot(sfrrange, ne_50, color=colr)
    plt.plot(sfrrange, ne_75, color='white', linestyle='dashed', linewidth=3)
    plt.plot(sfrrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{M_\odot/yr}$')
    plt.ylabel(r'$n_e$ ($\log{}$cm$^{-3}$)')
    plt.xlim(sfrmin, sfrmax)
    plt.ylim(1, 3)
    plt.show()


    # Plot ne vs sfr_sd

    # Change mask to full mass and sfr cuts for this plot so we are only plotting fully complete range
    if sample == 2:
        sample_mask = LO_Z_MASK
    elif sample == 3:
        sample_mask = HI_Z_MASK

    sfrsdmin = -2.25
    sfrsdmax = -.25

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrsdrange = []

    for i in np.arange(sfrsdmin, sfrsdmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(sfr_sd >= i, sfr_sd < i + b)], (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            sfrsdrange.append(i + b * 0.5)
        except IndexError:
            pass

    plt.hist2d(sfr_sd, ne, bins=200, norm=mpl.colors.LogNorm())
    if sample == 2:
        #plt.vlines(SFR50, 0, 3.5, color='b')
        plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (low-z, {sum(sample_mask)} galaxies)')
    elif sample == 3:
        #plt.vlines(SFR90, 0, 3.5, color='r')
        plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (all-z, {sum(sample_mask)} galaxies)')
    else:
        plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' ({sum(sample_mask)} galaxies)')
    plt.plot(sfrsdrange, ne_25, color='white', linestyle='dashed', linewidth=3)
    plt.plot(sfrsdrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrsdrange, ne_50, color='white', linewidth=3)
    plt.plot(sfrsdrange, ne_50, color=colr)
    plt.plot(sfrsdrange, ne_75, color='white', linestyle='dashed', linewidth=3)
    plt.plot(sfrsdrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r"$\log{M_\odot/yr/kpc^2}$")
    plt.ylabel(r'$n_e$ ($\log{}$cm$^{-3}$)')
    plt.xlim(sfrsdmin, sfrsdmax)
    plt.ylim(1, 3)
    plt.show()


def plot_sfr_vs_mass_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Generates a plot of sfr vs mass with each bin color-coded by median ne
    :param sample_mask: Optional extra mask that is placed after snr cuts
    :return:
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]  # bgs length
    ne, _ = bgs_ne_snr_cut()
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK & (z < Z50)
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK & (z < Z90)

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    ne = ne[sample_mask]

    o1, o2, c = sfr_ms()

    x_ms = np.linspace(0,20,100)
    y_ms = o1 * x_ms**2 + o2 * x_ms + c

    # Define the number of bins
    x_bins = 80
    y_bins = 60

    # font size for labels
    fs = 16

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr, ne, statistic='median', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    #ax.set_facecolor('gray')
    plt.pcolormesh(X, Y, stat.T, cmap=pride_colors_matplotlib(flag='trans'), shading='auto', vmin=1.824, vmax=2.224)
    plt.plot(x_ms, y_ms, 'white', linewidth='2.5')
    # If using samples 2 or 3, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 11.5, color='b')
        plt.vlines(M50, SFR50, 1.5, color='b')
        plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        plt.hlines(SFR90, M90, 11.5, color='r')
        plt.vlines(M90, SFR90, 1.5, color='r')
        plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" ({sum(sample_mask)} galaxies)")
    plt.xlim(8, 11.5)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$ ($\log{}$cm$^{-3}$)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{M_\odot/yr}$', size=fs)
    #plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_ne_narrow.png')
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr, ne, statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 11.5, color='b')
        plt.vlines(M50, SFR50, 1.5, color='b')
        plt.title(r"SFR vs $M_\star$" + f" vs count per bin (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        plt.hlines(SFR90, M90, 11.5, color='r')
        plt.vlines(M90, SFR90, 1.5, color='r')
        plt.title(r"SFR vs $M_\star$" + f" vs count per bin (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(rf"SFR vs $M_\star$" + f" vs count per bin ({sum(sample_mask)} galaxies)")
    plt.xlim(8, 11.5)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{M_\odot/yr}$', size=fs)
    #plt.savefig('figures/ne_plots/sfr_vs_mstar_counts.png')
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr, ne, statistic=iqr, bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)#, norm=mpl.colors.LogNorm())
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 11.5, color='b')
        plt.vlines(M50, SFR50, 1.5, color='b')
        plt.title(rf"SFR vs $M_\star$ vs inter-quartile range (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        plt.hlines(SFR90, M90, 11.5, color='r')
        plt.vlines(M90, SFR90, 1.5, color='r')
        plt.title(rf"SFR vs $M_\star$ vs inter-quartile range (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(rf"SFR vs $M_\star$ vs inter-quartile range ({sum(sample_mask)} galaxies)")
    plt.xlim(8, 11.5)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{M_\odot/yr}$', size=fs)
    #plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_iqr.png')
    plt.show()


def plot_sfrsd_vs_mass_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Generates a plot of sfr surface density vs mass with each bin color-coded by median ne
    :param sample_mask: Optional extra mask that is placed after snr cuts
    :return:
    """
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]  # bgs length
    z = CC.catalog['Z'][BGS_MASK]
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK & (z < Z50)
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK & (z < Z90)

    mass = mass[sample_mask]
    sfr_sd = sfr_sd[sample_mask]
    ne = ne[sample_mask]

    # Define the number of bins
    x_bins = 80
    y_bins = 40

    # font size for labels
    fs = 16

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr_sd, ne, statistic='median', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap=pride_colors_matplotlib(flag='rainbow'), shading='auto', vmin=1.95, vmax=2.35)
    ax.set_facecolor('gray')
    if sample == 2:
        #plt.hlines(SFR50, M50, 11.5, color='b')
        plt.vlines(M50, -20, 1.5, color='b')
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        #plt.hlines(SFR90, M90, 11.5, color='r')
        plt.vlines(M90, -20, 1.5, color='r')
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" ({sum(sample_mask)} galaxies)")
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r"$\log{M_\odot/yr/kpc^2}$", size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_ne_narrow.png')
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr_sd, ne, statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    if sample == 2:
        #plt.hlines(SFR50, M50, 11.5, color='b')
        plt.vlines(M50, -20, 1.5, color='b')
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        #plt.hlines(SFR90, M90, 11.5, color='r')
        plt.vlines(M90, -20, 1.5, color='r')
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count ({sum(sample_mask)} galaxies)")
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r"$\log{M_\odot/yr/kpc^2}$", size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_counts.png')
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr_sd, ne, statistic=iqr, bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)#, norm=mpl.colors.LogNorm())
    if sample == 2:
        #plt.hlines(SFR50, M50, 11.5, color='b')
        plt.vlines(M50, -20, 1.5, color='b')
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs IQR (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        #plt.hlines(SFR90, M90, 11.5, color='r')
        plt.vlines(M90, -20, 1.5, color='r')
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs IQR (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs IQR ({sum(sample_mask)} galaxies)")
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r"$\log{M_\odot/yr/kpc^2}$", size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_iqr.png')
    plt.show()



def plot_ne_histogram_binned(sample_mask=BGS_SNR_MASK):

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]  # bgs length
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    ne = ne[sample_mask]

    o1, o2, c = sfr_ms()

    distance = distance_from_ms(mass, sfr, o1, o2, c)

    # Set mass ranges for both plots

    mgap = 1
    masses = np.arange(8, 12, mgap)
    xmin = 1
    xmax = 3

    bin_width = 0.02

    # FIRST PLOT - Delta SFR

    for m in masses:
        mll = m
        mul = m + mgap
        mcount = len(np.where(generate_combined_mask(mass >= mll, mass < mul) == True)[0])

        dgap = 0.5
        dists = np.arange(1, -1.5, -dgap)
        n = len(dists)
        fig, axs = plt.subplots(n, 1, figsize=(6, 2 * n), sharex=True, gridspec_kw={'hspace': 0})
        if n == 1:
            axs = [axs]
        for i, (d, ax) in enumerate(zip(dists, axs)):
            dll = d
            dul = d + dgap
            # Create bin for particular plot
            mask = generate_combined_mask(mass >= mll, mass < mul, distance >= dll, distance < dul)

            data = ne[mask]
            ax.set_xlim(xmin, xmax)
            if data.size > 0:
                bins = np.arange(min(data), max(data) + bin_width, bin_width)
                median = np.median(data)
                mean = np.average(data)
            else:
                # Fallback bins if data is empty
                bins = np.linspace(0, 1, 10)  # arbitrary fallback just to render an empty plot
                median = 0
                mean = 0
            ax.axvline(median, color='red', linestyle='dashed', linewidth=1.5, label=f'Median = {median:.3f}')
            ax.axvline(mean, color='purple', linestyle='dashed', linewidth=1.5, label=f'Mean = {mean:.3f}')
            ax.hist(data, bins=bins)
            ax.text(0.02, 0.95, fr'{dll}$ \leq \Delta $SFR $ < ${dul} ({len(data)})', transform=axs[i].transAxes,
                    fontsize=12, verticalalignment='top')
            ax.legend(loc='upper right', fontsize=8)
        if sample == 2:
            fig.suptitle(f'{mll} $\leq M_\star < $ {mul} (low-z, {mcount} galaxies)', fontsize=16, y=.97)
        elif sample == 3:
            fig.suptitle(f'{mll} $\leq M_\star < $ {mul} (all-z, {mcount} galaxies)', fontsize=16, y=.97)
        else:
            fig.suptitle(f'{mll} $\leq M_\star < $ {mul} ({mcount})', fontsize=16, y=.97)
        plt.xlabel(r'$\log{n_e}$', fontsize=16)
        plt.tight_layout()

        plt.savefig(f'figures/ne_plots/sample_{str(sample)}_ne_delta_sfr_mass_{m}_{m+mgap}.png')
        plt.show()

    # SECOND PLOT - SFR

    for m in masses:
        mll = m
        mul = m + mgap
        mcount = len(np.where(generate_combined_mask(mass >= mll, mass < mul) == True)[0])

        sfrgap = 0.5
        sfrs = np.arange(1, -1.5, -sfrgap)
        n = len(sfrs)
        fig, axs = plt.subplots(n, 1, figsize=(6, 2 * n), sharex=True, gridspec_kw={'hspace': 0})
        if n == 1:
            axs = [axs]
        for i, (d, ax) in enumerate(zip(sfrs, axs)):
            dll = d
            dul = d + sfrgap
            # Create bin for particular plot
            mask = generate_combined_mask(mass >= mll, mass < mul, sfr >= dll, sfr < dul)
            data = ne[mask]
            ax.set_xlim(xmin, xmax)
            if data.size > 0:
                bins = np.arange(min(data), max(data) + bin_width, bin_width)
                median = np.median(data)
                mean = np.average(data)
            else:
                # Fallback bins if data is empty
                bins = np.linspace(0, 1, 10)  # arbitrary fallback just to render an empty plot
                median = 0
                mean = 0
            ax.axvline(median, color='red', linestyle='dashed', linewidth=1.5, label=f'Median = {median:.3f}')
            ax.axvline(mean, color='purple', linestyle='dashed', linewidth=1.5, label=f'Mean = {mean:.3f}')
            ax.hist(data, bins=bins)
            ax.text(0.02, 0.95, fr'{dll}$ \leq $SFR$ < ${dul} ({len(data)})', transform=axs[i].transAxes,
                    fontsize=12, verticalalignment='top')
            ax.legend(loc='upper right', fontsize=8)
        if sample == 2:
            fig.suptitle(f'{mll} $\leq M_\star < $ {mul} (low-z, {mcount} galaxies)', fontsize=16, y=.97)
        elif sample == 3:
            fig.suptitle(f'{mll} $\leq M_\star < $ {mul} (all-z, {mcount} galaxies)', fontsize=16, y=.97)
        else:
            fig.suptitle(f'{mll} $\leq M_\star < $ {mul} ({mcount})', fontsize=16, y=.97)
        plt.xlabel(r'$\log{n_e}$', fontsize=16)
        plt.tight_layout()

        plt.savefig(f'figures/ne_plots/sample_{str(sample)}_ne_sfr_mass_{m}_{m + mgap}.png')
        plt.show()


def plot_bpt_ne_color(sample_mask=BGS_SNR_MASK):
    """
    Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)
    Color-codes by n_e
    :return: None
    """
    # potentially change this so instead of a flat snr cut we keep uncertainties
    # and find other ways to deal with it
    snr_lim = 3#SNR_LIM

    # Extracting line fluxes from the catalog.
    # All are BGS length
    nii = CC.catalog['NII_6584_FLUX'][BGS_MASK]
    nii_snr = nii * np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK])
    ha = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    oiii = CC.catalog['OIII_5007_FLUX'][BGS_MASK]
    oiii_snr = oiii * np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK])
    hb = CC.catalog['HBETA_FLUX'][BGS_MASK]
    ne, ne_mask = bgs_ne_snr_cut()

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    # all input masks are BGS length
    bpt_mask = generate_combined_mask(sample_mask, nii_snr > snr_lim, oiii_snr > snr_lim)

    nh = np.log10(nii[bpt_mask] / ha[bpt_mask])  # x-axis
    oh = np.log10(oiii[bpt_mask] / hb[bpt_mask]) # y-axis
    ne = ne[bpt_mask]

    hii_boundary = lambda x: 0.61/(x - 0.05) + 1.3          # black dashed
    agn_boundary = lambda x: 0.61 / (x - 0.47) + 1.19       # red dotted
    shock_boundary = lambda x: 2.144507*x + 0.465028        # blue dotdash

    hii_object_mask         = (oh < agn_boundary(nh)) & (oh < hii_boundary(nh))         # below both red and black lines
    agn_object_mask         = (oh > agn_boundary(nh)) & (oh > shock_boundary(nh))       # above both red and blue
    composite_object_mask   = (oh > hii_boundary(nh)) & (oh < agn_boundary(nh))         # above black and below red
    shock_object_mask       = (oh > agn_boundary(nh)) & (oh < shock_boundary(nh))       # above red and below blue

    hii_ne_median = np.median(ne[hii_object_mask])
    agn_ne_median = np.median(ne[agn_object_mask])
    composite_ne_median = np.median(ne[composite_object_mask])
    shock_ne_median = np.median(ne[shock_object_mask])

    # Arrays to plot the separation lines
    x_for_line_1 = np.log10(np.logspace(-5,.049,300))
    hii_agn_line = hii_boundary(x_for_line_1)           # black dashed
    x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
    composite_line_2 = agn_boundary(x_for_line_2)       # red dotted
    x_for_line_3 = np.linspace(-.13,2,100)
    agn_line_3 = shock_boundary(x_for_line_3)           # blue dotdash

    # Define the number of bins
    x_bins = 70
    y_bins = 60

    # font size for labels
    fs = 16

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        nh, oh, ne, statistic='median', bins=[x_bins, y_bins]
    )

    # Creating color map for opaque colorbar
    #cmap = cm.plasma
    cmap = pride_colors_matplotlib(flag='trans')
    #norm = Normalize(vmin=1.5, vmax=2.5)

    f, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('gray')
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap=cmap, shading='auto', vmin=1.824, vmax=2.224)
    plt.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='k')
    plt.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r')
    plt.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b')
    plt.text(-1.3, -0.4, f"H II\n{hii_ne_median:.2f}", fontweight='bold')
    plt.text(-.15, -0.75, f"Composite\n{composite_ne_median:.2f}", fontweight='bold')
    plt.text(-1.0, 1.2, f"AGN\n{agn_ne_median:.2f}", fontweight='bold')
    plt.text(0.15, -0.25, f"Shocks\n{shock_ne_median:.2f}", fontweight='bold')
    #plt.text(0.005, 1.005, f'total: {sum(bpt_mask)}, snr $>$ {snr_lim}',
    #      horizontalalignment='left',
    #      verticalalignment='bottom',
    #      transform=ax.transAxes)
    plt.xlim(-1.75, 0.5)
    plt.ylim(-1, 1.5)
    if sample == 2:
        plt.title(fr'BPT diagram color coded by $n_e$ (low-z, {sum(bpt_mask)} galaxies)')#, fontsize=16)
    elif sample == 3:
        plt.title(f'BPT diagram color coded by $n_e$ (all-z, {sum(bpt_mask)} galaxies)')#, fontsize=16)
    else:
        plt.title(f'BPT diagram color coded by $n_e$ ({sum(bpt_mask)} galaxies)')#, fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$ ($\log{}$cm$^{-3}$)', fontsize=fs)
    #cbar = plt.colorbar(sm, ax=ax, label=r"$n_e$ (cm$^{-3}$)")
    plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$', fontsize=fs)
    plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$', fontsize=fs)
    plt.show()


def plot_extinction_vs_ne(sample_mask=BGS_SNR_MASK):

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]  # bgs length
    a_ha = CC.catalog['A_HALPHA'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    a_ha = a_ha[sample_mask]
    ne = ne[sample_mask]

    plt.hist2d(a_ha, ne, bins=100, norm=mpl.colors.LogNorm())
    plt.xlabel(r'$A(H\alpha)$')
    plt.ylabel(r'$\log{n_e/cm^3}$')
    plt.xlim(0,1.2)
    plt.ylim(0.5, 3.0)
    plt.colorbar(label='count')
    if sample == 2:
        plt.title(rf"$n_e$ vs $H\alpha$ extinction (low-z, {sum(sample_mask)} galaxies)")
    elif sample == 3:
        plt.title(rf"$n_e$ vs $H\alpha$ extinction (all-z, {sum(sample_mask)} galaxies)")
    else:
        plt.title(rf"$n_e$ vs $H\alpha$ extinction ({sum(sample_mask)} galaxies)")
    plt.show()


def plot_ne_vs_sfrsd_binned(sample_mask=BGS_SNR_MASK):

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
    elif sample_mask is HI_Z_MASK:
        sample = 3

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    sfr_sd = sfr_sd[sample_mask]
    ne = ne[sample_mask]

    #mpl_color_wheel = ['#E40303', '#FF8C00', '#FFED00', '#008026', '#004CFF', '#732982']
    mpl_color_wheel = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    mgap = .5
    masses = np.arange(8.5, 11.5, mgap)
    xmin = 1
    xmax = 3

    fig = plt.figure(figsize=(9,6))

    mind = -1
    for m in masses:
        mind += 1
        mcol = mpl_color_wheel[mind]
        mass_bin = (mass > m) & (mass <= m + mgap)
        sfr_sd_binned = sfr_sd[mass_bin]
        ne_binned = ne[mass_bin]

        sfrsd_min = -2.5
        sfrsd_max = 0

        # Calculate 25/50/75th percentiles
        ne_50 = []
        err_50 = []
        sfrsd_range = []

        b = 0.1
        for i in np.arange(sfrsd_min, sfrsd_max, b):
            try:
                ne_double_binned = ne_binned[generate_combined_mask(sfr_sd_binned >= i, sfr_sd_binned < i + b)]
                if len(ne_double_binned) == 0:
                    pass
                else:
                    p50 = np.median(ne_double_binned)
                    e50 = np.std(ne_double_binned) / np.sqrt(len(np.array(ne_double_binned)))
                    #print(p50, e50)
                    if e50 == 0:
                        e50 = 1
                    ne_50.append(p50)
                    err_50.append(e50)
                    sfrsd_range.append(i + b * 0.5)
            except IndexError:
                pass

        ne_50 = np.array(ne_50)
        err_50 = np.array(err_50)
        sfrsd_range = np.array(sfrsd_range)
        # Use inverse of variance as weights (i.e., 1/σ²)
        weights = 1 / err_50 ** 2

        try:
            # Weighted fit
            coeffs = np.polyfit(sfrsd_range, ne_50, deg=2, w=weights)

            # Evaluate the polynomial
            p = np.poly1d(coeffs)

            # Scatter with error bars
            plt.errorbar(sfrsd_range, ne_50, color=mcol, ecolor=mcol, yerr=err_50, fmt='o', capsize=5)

            # Plot the fit line
            x_fit = np.linspace(min(sfrsd_range), max(sfrsd_range), 500)
            plt.plot(x_fit, p(x_fit), color=mcol, label=fr'${m} < m \leq {m + mgap}$')
        except TypeError:
            pass

    plt.xlabel('$\log{\Sigma_{SFR} / M_\odot / yr / kpc}$', fontsize=16)
    plt.ylabel('$\log{n_e / cm^3}$', fontsize=16)
    plt.title('2nd order polynomial fit to $\Sigma_{SFR}$ vs $n_e$ binned by mass' + f' ({sum(sample_mask)} galaxies)', fontsize=16)
    plt.ylim(1.5, 3)
    plt.legend()
    plt.show()



def generate_all_plots():
    global SNR_LIM
    SNR_LIM = 5

    #print(sum(BGS_SNR_MASK))
    #print(sum((LO_Z_MASK)))
    #print(sum((HI_Z_MASK)))

    """
    # Plot the 1D distribution of electron density
    plot_ne_distribution()
    plot_ne_distribution(sample_mask=LO_Z_MASK)
    plot_ne_distribution(sample_mask=HI_Z_MASK)
    # Plot 1D distribution of star formation rate surface density
    plot_sfrsd_distribution()
    plot_sfrsd_distribution(sample_mask=LO_Z_MASK)
    plot_sfrsd_distribution(sample_mask=HI_Z_MASK)
    # Plot electron density, mass, sfr vs redshift
    plot_redshift_vs_ne_mass_sfr()
    plot_redshift_vs_ne_mass_sfr(sample_mask=LO_Z_MASK)
    plot_redshift_vs_ne_mass_sfr(sample_mask=HI_Z_MASK)
    # Plot mass, sfr, sfrsd vs ne with percentile trendlines
    plot_mass_sfr_sfrsd_vs_ne()
    plot_mass_sfr_sfrsd_vs_ne(sample_mask=LO_Z_MASK)
    plot_mass_sfr_sfrsd_vs_ne(sample_mask=HI_Z_MASK)
    # Plot sfr vs mass with ne color bins, plus count per bin and IQR
    plot_sfr_vs_mass_vs_ne()
    plot_sfr_vs_mass_vs_ne(sample_mask=LO_Z_MASK)
    plot_sfr_vs_mass_vs_ne(sample_mask=HI_Z_MASK)
    # Plot sfr surface density vs mass with ne color bins, plus count per bin and IQR
    plot_sfrsd_vs_mass_vs_ne()
    plot_sfrsd_vs_mass_vs_ne(sample_mask=LO_Z_MASK)
    plot_sfrsd_vs_mass_vs_ne(sample_mask=HI_Z_MASK)
    # Plot electron density histograms binned by mass, sfr, and deltasfr
    plot_ne_histogram_binned()
    plot_ne_histogram_binned(sample_mask=LO_Z_MASK)
    plot_ne_histogram_binned(sample_mask=HI_Z_MASK)
    # Plot BPT diagram 2d image with median ne colored in each bin
    plot_bpt_ne_color()
    plot_bpt_ne_color(sample_mask=LO_Z_MASK)
    plot_bpt_ne_color(sample_mask=HI_Z_MASK)
    # Plot extinction A(Ha) vs electron density
    plot_extinction_vs_ne()
    plot_extinction_vs_ne(sample_mask=LO_Z_MASK)
    plot_extinction_vs_ne(sample_mask=HI_Z_MASK)
    """
    # Plot fits to ne vs sfrsd in mass bins
    plot_ne_vs_sfrsd_binned()
    plot_ne_vs_sfrsd_binned(sample_mask=LO_Z_MASK)
    plot_ne_vs_sfrsd_binned(sample_mask=HI_Z_MASK)



def main():
    generate_all_plots()

if __name__ == '__main__':
    main()