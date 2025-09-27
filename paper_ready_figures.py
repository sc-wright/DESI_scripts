import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('TkAgg')
from matplotlib.gridspec import GridSpec

plt.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
# Create custom colormaps for plots
pink_blue_2val_cmap = LinearSegmentedColormap.from_list('pink_blue_cmap', ['#55CDFC', '#FFFFFF', '#F7A8B8'])

import numpy as np

#from numpy.linalg import lstsq

from scipy.stats import binned_statistic_2d, kstest, ks_2samp
from scipy.optimize import curve_fit

from import_custom_catalog import CC
from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from calculation_scripts import sfr_ms, distance_from_ms, calc_color
from sample_masks import (BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK,
                          BGS_SFR_MASK, BGS_MASS_MASK,
                          BGS_SNR_MASK, LO_Z_MASK, HI_Z_MASK,
                          Z50, Z90, M50, M90, SFR50, SFR90, bgs_sii_ne_snr_cut)
from sample_masks import bgs_ne_snr_cut, bgs_oii_ne_snr_cut, bgs_oii_ne_snr_cut, get_galaxy_type_mask


def histogram_plots():
    """
    This makes the 6x2 histogram plot for the paper
    :return:
    """
    redshift = CC.catalog['Z'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    mstar = CC.catalog['MSTAR_CIGALE'][BGS_MASK]



    # Make figure with 6 panels (e.g., 3 rows × 2 cols)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)


def plot_ne_distribution(sample_mask=BGS_SNR_MASK):
    """
    This plots a histogram of the ne distribution for the given sample mask
    :param snr_mask: default mask is the full snr>5 sample at all redshifts
    :return: none
    """

    ne, _ = bgs_ne_snr_cut()
    ne_oii, _ = bgs_oii_ne_snr_cut()
    ne_sii, _ = bgs_sii_ne_snr_cut()

    sample = 0
    sample_str = "custom sample"
    if sample_mask is BGS_SNR_MASK:
        sample = 1
        sample_str = "all galaxies"
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_str = "low-z sample"
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_str = "all-z sample"

    fs = 18

    # Statistics:
    #print("Median:", np.median(ne[sample_mask]))
    #print("Mean:", np.average(ne[sample_mask]))
    #print("Stdev:", np.std(ne[sample_mask]))

    plt.hist(ne_oii[sample_mask], bins=50)
    plt.xlim(0, 3.5)
    if sample == 2:
        tit = f"Electron density distribution (low-z, {len(ne[sample_mask])} galaxies)"
    elif sample == 3:
        tit = f"Electron density distribution (all-z, {len(ne[sample_mask])} galaxies)"
    else:
        tit = f"Electron density distribution ({len(ne[sample_mask])} galaxies)"
    plt.title(tit)
    plt.xlabel(r"$\log{n_e([OII])/cm^{-3}}$", fontsize=fs)
    plt.show()

    plt.hist(ne_sii[sample_mask], bins=50)
    plt.xlim(0, 3.5)
    if sample == 2:
        tit = f"Electron density distribution (low-z, {len(ne[sample_mask])} galaxies)"
    elif sample == 3:
        tit = f"Electron density distribution (all-z, {len(ne[sample_mask])} galaxies)"
    else:
        tit = f"Electron density distribution ({len(ne[sample_mask])} galaxies)"
    plt.title(tit)
    plt.xlabel(r"$\log{n_e([SII])/cm^{-3}}$", fontsize=fs)
    plt.show()

    # ne(OII) vs ne(SII)
    plt.hist2d(ne_oii[sample_mask], ne_sii[sample_mask], bins=80, norm=mpl.colors.LogNorm())
    plt.title(f"$n_e$ from different ions, {sample_str}")
    plt.xlim(0.5, 3.4)
    plt.ylim(0.5, 3.4)
    plt.xlabel(r"$\log{n_e([OII])/cm^{-3}}$", fontsize=fs)
    plt.ylabel(r"$\log{n_e([SII])/cm^{-3}}$", fontsize=fs)
    plt.show()

    ne_oii_sii_ks = ks_2samp(np.array(ne_oii[sample_mask]), np.array(ne_sii[sample_mask]))

    fig, ax = plt.subplots()

    counts, edges = np.histogram(ne_oii[sample_mask], bins=100)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label= "$n_e([OII])$")
    counts, edges = np.histogram(ne_sii[sample_mask], bins=100)
    cdf = np.cumsum(counts) / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centers, cdf, marker='o', mfc='none', label= "$n_e([SII])$")
    plt.xlabel(r"$\log{n_e/cm^{-3}}$", fontsize=fs)

    ks_string = f"K-S test: p = {ne_oii_sii_ks.pvalue:.3e}"
    plt.text(0.02, 0.98, ks_string, ha='left', va='top', transform=ax.transAxes, fontsize=fs - 2)
    #plt.legend(loc='lower right')
    plt.savefig('paper_figures/ne_ks.png', dpi=PLOT_DPI)
    plt.show()


def plot_redshift_vs_mass_sfr():
    """
    Plots mass and sfr vs redshift with completeness limits for each sample.
    Not built to work with custom masks
    :return: none
    """

    mass_bgs = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr_bgs = CC.catalog['SFR_HALPHA'][BGS_MASK]
    redshift_bgs = CC.catalog['Z'][BGS_MASK]

    mass = mass_bgs[BGS_SNR_MASK]
    sfr = sfr_bgs[BGS_SNR_MASK]
    redshift = redshift_bgs[BGS_SNR_MASK]

    fs = 16

    print("lo-z median:", np.median(redshift_bgs[LO_Z_MASK]))
    print("hi-z mean:", np.average(redshift_bgs[LO_Z_MASK]))
    print("hi-z median:", np.median(redshift_bgs[HI_Z_MASK]))
    print("hi-z mean:", np.average(redshift_bgs[HI_Z_MASK]))

    fig, ax = plt.subplots()
    plt.hist2d(redshift, mass, bins=60, norm=mpl.colors.LogNorm())
    plt.vlines(Z50, M50, 13, color='b', label=f'low-z completeness limits ({sum(LO_Z_MASK)} galaxies)')
    plt.hlines(M50, 0, Z50, color='b')
    plt.vlines(Z90, M90, 13, color='r', label=f'all-z completeness limits ({sum(HI_Z_MASK)} galaxies)')
    plt.hlines(M90, 0, Z90, color='r')
    #plt.title(f'Stellar mass vs redshift ({sum(BGS_SNR_MASK)} galaxies)')
    plt.xlabel("z", fontsize=fs)
    plt.ylabel(r'$\log{M_\star/M_\odot}$', fontsize=fs)
    plt.xlim(0, 0.4)
    plt.ylim(7, 11.5)
    plt.colorbar(label="count")
    plt.legend(loc='lower right')
    ax.text(0.01, 0.98, f'{sum(BGS_SNR_MASK)} galaxies',
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes, fontsize=fs-4)
    plt.savefig(f'paper_figures/paper_mass_redshift.png', dpi=PLOT_DPI)
    plt.show()

    fig, ax = plt.subplots()
    plt.hist2d(redshift, sfr, bins=60, norm=mpl.colors.LogNorm())
    plt.vlines(Z50, SFR50, 10, color='b', label=f'low-z completeness limits ({sum(LO_Z_MASK)} galaxies)')
    plt.hlines(SFR50, 0, Z50, color='b', label='_nolegend_')
    plt.vlines(Z90, SFR90, 10, color='r', label=f'all-z completeness limits ({sum(HI_Z_MASK)} galaxies)')
    plt.hlines(SFR90, 0, Z90, color='r', label='_nolegend_')
    #plt.title(f'SFR vs redshift ({sum(BGS_SNR_MASK)} galaxies)')
    plt.xlabel("z", fontsize=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', fontsize=fs)
    plt.xlim(0, 0.4)
    plt.ylim(-2, 2)
    plt.colorbar(label="count")
    plt.legend(loc='lower right')
    ax.text(0.01, 0.98, f'{sum(BGS_SNR_MASK)} galaxies',
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes, fontsize=fs-4)
    plt.savefig(f'paper_figures/paper_sfr_redshift.png', dpi=PLOT_DPI)
    plt.show()


def plot_sfr_ms(sample_mask=BGS_SNR_MASK, plot=True):

    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    mstar = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    clr = 'k'
    mlim = 0
    redshift_sample_mask = BGS_SNR_MASK
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50)
        mlim = M50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90)
        mlim = M90
        clr = 'r'

    ms_sample_mask = generate_combined_mask(redshift_sample_mask, mstar >= mlim)

    fs = 18

    #o1, o2, c = np.polyfit(mstar[ms_sample_mask], sfr[ms_sample_mask], 2)
    #o1, c = np.polyfit(mstar[ms_sample_mask], sfr[ms_sample_mask], 1)

    coeffs = np.polyfit(mstar[ms_sample_mask], sfr[ms_sample_mask], deg=2)

    # Evaluate the polynomial
    p = np.poly1d(coeffs)

    if plot:
        # x-axis arrays for our fit (complete [1] and incomplete[2] regions)
        x1 = np.linspace(mlim,20,100)
        x2 = np.linspace(0, mlim, 100)
        # full x-axis array
        xt = np.linspace(0, 20, 200)
        # x-axis for Whitaker+14, broken power law
        cmass_whitaker = 10.2
        xt_whit_lo = np.linspace(0, cmass_whitaker, 100)
        xt_whit_hi = np.linspace(cmass_whitaker, 20, 100)

        mstar_wht = np.array([9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 10.9, 11.1])
        loga = np.array([-9.54, -9.50, -9.54, -9.58, -9.69, -9.93, -10.11, -10.28, -10.53, -10.65])
        b = np.array([1.95, 1.86, 1.90, 1.98, 2.16, 2.63, 2.88, 3.03, 3.37, 3.45])

        yt_whitaker_zcorr_loz = loga * mstar_wht * np.log10(1 + 0.141) ** b
        yt_whitaker_zcorr_hiz = loga * mstar_wht * np.log10(1 + 0.237) ** b
        print(yt_whitaker_zcorr_hiz)

        # Our fit
        y1 = p(x1)
        y2 = p(x2)

        # Galaxy ages calculated with Ned Wright's Cosmology calculator using flat cosmology and mean redshifts for each sample:
        # hi-z mean: 0.141
        # hi-z mean: 0.237
        age_loz = 11.924
        age_hiz = 10.885
        yt_speagle_loz = (0.84 - 0.026*age_loz) * xt - (6.51 - 0.11*age_loz)
        yt_speagle_hiz = (0.84 - 0.026*age_hiz) * xt - (6.51 - 0.11*age_hiz)

        # Schreiber+15
        r_loz = np.log10(1 + 0.141)
        r_hiz = np.log10(1 + 0.237)
        xt_proc_loz = (xt - 9) - 0.36 - 2.5*r_loz
        xt_proc_hiz = (xt - 9) - 0.36 - 2.5*r_hiz
        yt_schreiber_loz = (xt - 9) - 0.5 + 1.5*r_loz - 0.3*(np.where(xt_proc_loz<0, 0, xt_proc_loz)**2)
        yt_schreiber_hiz = (xt - 9) - 0.5 + 1.5*r_hiz - 0.3*(np.where(xt_proc_hiz<0, 0, xt_proc_hiz)**2)

        # This lets us calculate the offset between Whitaker and Schreiber
        r_whit = np.log10(1 + 0.75)
        xt_proc_loz_sv = (10 - 9) - 0.36 - 2.5 * r_loz
        xt_proc_hiz_sv = (10 - 9) - 0.36 - 2.5 * r_hiz
        xt_proc_whit_sv = (10 - 9) - 0.36 - 2.5 * r_whit
        yt_schreiber_loz_sv = (10 - 9) - 0.5 + 1.5 * r_loz - 0.3 * (max(0, xt_proc_loz_sv) ** 2)
        yt_schreiber_hiz_sv = (10 - 9) - 0.5 + 1.5 * r_hiz - 0.3 * (max(0, xt_proc_hiz_sv) ** 2)
        yt_schreiber_whit_sv = (10 - 9) - 0.5 + 1.5 * r_hiz - 0.3 * (max(0, xt_proc_whit_sv) ** 2)

        loz_offset = yt_schreiber_whit_sv - yt_schreiber_loz_sv
        hiz_offset = yt_schreiber_whit_sv - yt_schreiber_hiz_sv

        # Whitaker has both a 2nd order polynomial fit and a broken power law
        yt_whitaker_p2_loz = -27.40 + 5.02 * xt + -0.22 * xt ** 2 - loz_offset
        yt_whitaker_p2_hiz = -27.40 + 5.02 * xt + -0.22 * xt ** 2 - hiz_offset
        # This is the 2nd order polynomial, we aren't including it
        # yt_whitaker_bpl_lo = 0.94 * (xt_whit_lo - 10.2) + 1.11
        # yt_whitaker_bpl_hi = 0.14 * (xt_whit_hi - 10.2) + 1.11

        fig, ax = plt.subplots(figsize=(6, 5))
        plt.hist2d(mstar[redshift_sample_mask], sfr[redshift_sample_mask], bins=(80,40), norm=mpl.colors.LogNorm())
        plt.plot(x1, y1, color='k', label='our polynomial fit')
        plt.plot(x2, y2, color='k', linestyle='--', label='_nolegend_')
        #plt.plot(xt, yt_speagle, label='Speagle+14', color='tab:blue')
        #plt.plot(xt_whit_lo, yt_whitaker_bpl_lo, color='tab:purple')
        #plt.plot(xt_whit_hi, yt_whitaker_bpl_hi, color='tab:purple')
        #plt.plot(xt, yt_schreiber, color='tab:green', label='Schreiber+15')
        #plt.plot(x, y2, color='k', linestyle='--', label='sSFR cut')
        plt.xlim(8, 11.5)
        plt.ylim(-1.5, 2)
        plt.colorbar(label='count')
        if sample == 2:
            #plt.title("SFR main sequence (low-z sample)")
            ax.text(0.01, 0.98, f'low-z sample',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes, fontsize=fs-4)
            #plt.plot(xt, yt_speagle_loz, label='Speagle+14', color='tab:blue')
            plt.plot(xt, yt_schreiber_loz, color='tab:green', label='Schreiber+15')
            #plt.plot(mstar_wht, yt_whitaker_zcorr_loz, color='tab:purple', label='Whitaker+14')
            plt.plot(xt, yt_whitaker_p2_loz, label=r'Whitaker+14', color='tab:purple')

        if sample == 3:
            #plt.title("SFR main sequence (all-z sample)")
            ax.text(0.01, 0.98, f'all-z sample',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes, fontsize=fs-4)
            #plt.plot(xt, yt_speagle_hiz, label='Speagle+14', color='tab:blue')
            plt.plot(xt, yt_schreiber_hiz, color='tab:green', label='Schreiber+15')
            #plt.plot(mstar_wht, yt_whitaker_zcorr_hiz, color='tab:purple', label='Whitaker+14')
            plt.plot(xt, yt_whitaker_p2_hiz, label=r'Whitaker+14', color='tab:purple')
        if sample == 2:
            plt.vlines(M50, SFR50, 13, color='b', label=f'low-z completeness limits')# ({sum(LO_Z_MASK)} galaxies)')
            plt.hlines(SFR50, 100, M50, color='b')
        elif sample == 3:
            plt.vlines(M90, SFR90, 13, color='r', label=f'all-z completeness limits')# ({sum(HI_Z_MASK)} galaxies)')
            plt.hlines(SFR90, 100, M90, color='r')
        plt.xlabel(r'$\log{M_\star/M_\odot}$', fontsize=fs)
        plt.ylabel(r'$\log{SFR/M_\odot/yr}$', fontsize=fs)
        plt.legend(loc='lower right')
        plt.savefig(f'paper_figures/paper_sfr_ms_sample_{sample}.png', dpi=PLOT_DPI)
        plt.show()
        """
        plt.hist2d(mstar, specific_sfr, bins=(200,70), norm=mpl.colors.LogNorm())
        #plt.plot(x, np.ones(len(x))*sSFR_cut, color='k', linestyle='--', label='sSFR cut')
        plt.xlim(8, 11.5)
        plt.colorbar(label='count')
        #plt.ylim(-.35, .35)
        plt.xlabel(r'$\log{M_\star/M_\odot}$')
        plt.ylabel(r'$\log{SFR / M_\star}$')
        #plt.legend(loc='upper left')
        plt.show()
        """

    return p

def plot_redshift_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Plots electron density vs redshift
    :return: none
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    redshift = CC.catalog['Z'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    zmax=0.4
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = np.array((BGS_SNR_MASK) & (sfr >= SFR50) & (mass >= M50))
        zmax = Z50
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = np.array((BGS_SNR_MASK) & (sfr >= SFR90) & (mass >= M90))
        zmax = Z90

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    ne = ne[sample_mask]
    redshift = redshift[sample_mask]

    fs = 16

    p, V = np.polyfit(redshift[redshift <= zmax], ne[redshift <= zmax], 1, cov=True)
    m = p[0]
    dm = np.sqrt(V[0][0])
    b = p[1]
    fit_x = np.linspace(0,.5,3)
    fit_y = m * fit_x + b

    fig, ax = plt.subplots()
    plt.hist2d(redshift, ne, bins=60, norm=mpl.colors.LogNorm())
    plt.plot(fit_x, fit_y, color='k', label='linear fit (slope = {:.3f}'.format(m) + ' +/- {:.3f})'.format(dm))
    if sample == 2:
        plt.vlines(Z50, 0, 3.5, color='b', label="Completeness upper limit")
        #plt.title(f'Electron density vs redshift (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(Z90, 0, 3.5, color='r', label="Completeness upper limit")
        #plt.title(f'Electron density vs redshift (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(f'Electron density vs redshift ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlabel("z", fontsize=fs)
    plt.ylabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.legend(loc='lower right')
    plt.xlim(0, 0.4)
    plt.ylim(0, 3.5)
    plt.colorbar(label="count")
    plt.savefig(f'paper_figures/paper_ne_redshift_sample_{sample}.png', dpi=PLOT_DPI)
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
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50)
        mlim = M50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90)
        mlim = M90
        clr = 'r'

    mass_sample = mass[sample_mask]
    sfr_sample = sfr[sample_mask]
    ne_sample = ne[sample_mask]

    fs = 20

    o1, o2, c = plot_sfr_ms(sample_mask=sample_mask, plot=False)

    #x_ms = np.linspace(0,20,100)
    #y_ms = o1 * x_ms**2 + o2 * x_ms + c

    x1 = np.linspace(mlim, 20, 100)
    x2 = np.linspace(0, mlim, 100)
    y1 = o1 * x1 ** 2 + o2 * x1 + c
    y2 = o1 * x2 ** 2 + o2 * x2 + c

    # Define the number of bins
    x_bins = 80
    y_bins = 40

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfr[redshift_sample_mask], ne[redshift_sample_mask], statistic='median', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.set_facecolor('gray')
    plt.pcolormesh(X, Y, stat.T, cmap=pink_blue_2val_cmap, shading='auto', vmin=1.824, vmax=2.224)
    #plt.plot(x1, y1, color=clr, label='polynomial fit')
    #plt.plot(x2, y2, color=clr, linestyle='--', label='_nolegend_')
    # If using samples 2 or 3, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 20, color='b')
        plt.vlines(M50, SFR50, 20, color='b')
        #plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.hlines(SFR90, M90, 20, color='r')
        plt.vlines(M90, SFR90, 20, color='r')
        #plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(rf"SFR vs $M_\star$ vs $n_e$" + f" ({sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-1.5, 2)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', size=fs)
    plt.savefig(f'paper_figures/sfr_vs_mstar_vs_ne_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfr[redshift_sample_mask], ne[redshift_sample_mask], statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    plt.plot(x1, y1, color=clr, label='polynomial fit')
    plt.plot(x2, y2, color=clr, linestyle='--', label='_nolegend_')
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 20, color='b')
        plt.vlines(M50, SFR50, 20, color='b')
        #plt.title(r"SFR vs $M_\star$" + f" vs count per bin (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.hlines(SFR90, M90, 20, color='r')
        plt.vlines(M90, SFR90, 20, color='r')
        #plt.title(r"SFR vs $M_\star$" + f" vs count per bin (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(rf"SFR vs $M_\star$" + f" vs count per bin ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-1.5, 2)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', size=fs)
    plt.savefig(f'paper_figures/sfr_vs_mstar_counts_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfr[redshift_sample_mask], ne[redshift_sample_mask], statistic=iqr, bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)#, norm=mpl.colors.LogNorm())
    plt.plot(x1, y1, color=clr, label='polynomial fit')
    plt.plot(x2, y2, color=clr, linestyle='--', label='_nolegend_')
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.hlines(SFR50, M50, 20, color='b')
        plt.vlines(M50, SFR50, 20, color='b')
        #plt.title(rf"SFR vs $M_\star$ vs inter-quartile range (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.hlines(SFR90, M90, 20, color='r')
        plt.vlines(M90, SFR90, 20, color='r')
        #plt.title(rf"SFR vs $M_\star$ vs inter-quartile range (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(rf"SFR vs $M_\star$ vs inter-quartile range ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-1.5, 2)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{SFR/M_\odot/yr}$', size=fs)
    plt.savefig(f'paper_figures/sfr_vs_mstar_vs_iqr_{sample}.png', dpi=PLOT_DPI)
    plt.show()


def plot_sfrsd_vs_mass_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Generates a plot of sfr vs mass with each bin color-coded by median ne
    :param sample_mask: Optional extra mask that is placed after snr cuts
    :return:
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfrsd = CC.catalog['SFR_SD'][BGS_MASK]  # bgs length
    ne, _ = bgs_ne_snr_cut()
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50)
        mlim = M50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90)
        mlim = M90
        clr = 'r'

    mass_sample = mass[sample_mask]
    sfrsd_sample = sfrsd[sample_mask]
    ne_sample = ne[sample_mask]

    # Define the number of bins
    x_bins = 80
    y_bins = 40

    # font size for labels
    fs = 20

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='median', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.set_facecolor('gray')
    plt.pcolormesh(X, Y, stat.T, cmap=pink_blue_2val_cmap, shading='auto', vmin=1.824, vmax=2.224)
    # If using samples 2 or 3, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$ ($\log{}$cm$^{-3}$)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', size=fs)
    plt.savefig(f'paper_figures/sfrsd_vs_mstar_vs_ne_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    plt.savefig(f'paper_figures/sfrsd_vs_mstar_counts_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic=iqr, bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)#, norm=mpl.colors.LogNorm())
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    else:
        #plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    plt.xlim(8, 11.5)
    plt.ylim(-2.5, 0)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    plt.savefig(f'paper_figures/sfrsd_vs_mstar_vs_iqr_{sample}.png', dpi=PLOT_DPI)
    plt.show()



def plot_mass_sfr_sfrsd_vs_ne(sample_mask=BGS_SNR_MASK):
    """
    Plot mass, sfr, sfrsd vs ne with percentile trendlines
    :return: none
    """
    mass_bgs = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr_bgs = CC.catalog['SFR_HALPHA'][BGS_MASK]
    sfr_sd_bgs = CC.catalog['SFR_SD'][BGS_MASK]
    z_bgs = CC.catalog['Z'][BGS_MASK]
    ne_bgs, _ = bgs_ne_snr_cut()  # these are both bgs length

    sample = 0
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK & (z_bgs < Z50)
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK & (z_bgs < Z90)

    mass = mass_bgs[sample_mask]
    sfr = sfr_bgs[sample_mask]
    sfr_sd = sfr_sd_bgs[sample_mask]
    z = z_bgs[sample_mask]
    ne = ne_bgs[sample_mask]

    fs = 20

    # set percentile line color
    colr = 'dodgerblue'
    colrmap = 'inferno'

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

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(mass, ne, bins=80, cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(M50, 0, 3.5, color='b')
        #plt.title(f'$n_e$ vs $M_\star$ (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs-4)
    elif sample == 3:
        plt.vlines(M90, 0, 3.5, color='r')
        #plt.title(f'$n_e$ vs $M_\star$ (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(f'$n_e$ vs $M_\star$ ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(mrange, ne_25, color='white', linewidth=3.5)
    plt.plot(mrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(mrange, ne_50, color='white', linewidth=3.5)
    plt.plot(mrange, ne_50, color=colr)
    plt.plot(mrange, ne_75, color='white',  linewidth=3.5)
    plt.plot(mrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{M_\star/M_\odot}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(massmin, massmax)
    plt.ylim(1, 3)
    plt.savefig(f'paper_figures/paper_ne_vs_mass_{sample}.png', dpi=PLOT_DPI)
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

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(sfr, ne, bins=80, cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        plt.vlines(SFR50, 0, 3.5, color='b')
        #plt.title(f'$n_e$ vs SFR (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(SFR90, 0, 3.5, color='r')
        #plt.title(f'$n_e$ vs SFR (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(f'$n_e$ vs SFR ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(sfrrange, ne_25, color='white', linewidth=3.5)
    plt.plot(sfrrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrrange, ne_50, color='white', linewidth=3.5)
    plt.plot(sfrrange, ne_50, color=colr)
    plt.plot(sfrrange, ne_75, color='white', linewidth=3.5)
    plt.plot(sfrrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{SFR/M_\odot/yr}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(sfrmin, sfrmax)
    plt.ylim(1, 3)
    plt.savefig(f'paper_figures/paper_ne_vs_sfr_{sample}.png', dpi=PLOT_DPI)
    plt.show()


    # Plot ne vs sfr_sd

    # Change mask to full mass and sfr cuts for this plot so we are only plotting the complete region
    if sample == 2:
        sample_mask = LO_Z_MASK
    elif sample == 3:
        sample_mask = HI_Z_MASK

    sfr_sd = sfr_sd_bgs[sample_mask]
    ne = ne_bgs[sample_mask]

    sfrsdmin = -2
    sfrsdmax = -.25

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrsdrange = []

    b = b/2

    for i in np.arange(sfrsdmin, sfrsdmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(sfr_sd >= i, sfr_sd < i + b)], (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            sfrsdrange.append(i + b * 0.5)
        except IndexError:
            pass

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(sfr_sd, ne, bins=80, cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        #plt.vlines(SFR50, 0, 3.5, color='b')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        #plt.vlines(SFR90, 0, 3.5, color='r')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(sfrsdrange, ne_25, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrsdrange, ne_50, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_50, color=colr)
    plt.plot(sfrsdrange, ne_75, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(sfrsdmin, sfrsdmax)
    plt.ylim(1, 3)
    plt.savefig(f'paper_figures/paper_ne_vs_sfrsd_{sample}.png', dpi=PLOT_DPI)
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
    cmap = pink_blue_2val_cmap
    #norm = Normalize(vmin=1.5, vmax=2.5)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('gray')
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap=cmap, shading='auto', vmin=1.824, vmax=2.224)
    plt.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='k')
    plt.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r')
    plt.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b')
    plt.text(-1.1, -0.4, f"H II\n{hii_ne_median:.2f}", fontweight='bold')
    plt.text(-.15, -0.75, f"Composite\n{composite_ne_median:.2f}", fontweight='bold')
    plt.text(-1.0, 1.1, f"AGN\n{agn_ne_median:.2f}", fontweight='bold')
    plt.text(0.15, -0.25, f"Shocks\n{shock_ne_median:.2f}", fontweight='bold')
    #plt.text(0.005, 1.005, f'total: {sum(bpt_mask)}, snr $>$ {snr_lim}',
    #      horizontalalignment='left',
    #      verticalalignment='bottom',
    #      transform=ax.transAxes)
    plt.xlim(-1.25, 0.4)
    plt.ylim(-1, 1.5)
    sam_title = ""
    if sample == 2:
        ax.text(0.01, 0.98, f'low-z, {sum(bpt_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
        sam_title = "hi_z"
    #    plt.title(fr'BPT diagram color coded by $n_e$ (low-z, {sum(bpt_mask)} galaxies)')#, fontsize=16)
    elif sample == 3:
        ax.text(0.01, 0.98, f'all-z, {sum(bpt_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
        sam_title = "all-z"
    #    plt.title(f'BPT diagram color coded by $n_e$ (all-z, {sum(bpt_mask)} galaxies)')#, fontsize=16)
    #else:
    #    plt.title(f'BPT diagram color coded by $n_e$ ({sum(bpt_mask)} galaxies)')#, fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e/cm^3}$', fontsize=fs)
    #cbar = plt.colorbar(sm, ax=ax, label=r"$n_e$ (cm$^{-3}$)")
    plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$', fontsize=fs)
    plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$', fontsize=fs)
    plt.savefig(f'paper_figures/paper_bpt_ne_color_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    # Perform K-S tests and make cumulative distributions

    out = ks_2samp(np.array(ne[hii_object_mask]), np.array(ne[agn_object_mask]))
    print(out.pvalue)

    plt.hist(ne[hii_object_mask], density=True, cumulative=True, bins=50)
    plt.xlabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.title("BPT HII galaxies cumulative distribution " + sam_title)
    plt.xlim(1, 3)
    plt.show()

    plt.hist(ne[agn_object_mask], density=True, cumulative=True, bins=50)
    plt.xlabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.title("BPT AGN galaxies cumulative distribution " + sam_title)
    plt.xlim(1, 3)
    plt.show()


def bpt_ks_tests():
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

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    # all input masks are BGS length
    bpt_mask = generate_combined_mask(nii_snr > snr_lim, oiii_snr > snr_lim)

    loz_mask = generate_combined_mask(LO_Z_MASK, bpt_mask)
    hiz_mask = generate_combined_mask(HI_Z_MASK, bpt_mask)

    nh_lo = np.log10(nii[loz_mask] / ha[loz_mask])  # x-axis
    oh_lo = np.log10(oiii[loz_mask] / hb[loz_mask]) # y-axis
    ne_lo = ne[loz_mask]

    nh_hi = np.log10(nii[hiz_mask] / ha[hiz_mask])  # x-axis
    oh_hi = np.log10(oiii[hiz_mask] / hb[hiz_mask]) # y-axis
    ne_hi = ne[hiz_mask]

    hii_boundary = lambda x: 0.61/(x - 0.05) + 1.3          # black dashed
    agn_boundary = lambda x: 0.61 / (x - 0.47) + 1.19       # red dotted
    shock_boundary = lambda x: 2.144507*x + 0.465028        # blue dotdash

    hii_lo_object_mask         = (oh_lo < agn_boundary(nh_lo)) & (oh_lo < hii_boundary(nh_lo))         # below both red and black lines
    agn_lo_object_mask         = (oh_lo > agn_boundary(nh_lo)) & (oh_lo > shock_boundary(nh_lo))       # above both red and blue
    composite_lo_object_mask   = (oh_lo > hii_boundary(nh_lo)) & (oh_lo < agn_boundary(nh_lo))         # above black and below red
    shock_lo_object_mask       = (oh_lo > agn_boundary(nh_lo)) & (oh_lo < shock_boundary(nh_lo))       # above red and below blue

    hii_hi_object_mask         = (oh_hi < agn_boundary(nh_hi)) & (oh_hi < hii_boundary(nh_hi))         # below both red and black lines
    agn_hi_object_mask         = (oh_hi > agn_boundary(nh_hi)) & (oh_hi > shock_boundary(nh_hi))       # above both red and blue
    composite_hi_object_mask   = (oh_hi > hii_boundary(nh_hi)) & (oh_hi < agn_boundary(nh_hi))         # above black and below red
    shock_hi_object_mask       = (oh_hi > agn_boundary(nh_hi)) & (oh_hi < shock_boundary(nh_hi))       # above red and below blue

    # K-S tests

    print("Comparing low-z sub-samples...")
    hii_agn_lo = ks_2samp(np.array(ne_lo[hii_lo_object_mask]), np.array(ne_lo[agn_lo_object_mask]))
    print("AGN vs. HII, lo-z:", hii_agn_lo.pvalue)
    hii_com_lo = ks_2samp(np.array(ne_lo[hii_lo_object_mask]), np.array(ne_lo[composite_lo_object_mask]))
    print("HII vs. COM, lo-z:", hii_com_lo.pvalue)
    com_agn_lo = ks_2samp(np.array(ne_lo[composite_lo_object_mask]), np.array(ne_lo[agn_lo_object_mask]))
    print("COM vs. AGN, lo-z:", com_agn_lo.pvalue)

    print("Comparing high-z sub-samples...")
    hii_agn_hi = ks_2samp(np.array(ne_hi[hii_hi_object_mask]), np.array(ne_hi[agn_hi_object_mask]))
    print("AGN vs. HII, hi-z:", hii_agn_hi.pvalue)
    hii_com_hi = ks_2samp(np.array(ne_hi[hii_hi_object_mask]), np.array(ne_hi[composite_hi_object_mask]))
    print("HII vs. COM, hi-z:", hii_com_hi.pvalue)
    com_agn_hi = ks_2samp(np.array(ne_hi[composite_hi_object_mask]), np.array(ne_hi[agn_hi_object_mask]))
    print("COM vs. AGN, hi-z:", com_agn_hi.pvalue)

    print("Comparing low-z vs high-z samples...")
    hii_lo_hi = ks_2samp(np.array(ne_lo[hii_lo_object_mask]), np.array(ne_hi[hii_hi_object_mask]))
    print("HII objects high vs low:", hii_lo_hi.pvalue)
    com_lo_hi = ks_2samp(np.array(ne_lo[composite_lo_object_mask]), np.array(ne_hi[composite_hi_object_mask]))
    print("COM objects high vs low:", com_lo_hi.pvalue)
    agn_lo_hi = ks_2samp(np.array(ne_lo[agn_lo_object_mask]), np.array(ne_hi[agn_hi_object_mask]))
    print("AGN objects high vs low:", agn_lo_hi.pvalue)

    fs = 16

    lo_bins = [np.array(ne_lo[hii_lo_object_mask]), np.array(ne_lo[composite_lo_object_mask]), np.array(ne_lo[agn_lo_object_mask])]
    #clrs_lo = ["#a6cee3", "#1f78b4", "#08306b"]
    clrs_lo = [plt.cm.Blues(i) for i in np.linspace(0.4, 0.9, 3)]

    hi_bins = [np.array(ne_hi[hii_hi_object_mask]), np.array(ne_hi[composite_hi_object_mask]), np.array(ne_hi[agn_hi_object_mask])]
    #clrs_hi = ["#fb9a99", "#e31a1c", "#67000d"]
    clrs_hi = [plt.cm.Reds(i) for i in np.linspace(0.4, 0.9, 3)]

    bin_name = ['HII', 'Composite', 'AGN']
    bins = np.linspace(1, 3, 40)

    fig, ax = plt.subplots()

    for ne_lo, clr_lo, ne_hi, clr_hi, lab in zip(lo_bins, clrs_lo, hi_bins, clrs_hi, bin_name):
        counts, edges = np.histogram(ne_lo, bins=bins)
        cdf = np.cumsum(counts) / np.sum(counts)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centers, cdf, marker='o', mfc='none', label=lab + ' (low)', color=clr_lo)
        counts, edges = np.histogram(ne_hi, bins=bins)
        cdf = np.cumsum(counts) / np.sum(counts)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centers, cdf, marker='o', mfc='none', label=lab + ' (all)', color=clr_hi)
    ks_string = f"K-S test\nHII: \tp = {hii_lo_hi.pvalue:.3e}\nCOM: \tp = {com_lo_hi.pvalue:.3f}\nAGN: \tp = {agn_lo_hi.pvalue:.3f}"
    plt.text(0.02, 0.98, ks_string, ha='left', va='top', transform=ax.transAxes, fontsize=fs-2)
    plt.xlabel(r'$\log({n_e}/cm^{3}$)', fontsize=fs)
    plt.legend(loc='lower right')
    plt.savefig('paper_figures/bpt_ks.png', dpi=PLOT_DPI)
    plt.show()


def plot_ne_vs_sfrsd_binned(sample_mask=BGS_SNR_MASK):
    """
    This function bins the sfrsd/mass/ne data and performs O(2) fits to each mass bin.
    It is not a single fit to all the data
    For the more rigorous and complete fit, view the sfrsd_fitting.py file
    This is kept for records and should not be used
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
    elif sample_mask is HI_Z_MASK:
        sample = 3

    mass = mass[sample_mask]
    sfr = sfr[sample_mask]
    sfr_sd = sfr_sd[sample_mask]
    ne = ne[sample_mask]

    fs = 18

    #mpl_color_wheel = ['#E40303', '#FF8C00', '#FFED00', '#008026', '#004CFF', '#732982']
    mpl_color_wheel = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    mgap = .5
    masses = np.arange(8.5, 11.5, mgap)
    xmin = 1
    xmax = 3

    fig = plt.figure(figsize=(5,5))

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

        ne_50_small = []
        sfrsd_range_small = []

        b = 0.25
        for i in np.arange(sfrsd_min, sfrsd_max, b):
            try:
                ne_double_binned = ne_binned[generate_combined_mask(sfr_sd_binned >= i, sfr_sd_binned < i + b)]
                if len(ne_double_binned) < 1:
                    pass
                elif len(ne_double_binned) < 10:
                    # We want there to be at least 10 objects in each bin.
                    p50 = np.median(ne_double_binned)
                    ne_50_small.append(p50)
                    sfrsd_range_small.append(i + b * 0.5)

                else:
                    p50 = np.median(ne_double_binned)
                    e50 = np.std(ne_double_binned) / np.sqrt(len(np.array(ne_double_binned)))

                    #print(p50, e50)
                    # If there is only one object in the bin, its error is 0
                    #if e50 == 0:
                    #    e50 = 1
                    ne_50.append(p50)
                    err_50.append(e50)
                    sfrsd_range.append(i + b * 0.5)
            except IndexError:
                pass

        ne_50 = np.array(ne_50)
        err_50 = np.array(err_50)
        sfrsd_range = np.array(sfrsd_range)
        ne_50_small = np.array(ne_50_small)
        sfrsd_range_small = np.array(sfrsd_range_small)
        # Use inverse of variance as weights (i.e., 1/σ²)
        weights = 1 / err_50 ** 2

        try:
            # Weighted fit
            coeffs = np.polyfit(sfrsd_range, ne_50, deg=2, w=weights)

            # Evaluate the polynomial
            p = np.poly1d(coeffs)

            # Scatter with error bars. Add tiny offset to x-axis just to make plot more readable
            plt.errorbar(sfrsd_range + mind/100, ne_50, color=mcol, ecolor=mcol, yerr=err_50, fmt='o', capsize=5)
            plt.scatter(sfrsd_range_small + mind/100, ne_50_small, facecolors='none', edgecolors=mcol, marker='o')

            # Plot the fit line
            x_fit = np.linspace(min(np.concatenate([sfrsd_range, sfrsd_range_small])), max(np.concatenate([sfrsd_range, sfrsd_range_small])), 500)
            plt.plot(x_fit + mind/100, p(x_fit), color=mcol, label=fr'${m} < \log m \leq {m + mgap}$ ({sum(np.array(mass_bin))})')
        except TypeError:
            pass

    plt.xlabel('$\log{\Sigma_{SFR} / M_\odot / yr / kpc}$', fontsize=fs)
    plt.ylabel('$\log{n_e / cm^3}$', fontsize=fs)
    #plt.title('2nd order polynomial fit to $\Sigma_{SFR}$ vs $n_e$ binned by mass' + f' ({sum(sample_mask)} galaxies)', fontsize=16)
    if sample == 2:
        plt.title("low-z")
    if sample == 3:
        plt.title("all-z")
    plt.ylim(1.5, 3)
    plt.legend(fontsize=fs-8)
    plt.savefig(f'paper_figures/paper_sfrsd_ne_binned_fits_{sample}.png', dpi=PLOT_DPI)
    plt.show()


def metallicity(sample_mask=BGS_SNR_MASK):
    oiii_5007_flux = np.array(CC.catalog['OIII_5007_FLUX'][BGS_MASK])
    oiii_5007_err_inv = np.array(np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK]))
    nii_6584_flux = np.array(CC.catalog['NII_6584_FLUX'][BGS_MASK])
    nii_6584_err_inv = np.array(np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK]))
    halpha_flux = np.array(CC.catalog['HALPHA_FLUX'][BGS_MASK])
    halpha_flux_err_inv = np.array(np.sqrt(CC.catalog['HALPHA_FLUX_IVAR'][BGS_MASK]))
    hbeta_flux = np.array(CC.catalog['HBETA_FLUX'][BGS_MASK])
    hbeta_flux_err_inv = np.array(np.sqrt(CC.catalog['HBETA_FLUX_IVAR'][BGS_MASK]))

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    ne, _ = bgs_ne_snr_cut()  # these are both bgs length
    redshift = CC.catalog['Z'][BGS_MASK]

    sample = 0
    tit = "custom sample"
    mlim = 0
    clr = 'k'
    if sample_mask is BGS_SNR_MASK:
        sample = 1
        tit = "all galaxies"
        #mlim = [M50, M90]
        #clr = ['b', 'r']
    elif sample_mask is LO_Z_MASK:
        sample = 2
        sample_mask = BGS_SNR_MASK & (sfr > SFR50) & (mass > M50)
        mlim = M50
        sfrlim = SFR50
        zlim = Z50
        clr = 'b'
        tit = 'low-z'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        sample_mask = BGS_SNR_MASK & (sfr > SFR90) & (mass > M90)
        mlim = M90
        sfrlim = SFR90
        zlim = Z90
        clr = 'r'
        tit = 'all-z'

    oiii_5007_snr = oiii_5007_flux * oiii_5007_err_inv
    nii_6584_snr = nii_6584_flux * nii_6584_err_inv
    halpha_snr = halpha_flux * halpha_flux_err_inv
    hbeta_snr = hbeta_flux * hbeta_flux_err_inv

    snr_lim = 3
    fs = 18

    # This is just the metallicity lines
    metallicity_mask = generate_combined_mask(oiii_5007_snr > snr_lim, nii_6584_snr > snr_lim, halpha_snr > snr_lim, hbeta_snr > snr_lim)

    # 03N2 from Pettini & Pagel 2004
    O3N2 = np.log10( (oiii_5007_flux / hbeta_flux) / (nii_6584_flux / halpha_flux) )

    # From PP04
    o3n2_metallicity = 8.73 - 0.32 * O3N2

    hii_galaxy_mask, agn_galaxy_mask, _, _ = get_galaxy_type_mask()

    # 4 is arbitrary, we are just removing the galaxies with failed mass fits
    mass_z_mask = generate_combined_mask(metallicity_mask, sfr > sfrlim, mass > 4, BGS_SNR_MASK, ~agn_galaxy_mask)

    plt.hist2d(mass[mass_z_mask], o3n2_metallicity[mass_z_mask], bins=(120, 90), norm=mpl.colors.LogNorm())
    plt.vlines(mlim, 0, 20, color=clr)
    plt.xlim(8, 11.5)
    plt.ylim(8, 9)
    plt.colorbar()
    plt.xlabel(r'$\log{M_\star/M_\odot}$')
    plt.ylabel(r'$12 + \log{O/H}$')
    plt.title(tit)
    plt.show()

    # Keep all galaxies except for AGN, make note that most composite galaxies are in the high-mass region
    full_mask = generate_combined_mask(metallicity_mask, sample_mask, ~agn_galaxy_mask)

    plt.hist2d(o3n2_metallicity[full_mask], ne[full_mask], bins=(120, 90), norm=mpl.colors.LogNorm())
    plt.xlim(8.0, 9)
    plt.ylim(1, 3)
    plt.colorbar(label='count')
    plt.xlabel(r'$12 + \log{O/H}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.title(tit)
    plt.show()


def total_sfr_sd(sample_mask = BGS_SNR_MASK):
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    # Half-light radius
    radius = CC.catalog['SHAPE_R'][BGS_MASK]
    #print(sum(radius[BGS_SNR_MASK] <= 0))

    sfrsd = sfr / (np.pi * radius ** 2)
    ne, _ = bgs_ne_snr_cut()
    z = CC.catalog['Z'][BGS_MASK]

    sample = 0
    mlim = 0
    clr = 'k'
    redshift_sample_mask = sample_mask
    if sample_mask is BGS_SNR_MASK:
        sample = 1
    elif sample_mask is LO_Z_MASK:
        sample = 2
        redshift_sample_mask = BGS_SNR_MASK & (z < Z50) & (radius > 0)
        mlim = M50
        clr = 'b'
    elif sample_mask is HI_Z_MASK:
        sample = 3
        redshift_sample_mask = BGS_SNR_MASK & (z < Z90) & (radius > 0)
        mlim = M90
        clr = 'r'

    #mass_sample = mass[sample_mask]
    #sfrsd_sample = sfrsd[sample_mask]
    #ne_sample = ne[sample_mask]

    # Define the number of bins
    x_bins = 90
    y_bins = 75

    # font size for labels
    fs = 20
    ymin = -2.5
    ymax = 3

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='median',
        bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x_edges, y_edges)
    ax.set_facecolor('gray')
    plt.pcolormesh(X, Y, stat.T, cmap=pink_blue_2val_cmap, shading='auto', vmin=1.824, vmax=2.224)
    # If using samples 2 or 3, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs $n_e$" + f" ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.xlim(8, 11.5)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$ ($\log{}$cm$^{-3}$)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', size=fs)
    #plt.savefig(f'paper_figures/sfrsd_total_vs_mstar_vs_ne_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    """
    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic='count',
        bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$" + f" vs count per bin ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.xlim(8, 11.5)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    #plt.savefig(f'paper_figures/sfrsd_total_vs_mstar_counts_{sample}.png', dpi=PLOT_DPI)
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass[redshift_sample_mask], sfrsd[redshift_sample_mask], ne[redshift_sample_mask], statistic=iqr,
        bins=[x_bins, y_bins]
    )

    # Plot the result
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.8)  # , norm=mpl.colors.LogNorm())
    # If the sample is constrained, we will mark the section with 90% completeness
    if sample == 2:
        plt.vlines(M50, -20, 20, color='b')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (low-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'low-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        plt.vlines(M90, -20, 20, color='r')
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range (all-z, {sum(redshift_sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'all-z, {sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        # plt.title(r"$\Sigma_{SFR}$ vs $M_\star$ vs inter-quartile range ({sum(sample_mask)} galaxies)")
        ax.text(0.01, 0.98, f'{sum(redshift_sample_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.xlim(8, 11.5)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{\Sigma_{SFR}/M_\odot/yr}$', size=fs)
    #plt.savefig(f'paper_figures/sfrsd_total_vs_mstar_vs_iqr_{sample}.png', dpi=PLOT_DPI)
    plt.show()
    """






    # Plot ne vs sfr_sd

    # If we include all completeness cuts, all of the SFRSD < 0 disappear and I cannot figure out why
    trendline_mask = np.array(redshift_sample_mask & (mass >= mlim))


    sfrsd = sfrsd[trendline_mask]
    mass = mass[trendline_mask]
    ne = ne[trendline_mask]

    print(min(sfrsd))

    sfrsdmin = -0.1
    sfrsdmax = 0.7

    colrmap = 'inferno'
    colr = 'dodgerblue'

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrsdrange = []

    b = 0.02

    for i in np.arange(sfrsdmin, sfrsdmax, b):
        try:
            p25, p50, p75 = np.percentile(ne[generate_combined_mask(sfrsd >= i, sfrsd < i + b)], (25, 50, 75))
            ne_25.append(p25)
            ne_50.append(p50)
            ne_75.append(p75)
            sfrsdrange.append(i + b * 0.5)
        except IndexError:
            pass

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.12)
    plt.hist2d(sfrsd, ne, bins=(500,70), cmap=colrmap, norm=mpl.colors.LogNorm())
    if sample == 2:
        #plt.vlines(SFR50, 0, 3.5, color='b')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (low-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'low-z, {sum(trendline_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    elif sample == 3:
        #plt.vlines(SFR90, 0, 3.5, color='r')
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' (all-z, {sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'all-z, {sum(trendline_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    else:
        #plt.title(r'$n_e$ vs $\Sigma_{SFR}$' + f' ({sum(sample_mask)} galaxies)')
        ax.text(0.01, 0.98, f'{sum(trendline_mask)} galaxies',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fs - 4)
    plt.plot(sfrsdrange, ne_25, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_25, color=colr, linestyle='dashed')
    plt.plot(sfrsdrange, ne_50, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_50, color=colr)
    plt.plot(sfrsdrange, ne_75, color='white', linewidth=3.5)
    plt.plot(sfrsdrange, ne_75, color=colr, linestyle='dashed')
    plt.xlabel(r'$\log{\Sigma_{SFR}/M_\odot/yr/kpc^2}$', fontsize=fs)
    plt.ylabel(r'$\log{n_e/cm^{-3}}$', fontsize=fs)
    plt.colorbar(label='count')
    plt.xlim(sfrsdmin, sfrsdmax)
    plt.ylim(1, 3)
    #plt.savefig(f'paper_figures/paper_ne_vs_sfrsd_total_{sample}.png', dpi=PLOT_DPI)
    plt.show()







def generate_all_plots():
    global SNR_LIM
    SNR_LIM = 5

    # Plot sfr and mass vs redshift with completeness limits labeled
    plot_redshift_vs_mass_sfr()

    # Plot sfr main sequence
    plot_sfr_ms(sample_mask=LO_Z_MASK)
    plot_sfr_ms(sample_mask=HI_Z_MASK)

    # Plot redshift vs ne
    plot_redshift_vs_ne(sample_mask=LO_Z_MASK)
    plot_redshift_vs_ne(sample_mask=HI_Z_MASK)

    # Plot sfr ms with ne colored bins
    plot_sfr_vs_mass_vs_ne(sample_mask=LO_Z_MASK)
    plot_sfr_vs_mass_vs_ne(sample_mask=HI_Z_MASK)

    # Plot sfrsd vs mass with ne colored bins
    plot_sfrsd_vs_mass_vs_ne(sample_mask=LO_Z_MASK)
    plot_sfrsd_vs_mass_vs_ne(sample_mask=HI_Z_MASK)

    # Plot ne vs mass, sfr, sfrsd with percentile trendlines
    plot_mass_sfr_sfrsd_vs_ne(sample_mask=LO_Z_MASK)
    plot_mass_sfr_sfrsd_vs_ne(sample_mask=HI_Z_MASK)

    # Plot SFRSD vs ne evolution in different bins
    plot_ne_vs_sfrsd_binned(sample_mask=LO_Z_MASK)
    plot_ne_vs_sfrsd_binned(sample_mask=HI_Z_MASK)

    # Plot BPT diagram color-coded by median ne
    plot_bpt_ne_color(sample_mask=LO_Z_MASK)
    plot_bpt_ne_color(sample_mask=HI_Z_MASK)
    bpt_ks_tests()


def generate_chosen_plots():
    #metallicity(sample_mask=LO_Z_MASK)
    #metallicity(sample_mask=HI_Z_MASK)
    plot_sfr_ms(sample_mask=LO_Z_MASK)
    plot_sfr_ms(sample_mask=HI_Z_MASK)
    #plot_ne_distribution(sample_mask=LO_Z_MASK)
    #plot_ne_distribution(sample_mask=HI_Z_MASK)

    pass


def main():
    global PLOT_DPI
    PLOT_DPI = 500

    #generate_all_plots()

    generate_chosen_plots()


if __name__ == '__main__':
    main()