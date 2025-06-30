import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import matplotlib.patheffects as path_effects

import numpy as np

from scipy.optimize import curve_fit

from scipy import stats

from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

import pandas as pd

from import_custom_catalog import CC
from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from sample_masks import (BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK,
                          BGS_SFR_MASK, BGS_MASS_MASK,
                          BGS_SNR_MASK, LO_Z_MASK, HI_Z_MASK,
                          Z50, Z90, M50, M90, SFR50, SFR90)
from sample_masks import bgs_ne_snr_cut
from spectrum_plot import Spectrum

from scipy.stats import binned_statistic_2d



def compare_stellar_mass():
    """
    Compares and plots the stellar masses derived from WISE color and CIGALE
    :return: none
    """

    # Make SNR cut of 10 for WISE bands 1 and 2
    w1_mask = CC.catalog['FLUX_W1'] * np.sqrt(CC.catalog['FLUX_IVAR_W1']) > SNR_LIM
    w2_mask = CC.catalog['FLUX_W2'] * np.sqrt(CC.catalog['FLUX_IVAR_W2']) > SNR_LIM

    mask = generate_combined_mask(w1_mask, w2_mask, CC.catalog['MSTAR_WISE'] != 0, CC.catalog['MSTAR_CIGALE'] != 0, CC.catalog['ZWARN'] == 0)

    # Extract WISE color and CIGALE derived masses from custom catalog
    wise_mstar = CC.catalog['MSTAR_WISE'][mask]
    cigale_mstar = CC.catalog['MSTAR_CIGALE'][mask]

    # Take the difference between the two
    difference = wise_mstar - cigale_mstar

    # Plot the comparison for diagnostic purposes
    # 2-axis histogram scatter plot
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(cigale_mstar, difference, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(7,13, 300), np.zeros(300), color='r')
    ax_main.set(xlabel=r"$\log{m_\star/m_\odot}$ from CIGALE", ylabel=r"$\log{m_{\star, WISE}/m_\odot} - \log{m_{\star, CIGALE}/m_\odot}$", xlim=(8,12), ylim=(-2, 3))

    ax_yDist.hist(difference, bins=500, orientation='horizontal', align='mid')
    ax_xDist.hist(cigale_mstar, bins=200, orientation='vertical', align='mid')
    ax_yDist.text(.03, .985, r"Mean: {:.2f}".format(np.average(difference)) + "\n" + r"Median: {:.2f}".format(np.median(difference)) + "\n" + r"$\sigma$: {:.2f}".format(np.std(difference)), verticalalignment='top',
                  transform=ax_yDist.transAxes)

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()


def check_sfr():
    """
    This plots histograms of SFR as calculated by CIGALE and with Halpha
    :return: none
    """

    cigale_sfr = np.array(CC.catalog['SFR_CIGALE'])
    halpha_sfr = np.array(CC.catalog['SFR_HALPHA'])

    sfr_mask = generate_combined_mask(cigale_sfr != 0, halpha_sfr != 0)

    ha_snr_mask = CC.catalog['HALPHA_AMP'] * np.sqrt(CC.catalog['HALPHA_AMP_IVAR']) > SNR_LIM
    hb_snr_mask = CC.catalog['HBETA_AMP'] * np.sqrt(CC.catalog['HBETA_AMP_IVAR']) > SNR_LIM

    snr_mask = generate_combined_mask(sfr_mask, ha_snr_mask, hb_snr_mask, CC.catalog['ZWARN'] == 0)

    cigale_sfr = cigale_sfr[snr_mask]
    halpha_sfr = halpha_sfr[snr_mask]

    plt.hist(cigale_sfr[cigale_sfr<25], bins=200)
    plt.xlabel("SFR (cigale)")
    plt.show()

    plt.hist(halpha_sfr, bins=200)
    plt.xlabel("SFR (halpha)")
    plt.show()


def compare_sfr():
    """
    This plots comparisons between the SFR as determined by CIGALE and Halpha flux.

    :return: none
    """
    snr_lim = 3

    cigale_sfr = np.array(CC.catalog['SFR_CIGALE'])[BGS_MASK]
    halpha_sfr = np.array(CC.catalog['SFR_HALPHA'])[BGS_MASK]
    stellar_mass = np.array(CC.catalog['MSTAR_CIGALE'])[BGS_MASK]
    gr_color = calc_color()
    gr_color = gr_color[BGS_MASK]

    snr_mask = BGS_SNR_MASK

    cigale_sfr = cigale_sfr[snr_mask]
    halpha_sfr = halpha_sfr[snr_mask]
    stellar_mass = stellar_mass[snr_mask]
    gr_color = gr_color[snr_mask]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(halpha_sfr, bins=200)
    ax.text(0.01, 0.99, f'mean: {np.average(halpha_sfr)}\nmedian: {np.median(halpha_sfr)}\nstdev: {np.std(halpha_sfr)}',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax.set(xlabel=r"SFR from $H\alpha$ ($\log{m_\star/m_\odot}$) (with my aperture correction including color)", xlim=(-3,2))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(cigale_sfr, bins=200)
    ax.text(0.01, 0.99, f'mean: {np.average(cigale_sfr)}\nmedian: {np.median(cigale_sfr)}\nstdev: {np.std(cigale_sfr)}',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax.set(xlabel=r"SFR from CIGALE ($\log{m_\star/m_\odot}$)")#, xlim=(-8,2.5))
    plt.show()

    #print(cigale_sfr)
    #print(f"SFR Avg: {np.average(cigale_sfr)}, stdev: {np.std(cigale_sfr)}")
    #print(halpha_sfr)
    #print(f"SFR Avg: {np.average(halpha_sfr)}, stdev: {np.std(halpha_sfr)}")

    mean_diff = np.average((halpha_sfr - cigale_sfr))
    err_diff = np.std((halpha_sfr - cigale_sfr))

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(halpha_sfr, halpha_sfr - cigale_sfr, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(-20,30, 100), np.zeros(100), color='r')
    ax_main.set(xlabel=r"SFR$_{H\alpha}$ [log(M$_\odot$/yr)]", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(-2,2), ylim=(-1.5,1.5))
    #ax_main.set(xlabel=r"$F_{r,model}/F_{r,aperture}$", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(0,20), ylim=(-10,5))

    ax_yDist.hist(halpha_sfr - cigale_sfr, bins=200, orientation='horizontal', align='mid')
    ax_yDist.text(1, .95, "mean: {:.3f}".format(mean_diff) + "\nstdev: {:.3f}".format(err_diff),
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax_xDist.hist(halpha_sfr, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(stellar_mass, halpha_sfr - cigale_sfr, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(-20,30, 100), np.zeros(100), color='r')
    ax_main.set(xlabel=r"Stellar Mass [$\log{M_\odot}$]", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(8,11.5), ylim=(-1.5,1.5))
    #ax_main.set(xlabel=r"$F_{r,model}/F_{r,aperture}$", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(0,20), ylim=(-10,5))

    ax_yDist.hist(halpha_sfr - cigale_sfr, bins=200, orientation='horizontal', align='mid')
    ax_yDist.text(1, .95, "mean: {:.3f}".format(mean_diff) + "\nstdev: {:.3f}".format(err_diff),
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax_xDist.hist(stellar_mass, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(gr_color, halpha_sfr - cigale_sfr, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(-20,30, 100), np.zeros(100), color='r')
    ax_main.set(xlabel=r"g-r", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(-.3,1.6), ylim=(-1.5,1.5))
    #ax_main.set(xlabel=r"$F_{r,model}/F_{r,aperture}$", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(0,20), ylim=(-10,5))

    ax_yDist.hist(halpha_sfr - cigale_sfr, bins=200, orientation='horizontal', align='mid')
    ax_yDist.text(1, .95, "mean: {:.3f}".format(mean_diff) + "\nstdev: {:.3f}".format(err_diff),
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax_xDist.hist(gr_color, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()


def compare_sfr_snr():
    """
    ider what this does. Can probably be deleted.
    :return:
    """
    cigale_sfr = np.array(CC.catalog['SFR_CIGALE'])
    halpha_sfr = np.array(CC.catalog['SFR_HALPHA'])

    sfr_mask = generate_combined_mask(cigale_sfr != 0, halpha_sfr != 0)
    ha_snr_hi_mask = CC.catalog['HALPHA_AMP'] * np.sqrt(CC.catalog['HALPHA_AMP_IVAR']) > 3
    hb_snr_hi_mask = CC.catalog['HBETA_AMP'] * np.sqrt(CC.catalog['HBETA_AMP_IVAR']) > 3
    ha_snr_lo_mask = CC.catalog['HALPHA_AMP'] * np.sqrt(CC.catalog['HALPHA_AMP_IVAR']) <= 3
    hb_snr_lo_mask = CC.catalog['HBETA_AMP'] * np.sqrt(CC.catalog['HBETA_AMP_IVAR']) <= 3
    ha_snr_lo_mask_2 = CC.catalog['HALPHA_AMP'] * np.sqrt(CC.catalog['HALPHA_AMP_IVAR']) > 1
    hb_snr_lo_mask_2 = CC.catalog['HBETA_AMP'] * np.sqrt(CC.catalog['HBETA_AMP_IVAR']) > 1
    snr_hi_mask = generate_combined_mask(sfr_mask, ha_snr_hi_mask, hb_snr_hi_mask, CC.catalog['ZWARN'] == 0)
    snr_lo_mask = generate_combined_mask(sfr_mask, ha_snr_lo_mask, hb_snr_lo_mask, CC.catalog['ZWARN'] == 0)
    snr_lo_mask_2 = generate_combined_mask(sfr_mask, ha_snr_lo_mask, ha_snr_lo_mask_2, hb_snr_lo_mask, hb_snr_lo_mask_2, CC.catalog['ZWARN'] == 0)
    cigale_sfr_hi = cigale_sfr[snr_hi_mask]
    halpha_sfr_hi = halpha_sfr[snr_hi_mask]
    cigale_sfr_lo = cigale_sfr[snr_lo_mask]
    halpha_sfr_lo = halpha_sfr[snr_lo_mask]

    plt.scatter(halpha_sfr_hi, halpha_sfr_hi - cigale_sfr_hi, marker='.', alpha=0.05, label=r'$SNR > 3$')
    plt.scatter(halpha_sfr_lo, halpha_sfr_lo - cigale_sfr_lo, marker='^', alpha=0.05, label=r"$SNR \leq 3$")
    plt.xlabel(r"SFR$_{H\alpha}$ [log(M$_\odot$/yr)]")
    plt.ylabel(r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]")
    plt.legend()
    #plt.xlim(-9,4)
    #plt.ylim(-10,4)
    plt.show()


def plot_color_excess_vs_sfr_comparison():
    """
    Plots E(B-V) against the difference between Halpha and CIGALE SFR
    :return: none
    """
    snr_lim = 5

    cigale_sfr = np.array(CC.catalog['SFR_CIGALE'])
    halpha_sfr = np.array(CC.catalog['SFR_HALPHA'])

    sfr_mask = generate_combined_mask(cigale_sfr != 0, halpha_sfr != 0)
    ha_snr_mask = CC.catalog['HALPHA_AMP'] * np.sqrt(CC.catalog['HALPHA_AMP_IVAR']) > snr_lim
    hb_snr_mask = CC.catalog['HBETA_AMP'] * np.sqrt(CC.catalog['HBETA_AMP_IVAR']) > snr_lim
    snr_mask = generate_combined_mask(sfr_mask, ha_snr_mask, hb_snr_mask, CC.catalog['ZWARN'] == 0)
    cigale_sfr = cigale_sfr[snr_mask]
    halpha_sfr = halpha_sfr[snr_mask]
    ebv = CC.catalog['EBV_CALC'][snr_mask]

    difference = halpha_sfr - cigale_sfr

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(ebv, difference, marker='.', alpha=.1)
    plt.xlabel("E(B-V) (from my sfr calculation)")
    plt.ylabel(r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]")
    plt.show()


def extract_high_sfr():
    """
    ***ALSO DEPRECATED. CAN LIKELY DELETE
    :return:
    """
    tids = [x for _, x in sorted(zip(CC.catalog['SFR_HALPHA'], CC.catalog['TARGETID']))]
    tids.reverse()
    print(tids[:20])


def signed_power_law(x, a, b):
    """
    Generic power law function for use in fitting algorithms
    :param x:
    :param a:
    :param b:
    :return:
    """
    return -a * x**b


def exp_decay(x, a, b):
    """
    Generic exponential decay function for use in fitting algorithms
    :param x:
    :param a:
    :param b:
    :return:
    """
    return -a * np.exp(-b * x)


def check_color_ratio():
    """
    ***NOT SURE WHAT THIS DOES EITHER. CHECK FOR DEPENDENCIES AND DOCUMENT/DELETE
    :return: none
    """

    snr_lim = 5
    ha_snr_mask = CC.catalog['HALPHA_AMP'] * np.sqrt(CC.catalog['HALPHA_AMP_IVAR']) > snr_lim
    hb_snr_mask = CC.catalog['HBETA_AMP'] * np.sqrt(CC.catalog['HBETA_AMP_IVAR']) > snr_lim
    balmer_snr_mask = generate_combined_mask(ha_snr_mask, hb_snr_mask)

    # Convert table columns to a NumPy array (faster than list comprehension)
    arr = np.array([CC.catalog['APFLUX_G'], CC.catalog['APFLUX_R'], CC.catalog['APFLUX_Z']])  # Shape (3, N, 8)
    # Check if all elements are zero along the last axis (8-tuple elements)
    is_zero_tuple = np.all(arr != 0, axis=-1)  # Shape (3, N)
    # Check if all three columns have zero tuples per row
    apflux_mask = np.all(is_zero_tuple, axis=0)  # Shape (N,)

    full_color_mask = generate_combined_mask(balmer_snr_mask, apflux_mask)

    apflux_g = CC.catalog['APFLUX_G'][full_color_mask]
    apflux_r = CC.catalog['APFLUX_R'][full_color_mask]
    apflux_z = CC.catalog['APFLUX_Z'][full_color_mask]

    print(apflux_g.shape)

    sm_g = np.mean(apflux_g[:, 0])
    sm_r = np.mean(apflux_r[:, 0])
    sm_z = np.mean(apflux_z[:, 0])

    g_array = np.zeros(7)
    r_array = np.zeros(7)
    z_array = np.zeros(7)

    for i in range(1, 8):
        g_array[i-1] = np.mean(apflux_g[:, i])/sm_g
        r_array[i-1] = np.mean(apflux_r[:, i])/sm_r
        z_array[i-1] = np.mean(apflux_z[:, i])/sm_z

    arcsecs = [0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.0]
    color_diff = -2.5 * (np.log10(g_array/r_array))
    print(list(color_diff))

    # Fit the model
    params, _ = curve_fit(signed_power_law, arcsecs, color_diff, p0=(0.1, -1))

    # Extract fitted parameters
    a_fit, b_fit = params
    print(f"Fitted parameters: a = {a_fit:.4f}, b = {b_fit:.4f}")

    # Generate smooth curve
    x_fit = np.linspace(min(arcsecs), max(arcsecs), 100)
    y_fit = signed_power_law(x_fit, a_fit, b_fit)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.plot(arcsecs, color_diff, '--', alpha=.2)
    #ax.plot(x_fit, y_fit, color="red", alpha=0.6)
    ax.plot(arcsecs, color_diff, '.')
    for xy in zip(arcsecs, color_diff):
        ax.annotate(f"({xy[0]:.2f}, {xy[1]:.2f})", xy=xy, textcoords='data')

    plt.xlabel("Aperture radius (arcsec) (of numerator flux)")
    plt.ylabel(r'$(g-r)_i - (g-r)_0$')
    plt.show()


def calc_color():
    """
    Calculates g-r color for all catalog objects
    :return: float array of g-r color (catalog length)
    """
    mag_g = CC.catalog['ABSMAG01_SDSS_G'] - CC.catalog['KCORR01_SDSS_G']
    mag_r = CC.catalog['ABSMAG01_SDSS_R'] - CC.catalog['KCORR01_SDSS_R']

    gr_col = mag_g - mag_r

    return gr_col


def angular_to_physical_radius(theta_e_arcsec, z):
    """
    Converts an angular effective radius in arcseconds to a physical radius in kpc.

    :param theta_e_arcsec: effective radius in arcseconds (value or array)
    :param z: redshift (value or array)
    :return: physical effective radius in kpc (singular value or array)
    """

    # Convert angular diameter distance from Mpc to kpc
    D_A = cosmo.angular_diameter_distance(z).to(u.kpc)  # kpc

    # Convert arcseconds to radians
    theta_e_rad = (theta_e_arcsec * u.arcsec).to(u.radian).value

    # Compute physical effective radius (R_e = θ_e * D_A)
    R_e_kpc = theta_e_rad * D_A.value

    return R_e_kpc


def identify_agn(input_mask):
    """
    Function to identify BPT AGN for any subsample
    :param input_mask: catalog-length boolean array define subsample to check for AGN
    :return: catalog-length boolean array to mask for BPT agn
    """
    snr_lim = 3

    nii = CC.catalog['NII_6584_FLUX'][input_mask]
    nii_snr = nii/(np.sqrt(1/CC.catalog['NII_6584_FLUX_IVAR'][input_mask]))
    ha = CC.catalog['HALPHA_FLUX'][input_mask]
    ha_snr = ha/(np.sqrt(1/CC.catalog['HALPHA_FLUX_IVAR'][input_mask]))
    oiii = CC.catalog['OIII_5007_FLUX'][input_mask]
    oiii_snr = oiii/(np.sqrt(1/CC.catalog['OIII_5007_FLUX_IVAR'][input_mask]))
    hb = CC.catalog['HBETA_FLUX'][input_mask]
    hb_snr = hb/(np.sqrt(1/CC.catalog['HBETA_FLUX_IVAR'][input_mask]))

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    zero_mask = generate_combined_mask(nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0)

    # Requiring all lines to have at least SNR of 1
    full_mask = generate_combined_mask(zero_mask, nii_snr > snr_lim, ha_snr > snr_lim, oiii_snr > snr_lim, hb_snr > snr_lim)

    nh = np.log10(nii / ha)  # this is the x-axis on bpt diagram
    oh = np.log10(oiii / hb)

    agn_mask = np.zeros(len(nh), dtype=bool)

    for i, (x, y) in enumerate(zip(nh, oh)):
        if full_mask[i]:
            if y > 0.61 / (x - 0.47) + 1.19 and y > 0.61 / (x - 0.47) + 1.19:
                agn_mask[i] = 1

    """
    x_for_line_1 = np.log10(np.logspace(-5,.049,300))
    hii_agn_line = 0.61/(x_for_line_1 - 0.05) + 1.3

    x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
    composite_line_2 = 0.61/(x_for_line_2 - 0.47) + 1.19

    x_for_line_3 = np.linspace(-.13,2,100)
    agn_line_3 = 2.144507*x_for_line_3 + 0.465028
    """

    return agn_mask


def sfr_diagnostic():
    """
    Plots A(Ha) vs SFR(Ha) with BPT agn flagged
    :return: none
    """
    mask = generate_combined_mask(BGS_MASK, CAT_SFR_MASK)

    agn_mask = identify_agn(mask)

    sfr_ha = CC.catalog['SFR_HALPHA'][mask]
    sfr_cigale = CC.catalog['SFR_CIGALE'][mask]
    a_ha = CC.catalog['A_HALPHA'][mask]

    plt.scatter(a_ha[~agn_mask], sfr_ha[~agn_mask] - sfr_cigale[~agn_mask], alpha=0.03, marker='o', label=f'Other ({sum(~agn_mask)})')
    plt.scatter(a_ha[agn_mask], sfr_ha[agn_mask] - sfr_cigale[agn_mask], alpha=0.03, marker='s', label=f'BPT AGN ({sum(agn_mask)})')
    plt.xlabel(r"A(H$\alpha$)")
    plt.ylabel(r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]")
    plt.xlim(-2, 4)
    plt.ylim(-3,2)
    plt.legend()
    plt.show()


def sfr_ms(plot=False):
    """
    Calculates a main sequence 2nd order polynomial fit for the sfr main sequence
    Includes a constant specific star formation rate cut
    :return: fit parameters: 1st order coefficient, 2nd order coefficient, constant
    """
    mask = generate_combined_mask(BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK)

    sfr = CC.catalog['SFR_HALPHA'][mask]
    mstar = CC.catalog['MSTAR_CIGALE'][mask]

    valid_sfr_mstar_mask = generate_combined_mask(~np.isnan(mstar), ~np.isnan(sfr))

    sfr = sfr[valid_sfr_mstar_mask]
    mstar = mstar[valid_sfr_mstar_mask]

    specific_sfr = np.log10((10**sfr) / 10**(mstar))

    sSFR_cut = -100

    passive_galaxy_mask = specific_sfr > sSFR_cut

    o1, o2, c = np.polyfit(mstar[passive_galaxy_mask], sfr[passive_galaxy_mask], 2)
    #m, b = np.polyfit(mstar[passive_galaxy_mask], sfr[passive_galaxy_mask], 1)

    x = np.linspace(0,20,100)
    y = o1 * x**2 + o2 * x + c
    #y = m * x + b
    y2 = x * sSFR_cut

    if plot:
        plt.hist2d(mstar, sfr, bins=(200,70), norm=mpl.colors.LogNorm())
        plt.plot(x, y, color='r', label='polynomial fit')
        #plt.plot(x, y2, color='k', linestyle='--', label='sSFR cut')
        plt.xlim(8, 11.5)
        plt.ylim(-3, 2.5)
        plt.colorbar(label='count')
        plt.title("SFR main sequence with sSFR cut of $10^{-10}$")
        plt.xlabel(r'$\log{M_\star/M_\odot}$')
        plt.ylabel(r'$\log{M_\odot/yr}$')
        plt.legend(loc='upper left')
        plt.show()

        plt.hist2d(mstar, specific_sfr, bins=(200,70), norm=mpl.colors.LogNorm())
        plt.plot(x, np.ones(len(x))*sSFR_cut, color='k', linestyle='--', label='sSFR cut')
        plt.xlim(8, 11.5)
        plt.colorbar(label='count')
        #plt.ylim(-.35, .35)
        plt.xlabel(r'$\log{M_\star/M_\odot}$')
        plt.ylabel(r'$\log{SFR / M_\star}$')
        plt.legend(loc='upper left')
        plt.show()

    return o1, o2, c


def sfr_spread_plots():
    """
    Makes diagnostic plots of Delta SFR vs A(Ha) and ne, identifies bpt agn
    :return: none
    """

    mask = generate_combined_mask(BGS_MASK, CAT_SFR_MASK)

    o1, o2, c = sfr_ms(plot=False)

    mstar = CC.catalog['MSTAR_CIGALE'][mask]
    sfr = CC.catalog['SFR_HALPHA'][mask]
    ne_oii = CC.catalog['NE_OII'][mask]
    ne_sii = CC.catalog['NE_SII'][mask]
    ne = (ne_oii + ne_sii) / 2
    A_ha = CC.catalog['A_HALPHA'][mask]

    agn_mask = identify_agn(mask)

    distance = np.zeros(len(sfr))

    for i, (m, s) in enumerate(zip(mstar, sfr)):
        distance[i] = distance_from_ms(m, s, o1, o2, c)

    scatter = plt.scatter(mstar, distance, marker='+', alpha=0.05, c=A_ha, vmax=-.5, vmin=2.5, cmap='plasma')
    plt.xlim(8, 11.5)
    plt.ylim(-3, 2.5)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$')
    plt.ylabel(r'$\Delta$ SFR [$\log{M_\odot/yr}$]')
    plt.colorbar(ScalarMappable(cmap=scatter.get_cmap(), norm=scatter.norm), ax=plt.gca(), label=r"$A(H\alpha)$")
    plt.show()

    plt.scatter(np.log10(ne[~agn_mask]), distance[~agn_mask], marker='+', alpha=0.05, label='Other')
    plt.scatter(np.log10(ne[agn_mask]), distance[agn_mask], marker='+', alpha=0.05, label='AGN')
    plt.xlim(0, 7)
    plt.ylim(-3, 2.5)
    plt.xlabel(r'$n_e$ [cm$^{-3}$]')
    plt.ylabel(r'$\Delta$ SFR [$\log{M_\odot/yr}$]')
    plt.legend()
    plt.show()

    plt.scatter(A_ha[~agn_mask], distance[~agn_mask], marker='+', alpha=0.05, label='Other')
    plt.scatter(A_ha[agn_mask], distance[agn_mask], marker='+', alpha=0.05, label='AGN')
    plt.xlim(-1, 3)
    plt.ylim(-2, 2)
    plt.xlabel(r'A(H$\alpha$)')
    plt.ylabel(r'$\Delta$ SFR [$\log{M_\odot/yr}$]')
    plt.legend()
    plt.show()

    a_mask = generate_combined_mask(~np.isnan(A_ha), ~np.isnan(distance))

    plt.hist2d(A_ha[a_mask], distance[a_mask], bins=200, norm=mpl.colors.LogNorm())
    plt.xlim(-1, 3)
    plt.ylim(-2, 2)
    plt.xlabel(r'A(H$\alpha$)')
    plt.ylabel(r'$\Delta$ SFR [$\log{M_\odot/yr}$]')
    plt.show()


def sfr_mstar_plots():
    mask = generate_combined_mask(BGS_MASK, CAT_SFR_MASK)
    mstar = CC.catalog['MSTAR_CIGALE'][mask]
    sfr = CC.catalog['SFR_HALPHA'][mask]
    ne_oii = CC.catalog['NE_OII'][mask]
    ne_sii = CC.catalog['NE_SII'][mask]
    ne = (ne_oii + ne_sii) / 2

    A_ha = CC.catalog['A_HALPHA'][mask]
    a_mask = ~np.isnan(A_ha)

    scatter = plt.scatter(mstar, sfr, marker='+', alpha=0.05, c=A_ha, vmax=-.5, vmin=2.5, cmap='plasma')
    plt.xlabel(r'$M_\star [\log{M_\odot}]$')
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]')
    plt.xlim(8, 11.5)
    plt.ylim(-3, 2.5)
    plt.colorbar(ScalarMappable(cmap=scatter.get_cmap(), norm=scatter.norm), ax=plt.gca(), label=r"$A(H\alpha)$")
    plt.show()

    scatter = plt.scatter(mstar, sfr, marker='+', alpha=0.05, c=np.log10(ne), vmax=4, vmin=2.7, cmap='plasma')
    plt.xlabel(r'$M_\star [\log{M_\odot}]$')
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]')
    plt.xlim(8, 11.5)
    plt.ylim(-3, 2.5)
    plt.colorbar(ScalarMappable(cmap=scatter.get_cmap(), norm=scatter.norm), ax=plt.gca(), label=r"$\log{n_e} ~[\mathrm{cm}^{-3}]$")
    plt.show()

    nan_mask = generate_combined_mask(~np.isnan(A_ha), ~np.isnan(mstar))

    plt.hist2d(mstar[nan_mask], A_ha[nan_mask], bins=500, norm=mpl.colors.LogNorm())
    plt.xlabel(r'$M_\star [\log{M_\odot}]$')
    plt.ylabel(r'$A(H\alpha)$')
    plt.xlim(8, 11.5)
    plt.ylim(-2, 4)
    plt.show()

    nan_mask = generate_combined_mask(~np.isnan(A_ha), ~np.isnan(sfr))

    plt.hist2d(sfr[nan_mask], A_ha[nan_mask], bins=200, norm=mpl.colors.LogNorm())
    plt.xlabel(r'SFR [$\log{M_\odot/yr}$]')
    plt.ylabel(r'$A(H\alpha)$')
    plt.xlim(-3, 2.5)
    plt.ylim(-2, 4)
    plt.show()


def distance_from_ms(mass, sfr, o1, o2, c):
    dist = sfr - (o1 * mass**2 + o2 * mass + c)
    return dist


def identify_strange_a():
    mask = generate_combined_mask(BGS_MASK, CAT_SFR_MASK)
    tids = CC.catalog['TARGETID'][mask]
    A = CC.catalog['A_HALPHA'][mask]

    print(tids[A < 0])


def plot_sfr_ha_binned():
    print("starting")
    mask = generate_combined_mask(BGS_MASK, CAT_SFR_MASK)

    o1, o2, c = sfr_ms()

    mstar = CC.catalog['MSTAR_CIGALE'][mask]
    sfr = CC.catalog['SFR_HALPHA'][mask]
    ne_oii = CC.catalog['NE_OII'][mask]
    ne_sii = CC.catalog['NE_SII'][mask]
    ne = (ne_oii + ne_sii) / 2

    A_ha = CC.catalog['A_HALPHA'][mask]

    agn_mask = identify_agn(mask)

    #valid_sfr_mstar_mask = generate_combined_mask(~np.isnan(mstar), ~np.isnan(sfr))
    #agn_mask = generate_combined_mask(~np.isnan(mstar), ~np.isnan(sfr), AGN_MASK)
    #non_agn_mask = generate_combined_mask(~np.isnan(mstar), ~np.isnan(sfr), ~AGN_MASK)
    #valid_sfr = sfr[valid_sfr_mstar_mask]
    #valid_mstar = mstar[valid_sfr_mstar_mask]

    distance = np.zeros(len(sfr))

    for i, (m, s) in enumerate(zip(mstar, sfr)):
        distance[i] = distance_from_ms(m, s, o1, o2, c)

    x_limits = (-1, 3)
    y_limits = (-2, 2)
    m = 7
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            m += 0.5
            ax = axes[i, j]
            mass_mask = generate_combined_mask(mstar >= m, mstar < m + 0.5)
            ax.scatter(A_ha[mass_mask], distance[mass_mask], marker='+', alpha=0.05)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            # Remove tick labels for cleaner look except for leftmost and bottom plots
            if i < 8:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.text(0.5, 0.04, r'A(H$\alpha$)', ha='center', fontsize=16)  # Move x label up slightly
    fig.text(0.04, 0.5, r'$\Delta$ SFR [$\log{M_\odot/yr}$]', va='center', rotation=90, fontsize=16)

    #plt.tight_layout()
    plt.show()

    """
    for i in range(8, 11.5):
        mass_mask = generate_combined_mask(mstar >= i, mstar < i + 0.5)
        plt.scatter(distance[mass_mask], A_ha[mass_mask], marker='+', alpha=0.05)
        #p25, p50, p75 = np.percentile(A_ha[generate_combined_mask(mstar >= i, mstar < i + .5)], (25, 50, 75))
        plt.xlabel(r'$M_\star [\log{M_\odot}]$')
        plt.ylabel(r'$\Delta$ SFR [$\log{M_\odot/yr}$]')
    """


def sfr_mass_ne_colorplot(custom_mask=None, title=''):
    """
    Generates a plot of sfr vs mass with each bin color-coded by median ne
    :param custom_mask: Optional extra mask that is placed after snr cuts
    :return:
    """

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]  # bgs length
    ne, _ = bgs_ne_snr_cut()

    snr_mask = BGS_SNR_MASK

    mass = mass[snr_mask]
    sfr = sfr[snr_mask]
    ne = ne[snr_mask]

    if custom_mask is not None:
        mass = mass[custom_mask]
        sfr = sfr[custom_mask]
        ne = ne[custom_mask]

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
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='viridis', shading='auto', vmin=1.824, vmax=2.224)

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{M_\odot/yr}$', size=fs)
    plt.title(title)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_ne_narrow.png')
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr, ne, statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{M_\odot/yr}$', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_counts.png')
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

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$\log{M_\star/M_\odot}$', size=fs)
    plt.ylabel(r'$\log{M_\odot/yr}$', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_iqr.png')
    plt.show()


def sfr_mass_ne_colorplot_largebin():

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]  # bgs length
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    snr_mask = generate_combined_mask(BGS_SFR_MASK, ne_mask)

    mass = mass[snr_mask]
    sfr = sfr[snr_mask]
    ne = ne[snr_mask]

    valid_mask = np.isfinite(mass) & np.isfinite(sfr) & np.isfinite(ne)
    mass = mass[valid_mask]
    sfr = sfr[valid_mask]
    ne = ne[valid_mask]

    ### Bins are arranged to match histogram.

    valid_mask = generate_combined_mask(mass > 8, mass < 12, sfr > -1.0, sfr < 1.5)
    mass = mass[valid_mask]
    sfr = sfr[valid_mask]
    ne = ne[valid_mask]

    # font size for labels
    fs = 16

    x_edges = np.array([8, 9, 10, 11, 12])  # 4 bins
    y_edges = np.linspace(-1, 1.5, 6)  # 5 bins from -1 to 1.5

    stat, _, _, _ = binned_statistic_2d(
        mass, sfr, ne, statistic='median', bins=[x_edges, y_edges]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='viridis', shading='auto', vmin=1.95, vmax=2.35)

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]', size=fs)

    # Loop over bins and add text
    for i in range(len(x_edges)-1):
        for j in range(len(y_edges)-1):
            # Get bin center
            x_center = 0.5 * (x_edges[i] + x_edges[i + 1])
            y_center = 0.5 * (y_edges[j] + y_edges[j + 1])

            # Get the median ne value
            value = stat[i, j]

            # Only show text if value is not nan
            if np.isfinite(value):
                txt = plt.text(x_center, y_center, f"{value:.2f}",
                               ha='center', va='center', color='white', fontsize=16, weight='bold')
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='black'),
                    path_effects.Normal()
                ])
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_ne_block.png')
    plt.show()

    stat, _, _, _ = binned_statistic_2d(
        mass, sfr, ne, statistic='count', bins=[x_edges, y_edges]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=1))

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_counts_block.png')
    plt.show()

    # Custom function to calculate iqr
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)

    # Compute the inter-quartile range of `ne` in each bin
    stat, _, _, _ = binned_statistic_2d(
        mass, sfr, ne, statistic=iqr, bins=[x_edges, y_edges]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Blues', shading='auto', vmax=.6)#, norm=mpl.colors.LogNorm())

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_iqr_block.png')
    plt.show()


def delta_sfr_mass_ne_colorplot():
    ##################

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]  # bgs length
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    snr_mask = generate_combined_mask(BGS_SFR_MASK, ne_mask)

    mass = mass[snr_mask]
    sfr = sfr[snr_mask]
    ne = ne[snr_mask]

    valid_mask = np.isfinite(mass) & np.isfinite(sfr) & np.isfinite(ne)
    mstar = mass[valid_mask]
    sfr = sfr[valid_mask]
    ne = ne[valid_mask]

    o1, o2, c = sfr_ms()

    distance = distance_from_ms(mstar, sfr, o1, o2, c)

    ###################

    # Define the number of bins
    # comment out when using large bins for matching histograms
    x_bins = 120
    y_bins = 60

    # font size for labels
    fs = 16

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr, ne, statistic='median', bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='viridis', shading='auto', vmin=1.5, vmax=2.5)

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_ne.png')
    plt.show()

    # Compute the count in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr, ne, statistic='count', bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='Greys', shading='auto', norm=mpl.colors.LogNorm(vmin=0.1))

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_counts.png')
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

    plt.xlim(8, 12)
    plt.ylim(-1, 1.5)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_iqr.png')
    plt.show()


def plot_oii_luminosity():

    l_oii = CC.catalog['OII_LUMINOSITY']
    oii_1_snr = CC.catalog['OII_3726_FLUX'] * np.sqrt(CC.catalog['OII_3726_FLUX_IVAR']) > 3
    oii_2_snr = CC.catalog['OII_3729_FLUX'] * np.sqrt(CC.catalog['OII_3729_FLUX_IVAR']) > 3
    oii_snr = generate_combined_mask(oii_1_snr, oii_2_snr, BGS_MASK)

    oii_ratio = CC.catalog['OII_DOUBLET_RATIO']

    plt.hist(l_oii[oii_snr], bins=200)
    plt.xlabel(r"$L_{[OII]}$")
    plt.show()

    plt.hist2d( l_oii[oii_snr], oii_ratio[oii_snr], bins=200, norm=mpl.colors.LogNorm())
    plt.ylabel(r'$F_{\lambda 3726} / F_{\lambda 3729}$')
    plt.xlabel(r"$L_{[OII]}$")
    plt.show()


def generate_ne_split_snr_complex(snr_lim=5):
    """
    Generates a float array of valid ne values and a BGS-length boolean array for those values.
    Valid ne values require one or both of the [OII] and [SII] line pairs to have SNR >=5
    If both are above, the mean of the two ne values is taken
    :param snr_lim: Changes the required snr for electron density. Default 5
    :return: float array of ne values (BGS length), boolean array mask for ne values (BGS length)
    """
    #tid = CC.catalog['TARGETID'][BGS_MASK]
    oii_1_snr = CC.catalog['OII_3726_FLUX'] * np.sqrt(CC.catalog['OII_3726_FLUX_IVAR']) > snr_lim  # mask
    oii_2_snr = CC.catalog['OII_3729_FLUX'] * np.sqrt(CC.catalog['OII_3729_FLUX_IVAR']) > snr_lim  # mask
    oii_snr = generate_combined_mask(oii_1_snr, oii_2_snr)  # mask for oii - catalog length
    sii_1_snr = CC.catalog['SII_6716_FLUX'] * np.sqrt(CC.catalog['SII_6716_FLUX_IVAR']) > snr_lim
    sii_2_snr = CC.catalog['SII_6731_FLUX'] * np.sqrt(CC.catalog['SII_6731_FLUX_IVAR']) > snr_lim
    sii_snr = generate_combined_mask(sii_1_snr, sii_2_snr)  # mask for sii - catalog length
    oii_snr = oii_snr[BGS_MASK]  # mask - bgs length
    sii_snr = sii_snr[BGS_MASK]  # mask - bgs length

    ne_oii = np.log10(CC.catalog['NE_OII'][BGS_MASK])  # ne values, bgs length
    ne_sii = np.log10(CC.catalog['NE_SII'][BGS_MASK])  # ne values, bgs length

    ne_oii = ne_oii.filled(-999)
    ne_sii = ne_sii.filled(-999)

    ne = np.zeros(len(ne_oii))  # bgs length

    for i in range(len(oii_snr)):
        # first, if ne_oii is a number and ne_sii is not:
        #print(ne_oii[i], ne_sii[i], oii_snr[i], sii_snr[i])
        if ne_oii[i] > 0 and ne_sii[i] < 0:
            # if ne_oii has good snr, keep it. otherwise discard.
            #print(1)
            if oii_snr[i] == True:
                #print(2)
                ne[i] = ne_oii[i]
            else:
                ne[i] = -999
        # second, if ne_oii is not a number and ne_sii is:
        elif ne_oii[i] < 0 and ne_sii[i] > 0:
            #print(3)
            # if ne_sii has good snr, keep it. otherwise discard.
            if sii_snr[i] == True:
                #print(4)
                ne[i] = ne_sii[i]
            else:
                ne[i] = -999
        # third, if neither are numbers, discard:
        elif ne_oii[i] < 0 and ne_sii[i] < 0:
            #print(5)
            ne[i] = -999
        # now we have implied that both are numbers
        # now we can check snr of each value
        elif oii_snr[i] == True and sii_snr[i] == False:
            #print(6)
            ne[i] = ne_oii[i]
        elif oii_snr[i] == False and sii_snr[i] == True:
            #print(7)
            ne[i] = ne_sii[i]
        elif oii_snr[i] == True and sii_snr[i] == True:
            #print(8)
            ne[i] = (ne_sii[i] + ne_oii[i]) / 2
        elif oii_snr[i] == False and sii_snr[i] == False:
            #print(9)
            ne[i] = -999

    ne_mask = ne > -998

    ne = np.array(ne)

    # ne and ne_mask are both BGS length
    return ne, ne_mask


def sfrsd_mass_ne_colorplot():

    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]  # bgs length
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    snr_mask = generate_combined_mask(BGS_SFR_MASK, ne_mask)

    mass = mass[snr_mask]
    sfr_sd = sfr_sd[snr_mask]
    ne = ne[snr_mask]

    valid_mask = np.isfinite(mass) & np.isfinite(sfr_sd) & np.isfinite(ne)
    mass = mass[valid_mask]
    sfr_sd = sfr_sd[valid_mask]
    ne = ne[valid_mask]

    # Define the number of bins
    x_bins = 120
    y_bins = 60

    # font size for labels
    fs = 16

    # Compute the median ne in each bin
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        mass, sfr_sd, ne, statistic='median', bins=[x_bins, y_bins]
    )

    # Plot the result
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, stat.T, cmap='viridis', shading='auto', vmin=1.95, vmax=2.35)

    plt.xlim(7.5, 12)
    plt.ylim(-8.5, -6.5)
    cbar = plt.colorbar()
    cbar.set_label(r'median $\log{n_e}$', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/pc^2}$]', size=fs)
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

    plt.xlim(7.5, 12)
    plt.ylim(-8.5, -6.5)
    cbar = plt.colorbar()
    cbar.set_label(r'count', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/pc^2}$]', size=fs)
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

    plt.xlim(7.5, 12)
    plt.ylim(-8.5, -6.5)
    cbar = plt.colorbar()
    cbar.set_label(r'IQR (dex)', fontsize=fs)
    plt.xlabel(r'$M_\star [\log{M_\odot}]$', size=fs)
    plt.ylabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/pc^2}$]', size=fs)
    plt.savefig('figures/ne_plots/sfr_vs_mstar_vs_iqr.png')
    plt.show()


def sfr_sd_vs_ne_plots():

    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    sfr_sd = CC.catalog['SFR_SD'][BGS_MASK]  # bgs length
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    snr_mask = generate_combined_mask(BGS_SFR_MASK, ne_mask)

    sfr = sfr[snr_mask]
    sfr_sd = sfr_sd[snr_mask]
    ne = ne[snr_mask]

    valid_mask = np.isfinite(sfr) & np.isfinite(sfr_sd) & np.isfinite(ne)
    sfr = sfr[valid_mask]
    sfr_sd = sfr_sd[valid_mask]
    ne = ne[valid_mask]

    ne_75 = []
    ne_50 = []
    ne_25 = []
    sfrrange = []
    b = 0.1

    for i in np.arange(-8.5, -6.5, b):
        bin = generate_combined_mask(sfr_sd >= i, sfr_sd < i + b)
        p25, p50, p75 = np.percentile(ne[bin], (25, 50, 75))
        ne_25.append(p25)
        ne_50.append(p50)
        ne_75.append(p75)
        sfrrange.append(i + b * 0.5)

    plt.scatter(sfr_sd, ne, marker='+', alpha=0.05, color='tab:orange')
    plt.plot(sfrrange, ne_25, color='tab:blue', linestyle='dashed')
    plt.plot(sfrrange, ne_50, color='tab:blue')
    plt.plot(sfrrange, ne_75, color='tab:blue', linestyle='dashed')
    plt.xlabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/kpc^2}$]')
    plt.ylabel(r"$\log{n_e} ~[\mathrm{cm}^{-3}]$")
    plt.xlim(-8.5, -6.5)
    plt.ylim(1, 3)
    plt.savefig('figures/ne_plots/ne_sfr_scatter.png')
    plt.show()

    plt.hist2d(sfr_sd, ne, bins=200)
    plt.plot(sfrrange, ne_25, color='r', linestyle='dashed')
    plt.plot(sfrrange, ne_50, color='r')
    plt.plot(sfrrange, ne_75, color='r', linestyle='dashed')
    plt.xlabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/kpc^2}$]')
    plt.ylabel(r"$\log{n_e} ~[\mathrm{cm}^{-3}]$")
    plt.xlim(-8.5, -6.5)
    plt.ylim(1, 3)
    plt.savefig('figures/ne_plots/ne_sfr_2dhist.png')
    plt.show()

    plt.hist2d(sfr_sd, ne, bins=200, norm=mpl.colors.LogNorm())
    plt.plot(sfrrange, ne_25, color='r', linestyle='dashed')
    plt.plot(sfrrange, ne_50, color='r')
    plt.plot(sfrrange, ne_75, color='r', linestyle='dashed')
    plt.xlabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/pc^2}$]')
    plt.ylabel(r"$\log{n_e} ~[\mathrm{cm}^{-3}]$")
    plt.xlim(-8.5, -6.5)
    plt.ylim(1, 3)
    plt.savefig('figures/ne_plots/ne_sfr_2dhist_logscale.png')
    plt.show()

    plt.hist2d(sfr_sd, sfr, bins=150, norm=mpl.colors.LogNorm())
    #plt.plot(sfrrange, ne_25, color='r', linestyle='dashed')
    #plt.plot(sfrrange, ne_50, color='r')
    #plt.plot(sfrrange, ne_75, color='r', linestyle='dashed')
    plt.xlabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/pc^2}$]')
    plt.ylabel(r'SFR [$\log{M_\odot/yr}$]')
    plt.xlim(-8.5, -6.5)
    plt.ylim(-1.5, 1.5)
    #plt.savefig('figures/ne_plots/ne_sfr_2dhist_logscale.png')
    plt.show()


def ne_vs_redshift():
    """
    Plots electron density vs redshift
    :return: none
    """
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]  # bgs length
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]  # bgs length
    redshift = CC.catalog['Z'][BGS_MASK]
    ne, ne_mask = bgs_ne_snr_cut()  # these are both bgs length

    snr_mask = generate_combined_mask(BGS_SFR_MASK, ne_mask)

    mass = mass[snr_mask]
    sfr = sfr[snr_mask]
    ne = ne[snr_mask]
    redshift = redshift[snr_mask]

    valid_mask = np.isfinite(mass) & np.isfinite(sfr) & np.isfinite(ne)
    mass = mass[valid_mask]
    sfr = sfr[valid_mask]
    ne = ne[valid_mask]
    redshift = redshift[valid_mask]

    m, b = np.polyfit(redshift, ne, 1)

    fit_x = np.linspace(0,.5,3)
    fit_y = m * fit_x + b

    plt.hist2d(redshift, ne, bins=100, norm=mpl.colors.LogNorm())
    plt.plot(fit_x, fit_y, color='r')
    plt.text(0.01, 3.6, r'slope = {:.3f}'.format(m), size='large')
    plt.xlabel("z")
    plt.ylabel(r'$\log{n_e}$ [cm$^{-3}$]')
    plt.savefig('figures/ne_plots/ne_redshift.png')
    plt.show()

    m, b = np.polyfit(redshift, mass, 1)

    fit_x = np.linspace(0,.5,3)
    fit_y = m * fit_x + b

    plt.hist2d(redshift, mass, bins=100, norm=mpl.colors.LogNorm())
    #plt.plot(fit_x, fit_y, color='r')
    #plt.text(0.01, 3.6, r'slope = {:.3f}'.format(m), size='large')
    plt.ylim(7,12)
    plt.xlabel("z")
    plt.ylabel(r'$\log{M_\star/M_\odot}$')
    plt.savefig('figures/ne_plots/mass_redshift.png')
    plt.show()


def plot_bpt_diag_ne_color():
    """
    Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)
    Color-codes by n_e
    :return: None
    """
    # potentially change this so instead of a flat snr cut we keep uncertainties
    # and find other ways to deal with it
    snr_lim = SNR_LIM

    # Extracting line fluxes from the catalog.
    # All are BGS length
    nii = CC.catalog['NII_6584_FLUX'][BGS_MASK]
    nii_snr = nii * np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK])
    ha = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    ha_snr = ha * np.sqrt(CC.catalog['HALPHA_FLUX_IVAR'][BGS_MASK])
    oiii = CC.catalog['OIII_5007_FLUX'][BGS_MASK]
    oiii_snr = oiii * np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK])
    hb = CC.catalog['HBETA_FLUX'][BGS_MASK]
    hb_snr = hb * np.sqrt(CC.catalog['HBETA_FLUX_IVAR'][BGS_MASK])
    ne, ne_mask = bgs_ne_snr_cut()

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    # all input masks are BGS length
    zero_mask = generate_combined_mask(nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0)
    full_mask = generate_combined_mask(zero_mask, ne_mask,
                                       nii_snr > snr_lim,
                                       ha_snr > snr_lim,
                                       oiii_snr > snr_lim,
                                       hb_snr > snr_lim)

    nh = np.log10(nii[full_mask] / ha[full_mask])  # x-axis
    oh = np.log10(oiii[full_mask] / hb[full_mask]) # y-axis
    ne = ne[full_mask]

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

    # Calculate where each source is
    #HII_sources = generate_combined_mask()

    # Creating color map for opaque colorbar
    cmap = cm.plasma
    norm = Normalize(vmin=1.5, vmax=2.5)

    f, ax = plt.subplots()
    plt.scatter(nh, oh, marker='.',
                alpha=0.1,
                c=ne, cmap=cmap, norm=norm)
    plt.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='k')
    plt.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r')
    plt.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b')
    plt.text(-1.3, -0.4, f"H II\n{hii_ne_median:.2f}", fontweight='bold')
    plt.text(-.23, -0.75, f"Composite\n{composite_ne_median:.2f}", fontweight='bold')
    plt.text(-1.0, 1.2, f"AGN\n{agn_ne_median:.2f}", fontweight='bold')
    plt.text(0.15, -0.25, f"Shocks\n{shock_ne_median:.2f}", fontweight='bold')
    plt.text(0.005, 1.005, f'total: {sum(full_mask)}, snr $>$ {snr_lim}',
          horizontalalignment='left',
          verticalalignment='bottom',
          transform=ax.transAxes)
    plt.xlim(-1.75, 0.5)
    plt.ylim(-1, 1.5)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, label=r"$n_e$ (cm$^{-3}$)")
    plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
    plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')
    plt.show()


def sfrsd_vs_a_ha():
    """
    This plots the relationship between exctinction in Halpha (A(Ha)) (mag) and star formation rate surface density
    :return: none
    """
    sample = (BGS_MASK) & (CAT_SFR_MASK)
    sfrsd = CC.catalog['SFR_SD'][sample]
    a_ha = CC.catalog['A_HALPHA'][sample]

    plt.hist2d(sfrsd, a_ha, bins=100, norm=mpl.colors.LogNorm(vmax=200))
    plt.xlabel(r'$\Sigma_{SFR}$ [$\log{M_\odot/yr/pc^2}$]')
    plt.ylabel(r'$A(H\alpha)$')
    plt.colorbar(label='count')
    plt.show()


def hydrogen_snr_distributions():
    """
    This explores and plots the relationship between Halpha and Hbeta SNR
    in correlation with the SNR of other emission lines
    :return: none
    """

    snr_lim = SNR_LIM

    oii_1_snr = CC.catalog['OII_3726_FLUX'] * np.sqrt(CC.catalog['OII_3726_FLUX_IVAR']) > snr_lim
    oii_2_snr = CC.catalog['OII_3729_FLUX'] * np.sqrt(CC.catalog['OII_3729_FLUX_IVAR']) > snr_lim
    oii_snr = generate_combined_mask(oii_1_snr, oii_2_snr)
    sii_1_snr = CC.catalog['SII_6716_FLUX'] * np.sqrt(CC.catalog['SII_6716_FLUX_IVAR']) > snr_lim
    sii_2_snr = CC.catalog['SII_6731_FLUX'] * np.sqrt(CC.catalog['SII_6731_FLUX_IVAR']) > snr_lim
    sii_snr = generate_combined_mask(sii_1_snr, sii_2_snr)

    ne_combined_mask = (oii_snr) & (sii_snr) & (BGS_MASK)

    ha_snr = CC.catalog['HALPHA_FLUX'] * np.sqrt(CC.catalog['HALPHA_FLUX_IVAR'])
    hb_snr = CC.catalog['HBETA_FLUX'] * np.sqrt(CC.catalog['HBETA_FLUX_IVAR'])

    plt.scatter(ha_snr[ne_combined_mask], hb_snr[ne_combined_mask])
    plt.xlabel(r"$H\alpha$ SNR")
    plt.ylabel(r"$H\beta$ SNR")
    plt.show()

    ha_combined_mask = generate_combined_mask(ne_combined_mask, ha_snr > snr_lim)

    hb_masked = hb_snr[ha_combined_mask]

    print(sum(hb_masked < 5) / len(hb_masked))

    plt.hist(hb_snr[ha_combined_mask], bins=100)
    plt.xlabel(r"$H\beta$ SNR")
    plt.ylabel("count")
    plt.title(r"$H\beta$ distribution for SNR$>$5 in [OII], [SII], $H\alpha$")
    plt.xlim(0, 75)
    plt.show()

    hb_cut = sorted(hb_snr[ha_combined_mask])
    cum = np.arange(1, len(hb_cut) + 1) / len(hb_cut)

    plt.plot(hb_cut, cum)
    plt.xlabel(r"$H\beta$ SNR")
    plt.ylabel("CDF")
    plt.title(r"$H\beta$ distribution for SNR$>$5 in [OII], [SII], $H\alpha$")
    plt.xlim(0, 10)
    plt.ylim(-0.02, 0.15)
    plt.show()


def sfr_mass_extinction():
    tids = CC.catalog['TARGETID'][BGS_MASK]
    sfr = CC.catalog['SFR_HALPHA'][BGS_MASK]
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]
    extinction = CC.catalog['A_HALPHA'][BGS_MASK]
    halpha_flux = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    halpha_uncert = 1/np.sqrt(CC.catalog['HALPHA_FLUX_IVAR'][BGS_MASK])
    hbeta_flux = CC.catalog['HBETA_FLUX'][BGS_MASK]
    hbeta_uncert = 1 / np.sqrt(CC.catalog['HBETA_FLUX_IVAR'][BGS_MASK])

    mass_sfr_mask = (BGS_SFR_MASK) & (BGS_MASS_MASK)
    #mass_sfr_mask = combined_snr_mask()

    hydrogen_flux_ratio = halpha_flux / hbeta_flux
    hydrogen_flux_ratio_uncert = hydrogen_flux_ratio * np.sqrt((halpha_uncert / halpha_flux)**2 + (hbeta_uncert / hbeta_flux)**2)
    hydrogen_flux_ratio_short = hydrogen_flux_ratio[mass_sfr_mask]
    hydrogen_flux_ratio_uncert_short = hydrogen_flux_ratio_uncert[mass_sfr_mask]

    neg_mask = np.array(hydrogen_flux_ratio_short < 2.86)
    neg_mask_2sig = np.array(hydrogen_flux_ratio_short + 2 * hydrogen_flux_ratio_uncert_short < 2.86)
    print(sum(neg_mask))
    print(sum(neg_mask_2sig))

    plot_2sig_mask = np.array(hydrogen_flux_ratio + 2 * hydrogen_flux_ratio_uncert < 2.86)
    plot_mask = (plot_2sig_mask) & (mass_sfr_mask)

    plt.hist2d(mass[mass_sfr_mask], sfr[mass_sfr_mask], bins=100, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.scatter(mass[plot_mask], sfr[plot_mask], alpha=0.1, color='r', label=r'$ F_{H\alpha}/F_{H\beta} + 2\sigma < 2.86$')
    plt.xlabel(r'$\log(M_\star/M_\odot)$')
    plt.ylabel(r'$\log(M_\odot/yr)$')
    plt.xlim(7.5,12)
    plt.ylim(-2,2)
    plt.legend()

    plt.show()



def run_analysis_scripts():
    global SNR_LIM
    SNR_LIM = 5

    print(sum(BGS_SNR_MASK))
    print(Z50, sum(LO_Z_MASK))
    print(Z90, sum(HI_Z_MASK))

    #sfr_spread_plots()


def main():
    run_analysis_scripts()

if __name__ == '__main__':
    main()