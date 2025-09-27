import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['text.usetex'] = True
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects

import numpy as np

from scipy.stats import binned_statistic_2d

from import_custom_catalog import CC
from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from sample_masks import (BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK,
                          BGS_SFR_MASK, BGS_MASS_MASK,
                          BGS_SNR_MASK, LO_Z_MASK, HI_Z_MASK,
                          Z50, Z90, M50, M90, SFR50, SFR90)
from sample_masks import bgs_ne_snr_cut


def sfr_ms(plot=False):
    """
    Calculates a main sequence 2nd order polynomial fit for the sfr main sequence
    No star formation rate cut
    This only includes SNR cuts for Halpha and Hbeta, plus successful CIGALE fitting.
    This was done to include as many sources as possible in the BGS
    :return: fit parameters: 1st order coefficient, 2nd order coefficient, constant
    """
    snr_mask = generate_combined_mask(BGS_MASK, CAT_SFR_MASK, CAT_MASS_MASK)

    sfr = CC.catalog['SFR_HALPHA'][snr_mask]
    mstar = CC.catalog['MSTAR_CIGALE'][snr_mask]

    specific_sfr = np.log10((10**sfr) / 10**(mstar))

    o1, o2, c = np.polyfit(mstar, sfr, 2)

    x = np.linspace(0,20,100)
    y = o1 * x**2 + o2 * x + c


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
        #plt.plot(x, np.ones(len(x))*sSFR_cut, color='k', linestyle='--', label='sSFR cut')
        plt.xlim(8, 11.5)
        plt.colorbar(label='count')
        #plt.ylim(-.35, .35)
        plt.xlabel(r'$\log{M_\star/M_\odot}$')
        plt.ylabel(r'$\log{SFR / M_\star}$')
        #plt.legend(loc='upper left')
        plt.show()

    return o1, o2, c


def distance_from_ms(mass, sfr, o1, o2, c):
    dist = sfr - (o1 * mass**2 + o2 * mass + c)
    return dist


def calc_color():
    """
    Calculates g-r color for all catalog objects
    :return: float array of g-r color (catalog length)
    """
    mag_g = CC.catalog['ABSMAG01_SDSS_G'] - CC.catalog['KCORR01_SDSS_G']
    mag_r = CC.catalog['ABSMAG01_SDSS_R'] - CC.catalog['KCORR01_SDSS_R']

    gr_col = mag_g - mag_r

    return gr_col