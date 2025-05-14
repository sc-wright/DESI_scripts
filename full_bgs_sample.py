



########################################################################################################################

# NOTE: This file is deprecated. Its functions have been split off into catalog_build.py and analyze_sample.py.
#       Please use those scripts instead.

########################################################################################################################


import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['text.usetex'] = True
#import smplotlib

import seaborn as sns

import numpy as np
from scipy import stats

from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import pandas as pd

from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from spectrum_plot import Spectrum

import time


# import DESI related modules -
import desispec.io                             # Input/Output functions related to DESI spectra
from desispec import coaddition                # Functions related to coadding the spectra

import fitsio

from desiutil.dust import dust_transmission
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras

# DESI targeting masks -
from desitarget.cmx.cmx_targetmask import cmx_mask as cmxmask
from desitarget.sv1.sv1_targetmask import desi_mask as sv1mask
from desitarget.sv2.sv2_targetmask import desi_mask as sv2mask
from desitarget.sv3.sv3_targetmask import desi_mask as sv3mask
from desitarget.targetmask import desi_mask as specialmask


class LSSCatalog:
    def __init__(self):
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.specprod = 'fuji'
        self.specprod_dir = f'{my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.lssCatalogsDir = f'{my_dir}/public/edr/vac/edr/lss/v2.0/LSScats/full'

        # reads in the entire BGS_ANY catalog. Quality cuts are already implemented.
        self.catalog = Table.read(f'{self.lssCatalogsDir}/BGS_ANY_full.dat.fits')

        # this all the non-vetoed targetids in the bgs. fully unique
        # bgs_tids = self.catalog['TARGETID'][self.catalog['ZWARN'] == 0]


    def add_col_to_table(self, colstr, data):
        """
        This takes an array of any type (data) and adds it to the self.catalog table (in memory) with the column name 'colstr'
        The array must be the same length as the other columns in the table

        :param colstr: str: the name of the column to add (CAPS by convention)
        :param data: any array: The data to add as a column. The array must be the same length as all the other columns in the table
        :return: None
        """

        print(f"Adding column {colstr} to table...")

        self.catalog.add_column(Table.Column(data=-1 * np.ones(len(self.catalog)), dtype=float, name=colstr))

        for i, v in enumerate(data):
            self.catalog[colstr][i] = v


class CustomTimer:
    def __init__(self, dlen, calcstr=''):
        self.t = time.time()
        self.lastFullSecElapsed = int(time.time() - self.t)
        self.dataLength = int(dlen)
        if calcstr != '':
            self.calcstr = ' ' + str(calcstr)
        else:
            self.calcstr = calcstr
    def update_time(self, i):
        elapsed = time.time() - self.t
        fullSecElapsed = int(elapsed)
        if fullSecElapsed > self.lastFullSecElapsed:
            self.lastFullSecElapsed = fullSecElapsed
            percent = 100 * (i + 1) / self.dataLength
            totalTime = elapsed / (percent / 100)
            remaining = totalTime - elapsed
            trString = (f"Calculating{self.calcstr}, " + str(int(percent)) + "% complete. approx "
                        + str(int(remaining) // 60) + "m" + str(int(remaining) % 60) + "s remaining...")
            print('\r' + trString, end='', flush=True)


class FSFCatalog:
    def __init__(self):
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.specprod = 'fuji'
        self.specprod_dir = f'{my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.fsfCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2/catalogs'

        self.fsfMeta = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji.fits', hdu=2)

        try:
            self.catalog = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji-processed.fits')
        except FileNotFoundError:
            # if the modified table doesn't exist, do the necessary calculations and create the table.
            # this can take several minutes to run. It should include a progress report for the longest step, but be patient for the rest.
            print("FITS with pre-calculated values not found, generating new file...")
            # The table gets read in during the first step.
            self.catalog = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji.fits', hdu=1)
            self.add_oii_lum_to_table()
            self.add_sii_lum_to_table()
            self.add_primary_to_table()
            self.write_table_to_disk()

    def add_oii_lum_to_table(self):
        """
        This calculates the [OII] luminosity and adds it in a new column to the table that already exists in memory.
        The luminosity column is called "OII_COMBINED_LUMINOSITY" which isn't a good name but it's too much trouble
        to change it now.
        It only needs to be called if fastspec-fuji-processed.fits does not exist.

        :return: None
        """
        # This adds the oii luminosity to the table (in memory)
        # It only needs to be called if fastspec-fuji-processed.fits does not exist

        self.catalog.add_column(Table.Column(data=-1 * np.ones(len(self.catalog)), dtype=float, name=f"OII_COMBINED_LUMINOSITY"))

        oII6Flux = np.array(self.catalog['OII_3726_FLUX'])
        oII9Flux = np.array(self.catalog['OII_3729_FLUX'])
        oIICombinedFlux = oII6Flux + oII9Flux
        redshift = np.array(self.catalog['Z'])
        npix = np.array(self.catalog['OII_3726_NPIX']) + np.array(self.catalog['OII_3729_NPIX'])
        dataLength = len(oIICombinedFlux)

        t = time.time()
        lastFullSecElapsed = int(time.time() - t)

        for i in range(dataLength):
            if npix[i] > 1:
                flux = oIICombinedFlux[i]
                if flux > 0:
                    oIILum = np.log10(get_lum(flux, redshift[i]))
                    self.catalog['OII_COMBINED_LUMINOSITY'][i] = oIILum

            # Displaying progress through the set to check for hanging
            elapsed = time.time() - t
            fullSecElapsed = int(elapsed)
            if fullSecElapsed > lastFullSecElapsed:
                lastFullSecElapsed = fullSecElapsed
                percent = 100 * (i + 1) / dataLength
                totalTime = elapsed / (percent / 100)
                remaining = totalTime - elapsed
                trString = ("Calculating [OII] luminosity, " + str(int(percent)) + "% complete. approx "
                            + str(int(remaining) // 60) + "m" + str(int(remaining) % 60) + "s remaining...")
                print('\r' + trString, end='', flush=True)


    def add_sii_lum_to_table(self):
        """
        This calculates the [SII] luminosity and adds it in a new column to the table that already exists in memory.
        The luminosity column is called "SII_COMBINED_LUMINOSITY" which isn't a good name but it's too much trouble
        to change it now.
        It only needs to be called if fastspec-fuji-processed.fits does not exist.

        :return: None
        """
        # This adds the sii luminosity to the table (in memory)
        # It only needs to be called if fastspec-fuji-processed.fits does not exist

        self.catalog.add_column(Table.Column(data=-1 * np.ones(len(self.catalog)), dtype=float, name=f"SII_COMBINED_LUMINOSITY"))

        sII16Flux = np.array(self.catalog['SII_6716_FLUX'])
        sII31Flux = np.array(self.catalog['SII_6731_FLUX'])
        sIICombinedFlux = sII16Flux + sII31Flux
        redshift = np.array(self.catalog['Z'])
        npix = np.array(self.catalog['SII_6716_NPIX']) + np.array(self.catalog['SII_6731_NPIX'])
        dataLength = len(sIICombinedFlux)

        t = time.time()
        lastFullSecElapsed = int(time.time() - t)

        for i in range(dataLength):
            if npix[i] > 1:
                flux = sIICombinedFlux[i]
                if flux > 0:
                    sIILum = np.log10(get_lum(flux, redshift[i]))
                    self.catalog['SII_COMBINED_LUMINOSITY'][i] = sIILum

            # Displaying progress through the set to check for hanging
            elapsed = time.time() - t
            fullSecElapsed = int(elapsed)
            if fullSecElapsed > lastFullSecElapsed:
                lastFullSecElapsed = fullSecElapsed
                percent = 100 * (i + 1) / dataLength
                totalTime = elapsed / (percent / 100)
                remaining = totalTime - elapsed
                trString = ("Calculating [SII] luminosity, " + str(int(percent)) + "% complete. approx "
                            + str(int(remaining) // 60) + "m" + str(int(remaining) % 60) + "s remaining...")
                print('\r' + trString, end='', flush=True)

    def add_primary_to_table(self):

        zpix = ZPIXCatalog()

        program_dict = zpix.generate_primary_dict()

        is_primary = np.zeros(len(self.catalog['TARGETID']), dtype=bool)

        for i, (tid, program) in enumerate(zip(self.catalog['TARGETID'], self.catalog['PROGRAM'])):
            target_primary_program = program_dict[tid]
            if program == target_primary_program:
                is_primary[i] = True

        self.add_col_to_table('ISPRIMARY', is_primary)

    def add_col_to_table(self, colstr, data):
        """
        This takes an array of any type (data) and adds it to the self.catalog table (in memory) with the column name 'colstr'
        The array must be the same length as the other columns in the table

        :param colstr: str: the name of the column to add (CAPS by convention)
        :param data: any array: The data to add as a column. The array must be the same length as all the other columns in the table
        :return: None
        """

        print(f"Adding column {colstr} to table...")

        self.catalog.add_column(Table.Column(data=-1 * np.ones(len(self.catalog)), dtype=bool, name=colstr))

        for i, v in enumerate(data):
            self.catalog[colstr][i] = v

    def write_table_to_disk(self):
        """
        This takes the current self.catalog table and writes it to a new fits file so calculations don't need to be remade
        The new fits file has just the one data table, the metadata (in hdu 2 of the original fits file) is lost.
        The metadata can still be read from the original fits file, as the rows are still matched.
        If the processed file is missing or needs to be remade, this is the last method to run to do that.

        :return: None
        """
        # This writes the current version of the table as it exists in memory to the disk
        # as "fastspec-fuji-processed.fits"
        # It only needs to be called if the processed table doesn't exist yet

        ogname = self.fsfCatalogsDir + "/fastspec-fuji-processed.fits"
        bakname = ogname + ".bak"

        print("Writing table...")
        try:
            print("renaming old table...")
            os.rename(ogname, bakname)
            print("writing new table to disk...")
            self.catalog.write(self.fsfCatalogsDir + "/fastspec-fuji-processed.fits")
        except:
            print("old table not found, writing table...")
            self.catalog.write(self.fsfCatalogsDir + "/fastspec-fuji-processed.fits")

        print("...done.")


class DR9Catalog:
    def __init__(self):
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.specprod = 'fuji'
        self.specprod_dir = f'{my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.dr9CatalogsDir = f'{my_dir}/public/edr/vac/edr/lsdr9-photometry/{self.specprod}/v2.1/observed-targets'

        self.catalog = Table.read(f'{self.dr9CatalogsDir}/targetphot-sv3-{self.specprod}.fits')

    def add_redshift_to_table(self):

        print("adding redshifts to DR9 table...")

        targetids = self.catalog['TARGETID']
        redshifts = np.ones(len(targetids)) * -1

        redshift_dict = res = {FSF.catalog['TARGETID'][i]: FSF.catalog['Z'][i] for i in range(len(FSF.catalog['TARGETID']))}

        t = CustomTimer(len(targetids))
        for i in range(len(targetids)):
            t.update_time(i)
            #print(targetids[i])
            try:
                redshifts[i] = redshift_dict[targetids[i]]
            except KeyError:
                pass

        self.add_col_to_table("Z", redshifts, float)
        self.write_table_to_disk()

        """
        #This method was going to take too long. Dictionary method is used instead.
        t = CustomTimer(len(targetids), calcstr="z retrieval")
        for i, tid in enumerate(targetids):
            t.update_time(i)
            try:
                redshift = FSF.catalog['Z'][np.logical_and(FSF.catalog['TARGETID'] == tid, FSF.catalog['ISPRIMARY'])][0]
                redshifts[i] = redshift
            except IndexError:
                print(f'{tid} does not have a known redshift.')

        z_in_cat_order = np.ones(len(self.catalog['TARGETID'])) * -1

        t = CustomTimer(len(targetids), calcstr='z write')
        for i, (z, tid) in enumerate(zip(redshifts, targetids)):
            t.update.time(i)
            z_in_cat_order[self.catalog['TARGETID'] == tid] = z
        """



        #self.add_col_to_table("Z", z_in_cat_order)
        #self.write_table_to_disk()



    def add_col_to_table(self, colstr, data, type):
        """
        This takes an array of any type (data) and adds it to the self.catalog table (in memory) with the column name 'colstr'
        The array must be the same length as the other columns in the table

        :param colstr: str: the name of the column to add (CAPS by convention)
        :param data: any array: The data to add as a column. The array must be the same length as all the other columns in the table
        :return: None
        """

        print(f"Adding column {colstr} to table...")

        self.catalog.add_column(Table.Column(data=-1 * np.ones(len(self.catalog)), dtype=type, name=colstr))

        for i, v in enumerate(data):
            self.catalog[colstr][i] = v


    def write_table_to_disk(self):
        """
        This takes the current self.catalog table and writes it to a new fits file so calculations don't need to be remade
        The new fits file has just the one data table, the metadata (in hdu 2 of the original fits file) is lost.
        The metadata can still be read from the original fits file, as the rows are still matched.
        If the processed file is missing or needs to be remade, this is the last method to run to do that.

        :return: None
        """
        # This writes the current version of the table as it exists in memory to the disk
        # as "fastspec-fuji-processed.fits"
        # It only needs to be called if the processed table doesn't exist yet

        ogname = self.dr9CatalogsDir + f"/targetphot-sv3-{self.specprod}.fits"
        bakname = ogname + ".bak"

        print("Writing table...")
        try:
            print("renaming old table...")
            os.rename(ogname, bakname)
            print(f"writing new table to disk at {self.dr9CatalogsDir}" + f"/targetphot-sv3-{self.specprod}.fits" + "...")
            self.catalog.write(self.dr9CatalogsDir + f"/targetphot-sv3-{self.specprod}.fits")
        except:
            print(f"old table not found, writing table to {self.dr9CatalogsDir}" + f"/targetphot-sv3-{self.specprod}.fits" + "...")
            self.catalog.write(self.dr9CatalogsDir + f"targetphot-sv3-{self.specprod}.fits")


class ZPIXCatalog:
    def __init__(self):
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.specprod = 'fuji'
        self.specprod_dir = f'{my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.catalog = Table.read(f'{self.specprod_dir}/zcatalog/zall-pix-{self.specprod}.fits', hdu="ZCATALOG")

    def generate_primary_dict(self):
        is_primary = self.catalog['ZCAT_PRIMARY']
        tids = self.catalog['TARGETID'][is_primary]
        programs = self.catalog['PROGRAM'][is_primary]

        primary_obs_dict = {tids[i]: programs[i] for i in range(len(tids))}
        return primary_obs_dict


class CombinedCatalog:
    def __init__(self):
        tids = FSF.catalog['TARGETID']
        redshifts = FSF.catalog['Z']
        oii_rat = FSF.catalog['OII_DOUBLET_RATIO']
        sii_rat = FSF.catalog['SII_DOUBLET_RATIO']
        primary = FSF.catalog['ISPRIMARY']

        cigale_results = read_cigale_results()

        cigale_mstar = {cigale_results['id'][i]: cigale_results['bayes.stellar.m_star'][i] for i in range(len(cigale_results['id']))}
        cigale_sfr = {cigale_results['id'][i]: cigale_results['bayes.sfh.sfr'][i] for i in range(len(cigale_results['id']))}

        mstar = []
        sfr = []

        for i, tid in enumerate(tids):
            try:
                m = cigale_mstar[tid]
                s = cigale_sfr[tid]
                mstar.append(m)
                sfr.append(s)
            except KeyError:
                mstar.append(0)
                sfr.append(0)
        if len(mstar) != len(sfr):
            print("something went wrong when combining the lists")
            return 0

        data = {'TARGETID': list(tids),
                'Z': list(redshifts),
                'OII_DOUBLET_RATIO': list(oii_rat),
                'SII_DOUBLET_RATIO': list(sii_rat),
                'MSTAR': list(mstar),
                'SFR': list(sfr),
                'ISPRIMARY': list(primary)}
        self.catalog = pd.DataFrame(data)


def generate_oii_snr_mask():
    """
    This makes a new mask of BGS size that includes the sample of sources with SNR > 2 for the OII amplitude.
    :return:
    """
    oii_26_snr_mask = (FSF.catalog['OII_3726_AMP'] * np.sqrt(FSF.catalog['OII_3726_AMP_IVAR'])) > SNR_LIM
    oii_29_snr_mask = (FSF.catalog['OII_3729_AMP'] * np.sqrt(FSF.catalog['OII_3729_AMP_IVAR'])) > SNR_LIM

    oii_snr_mask = generate_combined_mask(FSF_BGS_MASK, oii_26_snr_mask, oii_29_snr_mask)

    return oii_snr_mask


def generate_bgs_mask():
    lss = LSSCatalog()
    fsf = FSFCatalog()
    dr9 = DR9Catalog()
    zpix = ZPIXCatalog()

    # this all the non-vetoed targetids in the bgs. fully unique confirmed
    bgs_tids = lss.catalog['TARGETID'][lss.catalog['ZWARN'] == 0]

    # This is fully unique as well - we have selected primary observations in SV3 only
    fsf_bgs_mask = np.isin(fsf.catalog['TARGETID'], bgs_tids)
    fsf_bgs_mask = generate_combined_mask(fsf_bgs_mask, fsf.catalog['Z'] <= .4, fsf.catalog['ISPRIMARY'], fsf.catalog['SURVEY'] == 'sv3')

    dr9_bgs_mask = np.isin(dr9.catalog['TARGETID'], fsf.catalog['TARGETID'][fsf_bgs_mask])
    dr9_bgs_mask = generate_combined_mask(dr9_bgs_mask, dr9.catalog['MASKBITS'] == 0)

    return lss, fsf, dr9, fsf_bgs_mask, dr9_bgs_mask


def generate_sii_snr_mask():
    """
    This makes a new mask of BGS size that includes the sample of sources with SNR > 2 for the SII amplitude.
    :return:
    """
    sii_16_snr_mask = (FSF.catalog['SII_6716_AMP'] * np.sqrt(FSF.catalog['SII_6716_AMP_IVAR'])) > SNR_LIM
    sii_31_snr_mask = (FSF.catalog['SII_6731_AMP'] * np.sqrt(FSF.catalog['SII_6731_AMP_IVAR'])) > SNR_LIM

    sii_snr_mask = generate_combined_mask(FSF_BGS_MASK, sii_16_snr_mask, sii_31_snr_mask)

    return sii_snr_mask


def plot_loii_vs_redshift():

    high_snr_mask = generate_oii_snr_mask()

    loii = FSF.catalog['OII_COMBINED_LUMINOSITY'][FSF_BGS_MASK]
    hi_snr_loii = FSF.catalog['OII_COMBINED_LUMINOSITY'][high_snr_mask]
    z = FSF.catalog['Z'][FSF_BGS_MASK]
    hi_snr_z = FSF.catalog['Z'][high_snr_mask]

    fig = plt.figure(figsize=(5, 4))
    gs = GridSpec(4, 3)
    ax_main = plt.subplot(gs[:3, :3])
    ax_xDist = plt.subplot(gs[3, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    sp = ax_main.scatter(z, loii, marker='.', alpha=0.2, label=f"All")# ({sum(FSF_BGS_MASK)})")
    ax_main.scatter(hi_snr_z, hi_snr_loii, marker='.', alpha=0.2, label=rf"SNR $\geq$ 3")# ({sum(high_snr_mask)})")

    ax_main.set(xlabel=r"Redshift", ylabel=r"$L_{[OII]}$ [erg s$^{-1}$]", ylim=(35.1, 42.8))
    ax_main.legend(loc=4, bbox_to_anchor=(1, .1))

    ax_xDist.hist(z, bins=100, orientation='vertical', align='mid')#, alpha=0.3)
    ax_xDist.hist(hi_snr_z, bins=100, orientation='vertical', align='mid')
    ax_xDist.set(xlabel="Redshift")

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.savefig('figures/loii_vs_redshift.png', dpi=800)
    plt.show()


def plot_ne_vs_stellar_mass(mass_range=False):
    """
    Plots the OII doublet ratio vs the stellar mass with a histogram of the ratios
    Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

    :param mass_range: tuple with 2 floats: low and high end of the mass range to plot, respectively
    :return:
    """

    cigale_results = read_cigale_results()
    cigale_mstar = {cigale_results['id'][i]: cigale_results['bayes.stellar.m_star'][i] for i in range(len(cigale_results['id']))}

    doublet_snr_mask = generate_combined_mask(generate_oii_snr_mask(), FSF.catalog['OII_DOUBLET_RATIO'] > 0.3839, FSF.catalog['OII_DOUBLET_RATIO'] < 1.4558)

    tids = FSF.catalog['TARGETID'][doublet_snr_mask]
    n_oii = np.log10(calc_electron_density_oii(doublet_snr_mask))
    colr = FSF.catalog['OII_COMBINED_LUMINOSITY'][doublet_snr_mask]

    mstar_list = []
    ne_list = []
    colr_list = []

    for i, tid in enumerate(tids):
        try:
            m = cigale_mstar[tid]
            mstar_list.append(m)
            ne_list.append(n_oii[i])
            colr_list.append(colr[i])
        except KeyError:
            pass
    if len(mstar_list) != len(ne_list):
        print("something went wrong when combining the lists")
        return 0

    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(2, 4)
    ax_main = plt.subplot(gs[0:2, :3])
    ax_yDist = plt.subplot(gs[0:2, 3], sharey=ax_main)
    plt.subplots_adjust(wspace=.0, top=0.99)
    axs = [ax_main, ax_yDist]
    sp = ax_main.scatter(np.log10(mstar_list), ne_list, c=colr_list, marker='+', alpha=0.12, vmax=41.5, vmin=39)
    fig.colorbar(sp, ax=axs, label=r"$L_{[OII]}$", location='top')
    ax_main.text(0.005, 1.005, f'total: {len(ne_list)}, snr $>$ {SNR_LIM}',
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax_main.transAxes)
    ax_main.set(xlabel=r"$\log{M_{\star}/M_\odot}$", ylabel="$\log{n_e}$", ylim=(.4, 5.2), xlim=(6, 11.7))

    ax_yDist.hist(ne_list, bins=100, orientation='horizontal', align='mid')
    ax_yDist.set(xlabel='count')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    if mass_range:
        plt.savefig('figures/oii_ratio_vs_mass_loiicol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
    else:
        plt.savefig("figures/ne_vs_mass.png", dpi=800)
    plt.show()


def plot_sii_rat_vs_stellar_mass(mass_range=False):
    """
    Plots the SII doublet ratio vs the stellar mass with a histogram of the ratios
    Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

    :param mass_range: tuple with 2 floats: low and high end of the mass range to plot, respectively
    :return:
    """

    full_mask = generate_sii_snr_mask()

    sii_rat = FSF.catalog['SII_DOUBLET_RATIO'][full_mask]
    stellar_mass = FSF.catalog['LOGMSTAR'][full_mask]
    colr = FSF.catalog['SII_COMBINED_LUMINOSITY'][full_mask]

    if mass_range:
        plot_mask = generate_combined_mask([stellar_mass >= mass_range[0], stellar_mass < mass_range[1]])
    else:
        plot_mask = np.ones(len(stellar_mass), dtype=bool)

    fig = plt.figure(figsize=(5, 6))
    gs = GridSpec(2, 4)
    ax_main = plt.subplot(gs[0:2, :3])
    ax_yDist = plt.subplot(gs[0:2, 3], sharey=ax_main)
    plt.subplots_adjust(wspace=.0, top=0.99)
    axs = [ax_main, ax_yDist]
    sp = ax_main.scatter(stellar_mass[plot_mask], sii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3, vmax=41.5, vmin=39)
    fig.colorbar(sp, ax=axs, label=r"$L_{[SII]} \mathrm{\ [erg\ s}^{-1} \mathrm{]}$", location='top')
    ax_main.text(0.005, 1.005, f'total: {sum(plot_mask)}, snr $>$ {SNR_LIM}',
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax_main.transAxes)
    ax_main.set(xlabel=r"$\log{M_{\star}/M_\odot}$", ylabel="$\lambda 6731 / \lambda 6716$", ylim=(0, 2))

    ax_yDist.hist(sii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
    ax_yDist.set(xlabel='count')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    if mass_range:
        plt.savefig('figures/sii_ratio_vs_mass_lsiicol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
    else:
        plt.savefig("figures/sii_ratio_vs_mass_lsiicol.png", dpi=800)
    plt.show()

def plot_oii_rat_vs_stellar_mass(mass_range=False):
    """
    Plots the SII doublet ratio vs the stellar mass with a histogram of the ratios
    Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

    :param mass_range: tuple with 2 floats: low and high end of the mass range to plot, respectively
    :return:
    """

    full_mask = generate_oii_snr_mask()

    sii_rat = FSF.catalog['OII_DOUBLET_RATIO'][full_mask]
    stellar_mass = FSF.catalog['LOGMSTAR'][full_mask]
    colr = FSF.catalog['OII_COMBINED_LUMINOSITY'][full_mask]

    if mass_range:
        plot_mask = generate_combined_mask([stellar_mass >= mass_range[0], stellar_mass < mass_range[1]])
    else:
        plot_mask = np.ones(len(stellar_mass), dtype=bool)

    fig = plt.figure(figsize=(5, 6))
    gs = GridSpec(2, 4)
    ax_main = plt.subplot(gs[0:2, :3])
    ax_yDist = plt.subplot(gs[0:2, 3], sharey=ax_main)
    plt.subplots_adjust(wspace=.0, top=0.99)
    axs = [ax_main, ax_yDist]
    sp = ax_main.scatter(stellar_mass[plot_mask], sii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3, vmax=41.5, vmin=39)
    fig.colorbar(sp, ax=axs, label=r"$L_{[OII]} \mathrm{\ [erg\ s}^{-1} \mathrm{]}$", location='top')
    ax_main.text(0.005, 1.005, f'total: {sum(plot_mask)}, snr $>$ {SNR_LIM}',
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax_main.transAxes)
    ax_main.set(xlabel=r"$\log{M_{\star}/M_\odot}$", ylabel="$\lambda 3726 / \lambda 3729$", ylim=(0, 2))

    ax_yDist.hist(sii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
    ax_yDist.set(xlabel='count')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    if mass_range:
        plt.savefig('figures/oii_ratio_vs_mass_lsiicol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
    else:
        plt.savefig("figures/oii_ratio_vs_mass_lsiicol.png", dpi=800)
    plt.show()


def luminosity_functions():

    # this is two independent masks
    oii_26_snr_mask = (FSF.catalog['OII_3726_AMP'] * np.sqrt(FSF.catalog['OII_3726_AMP_IVAR'])) > SNR_LIM
    oii_29_snr_mask = (FSF.catalog['OII_3729_AMP'] * np.sqrt(FSF.catalog['OII_3729_AMP_IVAR'])) > SNR_LIM

    # this takes the luminosity where the snr of at least one is above the limit

    oii_snr_mask = oii_26_snr_mask | oii_29_snr_mask
    oii_full_mask = generate_combined_mask(oii_snr_mask, FSF_BGS_MASK)

    fig, (axs) = plt.subplots(2, 2)
    fig.suptitle("[OII] Luminosity Functions for BGS galaxies")

    oii_mask = generate_combined_mask(oii_full_mask, FSF.catalog['Z'] >= 0, FSF.catalog['Z'] < 0 + .1)
    loii = FSF.catalog['OII_COMBINED_LUMINOSITY'][oii_mask]
    axs[0, 0].hist(loii, bins=100)
    axs[0, 0].set(xlim=(37, 42.5), ylim=(1, 2000), yscale='log')
    axs[0, 0].text(.03, .97, r"$0.0 \leq Z < 0.1$", horizontalalignment='left', verticalalignment='top',
                   transform=axs[0, 0].transAxes)
    axs[0, 0].get_xaxis().set_ticklabels([])

    oii_mask = generate_combined_mask(oii_full_mask, FSF.catalog['Z'] >= .1, FSF.catalog['Z'] < .1 + .1)
    loii = FSF.catalog['OII_COMBINED_LUMINOSITY'][oii_mask]
    axs[0, 1].hist(loii, bins=100)
    axs[0, 1].set(xlim=(37, 42.5), ylim=(1, 2000), yscale='log')
    axs[0, 1].text(.03, .97, r"$0.1 \leq Z < 0.2$", horizontalalignment='left', verticalalignment='top',
                   transform=axs[0, 1].transAxes)
    axs[0,1].get_yaxis().set_ticklabels([])
    axs[0,1].get_xaxis().set_ticklabels([])

    oii_mask = generate_combined_mask(oii_full_mask, FSF.catalog['Z'] >= .2, FSF.catalog['Z'] < .2 + .1)
    loii = FSF.catalog['OII_COMBINED_LUMINOSITY'][oii_mask]
    axs[1, 0].hist(loii, bins=100)
    axs[1, 0].set(xlim=(37, 42.5), ylim=(1, 2000), yscale='log')
    axs[1, 0].text(.03, .97, r"$0.2 \leq Z < 0.3$", horizontalalignment='left', verticalalignment='top',
                   transform=axs[1, 0].transAxes)

    oii_mask = generate_combined_mask(oii_full_mask, FSF.catalog['Z'] >= .3, FSF.catalog['Z'] < .3 + .1)
    loii = FSF.catalog['OII_COMBINED_LUMINOSITY'][oii_mask]
    axs[1, 1].hist(loii, bins=100)
    axs[1, 1].set(xlim=(37, 42.5), ylim=(1, 2000), yscale='log')
    axs[1, 1].text(.03, .97, r"$0.3 \leq Z < 0.4$", horizontalalignment='left', verticalalignment='top',
                   transform=axs[1, 1].transAxes)
    axs[1,1].get_yaxis().set_ticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)

    fig.supxlabel(r"$L_{[OII]}$ [erg s$^{-1}$]")

    plt.savefig("figures/oii luminosity functions bgs.png", dpi=600)


def k_lambda_2001(wavelength):
    # From
    # Wavelength is in angstroms - convert to microns
    wl = wavelength * 1e-4

    if wl <= 2.2000 and wl > .6300:
        k = 1.17 * (-1.1857 + (1.040 / wl)) + 1.78
    elif wl >= .1200:
        k = 1.17 * (-2.156 + (1.509 / wl) - (0.198 / wl**2) + (0.011 / wl**3)) + 1.78
    else:
        print(wavelength, "outside wavelength range")
        return 0

    return k


def k_lambda_2000(wavelength):
    # From
    # Wavelength is in angstroms - convert to microns
    wl = wavelength * 1e-4

    if wl <= 2.2000 and wl > .6300:
        k = 2.659 * (-1.1857 + (1.040 / wl)) + 4.05
    elif wl >= .1200:
        k = 3.659 * (-2.156 + (1.509 / wl) - (0.198 / wl**2) + (0.011 / wl**3)) + 4.05
    else:
        print(wavelength, "outside wavelength range")
        return 0

    return k


def plot_k():
    wl_range = np.linspace(1200, 22000, 300)
    k_2000 = np.zeros(300)
    k_2001 = np.zeros(300)
    for i, wl in enumerate(wl_range):
        k_2000[i] = k_lambda_2000(wl)
        k_2001[i] = k_lambda_2001(wl)

    plt.plot(wl_range, k_2000, label=r'$k_{2000}$')
    plt.plot(wl_range, k_2001, label=r'$k_{2001}$')
    plt.xlabel(r'$\lambda$ (\AA)')
    plt.ylabel(r'$k(\lambda)$')
    plt.legend()
    plt.show()


def calc_SFR_Halpha():
    halpha_snr_mask = FSF.catalog['HALPHA_AMP'] * np.sqrt(FSF.catalog['HALPHA_AMP_IVAR']) > SNR_LIM
    hbeta_snr_mask = FSF.catalog['HBETA_AMP'] * np.sqrt(FSF.catalog['HBETA_AMP_IVAR']) > SNR_LIM
    full_mask = generate_combined_mask(FSF_BGS_MASK, halpha_snr_mask, hbeta_snr_mask)

    E_beta_alpha = 2.5 * np.log10(2.86 / (FSF.catalog['HALPHA_FLUX'][full_mask] / FSF.catalog['HBETA_FLUX'][full_mask]))

    EBV = E_beta_alpha / (k_lambda_2000(6563) - k_lambda_2000(4861))

    H_alpha_flux_int = FSF.catalog['HALPHA_FLUX'][full_mask] * 10 ** (0.4 * k_lambda_2000(6563) * EBV)

    redshifts = FSF.catalog['Z'][full_mask]
    tids = FSF.catalog['TARGETID'][full_mask]

    unextincted_halpha_lum = np.ones(len(H_alpha_flux_int)) * -1
    ctime = CustomTimer(len(H_alpha_flux_int), "Halpha luminosity")
    for i, (flux, z) in enumerate(zip(H_alpha_flux_int, redshifts)):
        unextincted_halpha_lum[i] = get_lum(flux, z)
        ctime.update_time(i)

    # using the table from Kennicutt 2012
    halpha_sfr_log = np.log10(unextincted_halpha_lum) - 41.27
    # using the method from Kennicutt 1998 (as listed in https://arxiv.org/pdf/2312.00300 sect 3.3)
    halpha_sfr = unextincted_halpha_lum * 7.9E-42

    halpha_sfr_table = pd.DataFrame(list(zip(tids, halpha_sfr_log)), columns=['TARGETID', 'LOG_SFR'])
    halpha_sfr_table = halpha_sfr_table.drop_duplicates()
    return halpha_sfr_table


def calc_sfr_corrected_halpha():
    # This version uses the APERCORR_R from the FSF catalog

    halpha_snr_mask = FSF.catalog['HALPHA_AMP'] * np.sqrt(FSF.catalog['HALPHA_AMP_IVAR']) > SNR_LIM
    hbeta_snr_mask = FSF.catalog['HBETA_AMP'] * np.sqrt(FSF.catalog['HBETA_AMP_IVAR']) > SNR_LIM
    full_mask = generate_combined_mask(FSF_BGS_MASK, halpha_snr_mask, hbeta_snr_mask)

    redshifts = FSF.catalog['Z'][full_mask]
    tids = FSF.catalog['TARGETID'][full_mask]

    E_beta_alpha = 2.5 * np.log10(
        2.86 / (FSF.catalog['HALPHA_FLUX'][full_mask] / FSF.catalog['HBETA_FLUX'][full_mask]))

    EBV = E_beta_alpha / (k_lambda_2000(6563) - k_lambda_2000(4861))

    ha_flux = FSF.catalog['HALPHA_FLUX'][full_mask] * 10 ** (0.4 * k_lambda_2000(6563) * EBV)

    # perform aperture correction
    #ha_flux = ha_flux * FSF.catalog['APERCORR_R'][full_mask]

    ha_lum = get_lum(ha_flux, redshifts)

    ha_sfr_log = np.log10(ha_lum) - 41.27

    halpha_sfr_table = pd.DataFrame(list(zip(tids, ha_sfr_log)), columns=['TARGETID', 'LOG_SFR'])

    return halpha_sfr_table


def calc_sfr_corrected_halpha_by_hand():

    halpha_snr_mask = FSF.catalog['HALPHA_AMP'] * np.sqrt(FSF.catalog['HALPHA_AMP_IVAR']) > SNR_LIM
    hbeta_snr_mask = FSF.catalog['HBETA_AMP'] * np.sqrt(FSF.catalog['HBETA_AMP_IVAR']) > SNR_LIM
    full_mask = generate_combined_mask(FSF_BGS_MASK, halpha_snr_mask, hbeta_snr_mask)

    ha_dict = aperture_correct_ha()

    redshifts = FSF.catalog['Z'][full_mask]
    tids = FSF.catalog['TARGETID'][full_mask]

    success_tid = []
    halpha_lum = []

    for i, (tid, z) in enumerate(zip(tids, redshifts)):
        try:
            flux = ha_dict[tid]
            halpha_lum.append(get_lum(flux, z))
            success_tid.append(tid)
        except KeyError:
            print(f"Ha failure for {tid}")

    success_tid = np.array(success_tid)
    halpha_lum = np.array(halpha_lum)

    # using the table from Kennicutt 2012
    halpha_sfr_log = np.log10(halpha_lum) - 41.27
    # using the method from Kennicutt 1998 (as listed in https://arxiv.org/pdf/2312.00300 sect 3.3)
    halpha_sfr = halpha_lum * 7.9E-42


    halpha_sfr_table = pd.DataFrame(list(zip(success_tid, halpha_sfr_log)), columns=['TARGETID', 'LOG_SFR'])
    #halpha_sfr_table = halpha_sfr_table.drop_duplicates()

    return halpha_sfr_table



def calc_mstar_WISE(f, z):
    # using methods from Jarrett et al. 2023 (https://arxiv.org/abs/2301.05952)

    target_flux = f * u.nanomaggy

    zero_point_star_equiv = u.zero_point_flux(3631.1 * u.Jy)
    mag = u.Magnitude(target_flux.to(u.AB, zero_point_star_equiv))

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    D_l = cosmo.luminosity_distance(z).value  # Mpc

    abs_mag = mag.value - 5 * np.log10(D_l) - 25

    lw1 = 10 ** (-0.4 * (abs_mag - 3.24))

    A0 = -12.62185
    A1 = 5.00155
    A2 = -0.43857
    A3 = 0.01593

    log_mstar = A0 + A1 * np.log10(lw1) + A2 * np.log10(lw1)**2 + A3 * np.log10(lw1)**3

    return log_mstar


def calc_mstar_WISE_color(w1, w2, z):
    """
    :param w1: flux in W1 band in nanomaggies
    :param w2: flux in W2 band in nanomaggies
    :param z: redshift
    :return:
    """

    # these fluxes are in nmgy
    w1_flux = w1
    w2_flux = w2

    # Set flat cosmology
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    # Calculate luminosity distance in Mpc
    D_l_unit = cosmo.luminosity_distance(z)
    D_l = D_l_unit.value  # Mpc
    #print(D_l_unit, D_l)
    # Calculate magnitudes from the fluxes
    # fluxes are in nanomaggies

    # conversion to magnitudes from https://www.legacysurvey.org/dr9/description/#photometry
    # then convert to vega mag
    # from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
    w1_mag_AB = -2.5 * np.log10(w1_flux) + 22.5
    w1_mag_vega = -2.5 * np.log10(w1_flux) + 22.5 - 2.699
    w2_mag_vega = -2.5 * np.log10(w2_flux) + 22.5 - 3.339

    # color is difference between  magnitudes
    col = w1_mag_vega - w2_mag_vega

    # Calculate absolute magnitudes using luminosity distance
    # convert D_l to pc
    abs_mag_w1_AB = w1_mag_AB - 5 * np.log10(D_l * 1e6 / 10)# - 2.5*np.log10(1.2) # last term is just for debugging
    abs_mag_w1_vega = abs_mag_w1_AB - 2.699

    # calculate luminosity using absolute magnitude
    # vega mag of sun is 3.24
    # AB mag of sun is 5.92
    log_lw1 = -0.4 * (abs_mag_w1_vega  - 3.24)

    A0 = -0.376
    A1 = -1.053

    # multiply m/l by l to get mass
    log_mass = A0 + (A1 * col) + log_lw1

    return log_mass


def all_WISE_mstars():

    tids = DR9.catalog['TARGETID'][DR9_BGS_MASK]
    fw1 = DR9.catalog['FLUX_W1'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W1'][DR9_BGS_MASK]
    fw2 = DR9.catalog['FLUX_W2'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W2'][DR9_BGS_MASK]
    redshifts = DR9.catalog['Z'][DR9_BGS_MASK]
    mstar_dict = {}

    for i, (tid, flux1, flux2, z) in enumerate(zip(tids, fw1, fw2, redshifts)):
        # There are some sources without good redshift fits, with z=-1. The BGS_MASK should remove these,
        # but this is a catch in case one gets through. It should never trigger.
        if z < 0:
            print("redshift error")
            return 1
        # The DR9 catalog contains duplicate entries for many of the target ids. I'm not sure why.
        # The fluxes are all identical, so it should be ok to skip entries if they are duplicated.
        if tid in mstar_dict.keys():
            pass
        else:
            mstar_dict[tid] = calc_mstar_WISE_color(flux1, flux2, z)
    return mstar_dict


def nmgy_to_mjy(flux):
    return flux * (10 ** (-(48.6 + 22.5)/2.5)) * (10 ** 23) * 10 ** 3


def generate_cigale_input_table_separate():

    cigale_dir = os.path.expanduser('~') + '/Documents/school/research/cigale'

    fsf_tids = FSF.catalog['TARGETID'][FSF_BGS_MASK]
    fsf_redshifts = FSF.catalog['Z'][FSF_BGS_MASK]
    dr9_tids = DR9.catalog['TARGETID'][DR9_BGS_MASK]
    dr9_release = DR9.catalog['RELEASE'][DR9_BGS_MASK]

    filters = ['FLUX_G', 'FLUX_IVAR_G', 'FLUX_R', 'FLUX_IVAR_R', 'FLUX_Z', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_IVAR_W1', 'FLUX_W2', 'FLUX_IVAR_W2', 'FLUX_W3', 'FLUX_IVAR_W3', 'FLUX_W4', 'FLUX_IVAR_W4']

    g_ls = DR9.catalog['FLUX_G'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_G'][DR9_BGS_MASK]
    r_ls = DR9.catalog['FLUX_R'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_R'][DR9_BGS_MASK]
    z_ls = DR9.catalog['FLUX_Z'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_Z'][DR9_BGS_MASK]
    w1_ls = DR9.catalog['FLUX_W1'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W1'][DR9_BGS_MASK]
    w2_ls = DR9.catalog['FLUX_W2'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W2'][DR9_BGS_MASK]
    w3_ls = DR9.catalog['FLUX_W3'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W3'][DR9_BGS_MASK]
    w4_ls = DR9.catalog['FLUX_W4'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W4'][DR9_BGS_MASK]

    g_err_ls = DR9.catalog['FLUX_IVAR_G'][DR9_BGS_MASK]
    r_err_ls = DR9.catalog['FLUX_IVAR_R'][DR9_BGS_MASK]
    z_err_ls = DR9.catalog['FLUX_IVAR_Z'][DR9_BGS_MASK]
    w1_err_ls = DR9.catalog['FLUX_IVAR_W1'][DR9_BGS_MASK]
    w2_err_ls = DR9.catalog['FLUX_IVAR_W2'][DR9_BGS_MASK]
    w3_err_ls = DR9.catalog['FLUX_IVAR_W3'][DR9_BGS_MASK]
    w4_err_ls = DR9.catalog['FLUX_IVAR_W4'][DR9_BGS_MASK]

    emm_dict = {}

    for i, (tid, release, g, r, z, w1, w2, w3, w4, g_err, r_err, z_err, w1_err, w2_err, w3_err, w4_err) in enumerate(zip(dr9_tids, dr9_release, g_ls, r_ls, z_ls, w1_ls, w2_ls, w3_ls, w4_ls, g_err_ls, r_err_ls, z_err_ls, w1_err_ls, w2_err_ls, w3_err_ls, w4_err_ls)):
        # Duplicates are identical so this is fine
        lis = [release, g, g_err, r, r_err, z, z_err, w1, w1_err, w2, w2_err, w3, w3_err, w4, w4_err]
        emm_dict.update({tid: lis})


    # DECam section - release 9010
    print("generating list for release 9010...")

    tids_list = []
    dr9_g_filter = []
    dr9_g_err = []
    dr9_r_filter = []
    dr9_r_err = []
    dr9_z_filter = []
    dr9_z_err = []
    dr9_w1_filter = []
    dr9_w1_err = []
    dr9_w2_filter = []
    dr9_w2_err = []
    dr9_w3_filter = []
    dr9_w3_err = []
    dr9_w4_filter = []
    dr9_w4_err = []


    with open(f'{cigale_dir}/9010/cigale_input_data_9010.dat', 'w') as file:
        file.write(f'# id\tredshift\tDECam_g\tDECam_g_err\tDECam_r\tDECam_r_err\tDECam_z\tDECam_z_err\tWISE1\tWISE1_err\tWISE2\tWISE2_err\tWISE3\tWISE3_err\tWISE4\tWISE4_err\n')

        for i, tid in enumerate(fsf_tids):
            try:
                write_str = ""
                if emm_dict[tid][0] in (9010, 9012):
                    write_str += f'{tid}\t{fsf_redshifts[i]}'

                    for colr in [1, 3, 5]:
                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr+1])
                        if err / val < 0.10:
                            err = val * 0.10
                        write_str += f'\t{val}\t{err}'

                    for colr in [7, 9, 11, 13]:
                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr+1])
                        if err / val < 0.13:
                            err = val * 0.13
                        write_str += f'\t{val}\t{err}'

                    write_str += '\n'

                    file.write(write_str)
            except KeyError:
                pass

    # bok section - release 9011
    print("generating list for release 9011...")

    tids_list = []
    dr9_g_filter = []
    dr9_g_err = []
    dr9_r_filter = []
    dr9_r_err = []
    dr9_w1_filter = []
    dr9_w1_err = []
    dr9_w2_filter = []
    dr9_w2_err = []
    dr9_w3_filter = []
    dr9_w3_err = []
    dr9_w4_filter = []
    dr9_w4_err = []

    with open(f'{cigale_dir}/9011/cigale_input_data_9011.dat', 'w') as file:
        file.write(f'# id\tredshift\tbok_g\tbok_g_err\tbok_r\tbok_r_err\tWISE1\tWISE1_err\tWISE2\tWISE2_err\tWISE3\tWISE3_err\tWISE4\tWISE4_err\n')

        for i, tid in enumerate(fsf_tids):
            try:
                write_str = ""
                if emm_dict[tid][0] in [9011]:
                    write_str += f'{tid}\t{fsf_redshifts[i]}'

                    for colr in [1, 3]:
                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr + 1])
                        if err / val < 0.10:
                            err = val * 0.10
                        write_str += f'\t{val}\t{err}'

                    for colr in [7, 9, 11, 13]:
                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr + 1])
                        if err / val < 0.13:
                            err = val * 0.13
                        write_str += f'\t{val}\t{err}'

                    write_str += '\n'

                    file.write(write_str)
            except KeyError:
                pass


def generate_cigale_input_table(custom_tids = None):

    cigale_dir = os.path.expanduser('~') + '/Documents/school/research/cigale'

    fsf_tids = FSF.catalog['TARGETID'][FSF_BGS_MASK]
    fsf_redshifts = FSF.catalog['Z'][FSF_BGS_MASK]

    if custom_tids is None:
        pass
    else:
        custom_select = np.isin(fsf_tids, custom_tids)
        fsf_tids = fsf_tids[custom_select]
        fsf_redshifts = fsf_redshifts[custom_select]

    print(fsf_tids[:5])

    dr9_tids = DR9.catalog['TARGETID'][DR9_BGS_MASK]
    dr9_release = DR9.catalog['RELEASE'][DR9_BGS_MASK]

    if custom_tids is None:
        folder = '_full_sky'
        file_name = 'cigale_input_data_full.dat'
    else:
        folder = 'custom_target_list'
        file_name = 'cigale_input_data_custom.dat'


    filters = ['FLUX_G', 'FLUX_IVAR_G', 'FLUX_R', 'FLUX_IVAR_R', 'FLUX_Z', 'FLUX_IVAR_Z', 'FLUX_W1', 'FLUX_IVAR_W1', 'FLUX_W2', 'FLUX_IVAR_W2', 'FLUX_W3', 'FLUX_IVAR_W3', 'FLUX_W4', 'FLUX_IVAR_W4']

    g_ls = DR9.catalog['FLUX_G'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_G'][DR9_BGS_MASK]
    r_ls = DR9.catalog['FLUX_R'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_R'][DR9_BGS_MASK]
    z_ls = DR9.catalog['FLUX_Z'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_Z'][DR9_BGS_MASK]
    w1_ls = DR9.catalog['FLUX_W1'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W1'][DR9_BGS_MASK]
    w2_ls = DR9.catalog['FLUX_W2'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W2'][DR9_BGS_MASK]
    w3_ls = DR9.catalog['FLUX_W3'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W3'][DR9_BGS_MASK]
    w4_ls = DR9.catalog['FLUX_W4'][DR9_BGS_MASK] / DR9.catalog['MW_TRANSMISSION_W4'][DR9_BGS_MASK]

    g_err_ls = 1/np.sqrt(DR9.catalog['FLUX_IVAR_G'][DR9_BGS_MASK])
    r_err_ls = 1/np.sqrt(DR9.catalog['FLUX_IVAR_R'][DR9_BGS_MASK])
    z_err_ls = 1/np.sqrt(DR9.catalog['FLUX_IVAR_Z'][DR9_BGS_MASK])
    w1_err_ls = 1/np.sqrt(DR9.catalog['FLUX_IVAR_W1'][DR9_BGS_MASK])
    w2_err_ls = 1/np.sqrt(DR9.catalog['FLUX_IVAR_W2'][DR9_BGS_MASK])
    w3_err_ls = 1/np.sqrt(DR9.catalog['FLUX_IVAR_W3'][DR9_BGS_MASK])
    w4_err_ls = 1/np.sqrt(DR9.catalog['FLUX_IVAR_W4'][DR9_BGS_MASK])

    emm_dict = {}

    for tid, release, g, r, z, w1, w2, w3, w4, g_err, r_err, z_err, w1_err, w2_err, w3_err, w4_err in zip(dr9_tids, dr9_release, g_ls, r_ls, z_ls, w1_ls, w2_ls, w3_ls, w4_ls, g_err_ls, r_err_ls, z_err_ls, w1_err_ls, w2_err_ls, w3_err_ls, w4_err_ls):
        # Duplicates are identical so this is fine
        lis = [release, g, g_err, r, r_err, z, z_err, w1, w1_err, w2, w2_err, w3, w3_err, w4, w4_err]
        emm_dict.update({tid: lis})


    with open(f'{cigale_dir}/{folder}/{file_name}', 'w') as file:
        file.write(
            f'# id\tredshift\tDECam_g\tDECam_g_err\tDECam_r\tDECam_r_err\tDECam_z\tDECam_z_err\tbok_g\tbok_g_err\tbok_r\tbok_r_err\tWISE1\tWISE1_err\tWISE2\tWISE2_err\tWISE3\tWISE3_err\tWISE4\tWISE4_err\n')

        for i, tid in enumerate(fsf_tids):

            try:
                write_str = ""

                # DECam section - release 9010

                if emm_dict[tid][0] in (9010, 9012):
                    write_str += f'{tid}\t{fsf_redshifts[i]}'

                    for colr in [1, 3, 5]:

                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr+1])
                        if err / abs(val) < 0.10:
                            err = abs(val) * 0.10
                        write_str += f'\t{val}\t{err}'

                    write_str += f'\tnan\tnan\tnan\tnan'

                    for colr in [7, 9, 11, 13]:
                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr+1])
                        if err / abs(val) < 0.13:
                            err = abs(val) * 0.13
                        write_str += f'\t{val}\t{err}'

                    write_str += '\n'

                    file.write(write_str)

                # bok section - release 9011

                elif emm_dict[tid][0] in [9011]:
                    write_str += f'{tid}\t{fsf_redshifts[i]}'

                    write_str += f'\tnan\tnan\tnan\tnan\tnan\tnan'

                    for colr in [1, 3]:
                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr + 1])
                        if err / abs(val) < 0.10:
                            err = abs(val) * 0.10
                        write_str += f'\t{val}\t{err}'

                    for colr in [7, 9, 11, 13]:
                        val = nmgy_to_mjy(emm_dict[tid][colr])
                        err = nmgy_to_mjy(emm_dict[tid][colr + 1])
                        if err / abs(val) < 0.13:
                            err = abs(val) * 0.13
                        write_str += f'\t{val}\t{err}'

                    write_str += '\n'

                    file.write(write_str)
            except KeyError:
                pass


def calc_electron_density_oii(mask):

    a = 0.3771
    b = 2468
    c = 638.4

    R = FSF.catalog['OII_DOUBLET_RATIO'][mask]

    n = (c*R - a*b) / (a - R)

    return n


def calc_electron_density_sii(mask):

    a = 0.4315
    b = 2107
    c = 627.1

    R = FSF.catalog['SII_DOUBLET_RATIO'][mask]

    n = (c * R - a * b) / (a - R)

    return n


def read_cigale_results(folder='_full_sky'):
    cigale_dir = os.path.expanduser('~') + '/Documents/school/research/cigale'

    #results_1 = pd.read_table(f"{cigale_dir}/9010/out/results.txt", header=0, sep='\s+')
    #results_2 = pd.read_table(f"{cigale_dir}/9011/out/results.txt", header=0, sep='\s+')

    #cigale_results = pd.concat([results_1, results_2], ignore_index=True, sort=False)

    cigale_results = pd.read_table(f"{cigale_dir}/{folder}/out/results.txt", header=0, sep='\s+')

    return cigale_results


def analyze_chi2_cigale():
    results = read_cigale_results(folder='_full_sky')

    plt.hist(results['best.reduced_chi_square'], bins=100, range=(0, 30))
    plt.xlabel(r"$\chi^2_\nu$")
    plt.title("All")
    plt.show()
    """
    cigale_dir = os.path.expanduser('~') + '/Documents/school/research/cigale'
    result_10 = pd.read_table(f"{cigale_dir}/9010/out/results.txt", header=0, sep='\s+')
    result_11 = pd.read_table(f"{cigale_dir}/9011/out/results.txt", header=0, sep='\s+')
    plt.hist(result_10['best.reduced_chi_square'], bins=100, range=(0, 30))
    plt.xlabel(r"$\chi^2_\nu$")
    plt.title("Southern Hemisphere (9010)")
    plt.show()
    plt.hist(result_11['best.reduced_chi_square'], bins=100, range=(0, 30))
    plt.xlabel(r"$\chi^2_\nu$")
    plt.title("Northern Hemisphere (9011)")
    plt.show()
    """


def compare_cigale_sfr_vs_catalog():
    cigale_results = read_cigale_results()

    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'

    cigale_sfr_results = {cigale_results['id'][i]: cigale_results['bayes.sfh.sfr'][i] for i in range(len(cigale_results['id']))}
    #catalog_results = Table.read(f'{my_dir}/public/edr/vac/edr/stellar-mass-emline/v1.0/edr_galaxy_stellarmass_lineinfo_v1.0.fits')
    cigale_sfr = []
    catalog_sfr = []

    for i, tid in enumerate(FSF.catalog['TARGETID']):
        try:
            cm = np.log10(cigale_sfr_results[tid])
            wm = FSF.catalog['SFR'][i]
            if not np.isnan(cm) and not np.isnan(wm):
                cigale_sfr.append(cm)
                catalog_sfr.append(np.log10(wm))
            if len(cigale_sfr) != len(catalog_sfr):
                print(i) #something's gone wrong, this is the item that caused the problem
                break
        except KeyError:
            pass

    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(catalog_sfr, cigale_sfr, marker='+', alpha=0.05)
    #ax_main.plot(np.linspace(7,13, 300), np.linspace(7, 13, 300), color='r')
    ax_main.set(xlabel=r"SFR from catalog", ylabel=r"SFR from CIGALE")#, xlim=(7,13), ylim=(7,13))

    ax_yDist.hist(cigale_sfr, bins=200, orientation='horizontal', align='mid')
    #ax_xDist.hist(catalog_sfr, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()


def compare_sfr():
    cigale_results = read_cigale_results()

    os.path.expanduser('~') + '/Documents/school/research/cigale'

    cigale_sfr_results = {cigale_results['id'][i]: cigale_results['bayes.sfh.sfr'][i] for i in range(len(cigale_results['id']))}
    halpha_sfr_table = calc_sfr_corrected_halpha_by_hand()
    aperture_dict = calc_aperture()
    cigale_sfr = []
    halpha_sfr = []
    aperture = []

    for i, tid in enumerate(halpha_sfr_table['TARGETID']):
        try:
            cm = np.log10(cigale_sfr_results[tid])
            wm = halpha_sfr_table['LOG_SFR'][i]
            ap = aperture_dict[tid]
            if np.isfinite(cm) and np.isfinite(wm):  # don't include any nans or /0's
                cigale_sfr.append(cm)
                halpha_sfr.append(wm)
                aperture.append(ap)
            if len(cigale_sfr) != len(halpha_sfr):
                print(i) #something's gone wrong, this is the item that caused the problem
                break
        except KeyError:
            pass

    cigale_sfr = np.array(cigale_sfr)
    halpha_sfr = np.array(halpha_sfr)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(halpha_sfr, bins=200)
    ax.text(0.01, 0.99, f'mean: {np.average(halpha_sfr)}\nmedian: {np.median(halpha_sfr)}\nstdev: {np.std(halpha_sfr)}',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax.set(xlabel=r"SFR from $H\alpha$ ($\log{m_\star/m_\odot}$) (with my aperture correction including color)", xlim=(-8,2.5))
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(cigale_sfr, bins=200)
    ax.text(0.01, 0.99, f'mean: {np.average(cigale_sfr)}\nmedian: {np.median(cigale_sfr)}\nstdev: {np.std(cigale_sfr)}',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax.set(xlabel=r"SFR from CIGALE ($\log{m_\star/m_\odot}$)", xlim=(-8,2.5))
    plt.show()

    #print(cigale_sfr)
    #print(f"SFR Avg: {np.average(cigale_sfr)}, stdev: {np.std(cigale_sfr)}")
    #print(halpha_sfr)
    #print(f"SFR Avg: {np.average(halpha_sfr)}, stdev: {np.std(halpha_sfr)}")

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(halpha_sfr, halpha_sfr - cigale_sfr, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(-10,10, 100), np.zeros(100), color='r')
    ax_main.set(xlabel=r"SFR$_{H\alpha}$ [log(M$_\odot$/yr)]", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(-9,4), ylim=(-10,4))
    #ax_main.set(xlabel=r"$F_{r,model}/F_{r,aperture}$", ylabel=r"SFR$_{H\alpha}$ - SFR$_{CIGALE}$ [log(M$_\odot$/yr)]", xlim=(0,20), ylim=(-10,5))

    ax_yDist.hist(halpha_sfr - cigale_sfr, bins=200, orientation='horizontal', align='mid')
    ax_xDist.hist(halpha_sfr, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()




def compare_stellar_mass():
    cigale_results = read_cigale_results()

    cigale_mstar_results = {cigale_results['id'][i]: cigale_results['bayes.stellar.m_star'][i] for i in range(len(cigale_results['id']))}
    wise_mstar_dictionary = all_WISE_mstars()
    cigale_mstar = []
    wise_mstar = []

    for i, tid in enumerate(wise_mstar_dictionary.keys()):
        try:
            cm = np.log10(cigale_mstar_results[tid])
            wm = wise_mstar_dictionary[tid]
            cigale_mstar.append(cm)
            wise_mstar.append(wm)
            if len(cigale_mstar) != len(wise_mstar):
                print(i) #something's gone wrong, this is the item that caused the problem
                break
        except KeyError:
            pass

    cigale_mstar = np.array(cigale_mstar)
    wise_mstar = np.array(wise_mstar)

    nan_mask = generate_combined_mask(np.invert(np.isnan(cigale_mstar)), np.invert(np.isnan(wise_mstar)))

    cigale_mstar = cigale_mstar[nan_mask]
    wise_mstar = wise_mstar[nan_mask]

    difference = wise_mstar - cigale_mstar

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


def compare_stellar_mass_vs_catalog():
    cigale_results = read_cigale_results()

    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'

    cigale_mstar_results = {cigale_results['id'][i]: cigale_results['bayes.stellar.m_star'][i] for i in range(len(cigale_results['id']))}
    catalog_results = Table.read(f'{my_dir}/public/edr/vac/edr/stellar-mass-emline/v1.0/edr_galaxy_stellarmass_lineinfo_v1.0.fits')
    cigale_mstar = []
    catalog_mstar = []

    for i, tid in enumerate(catalog_results['TARGETID']):
        try:
            cm = np.log10(cigale_mstar_results[tid])
            wm = catalog_results['SED_MASS'][i]
            cigale_mstar.append(cm)
            catalog_mstar.append(np.log10(wm))
            if len(cigale_mstar) != len(catalog_mstar):
                print(i) #something's gone wrong, this is the item that caused the problem
                break
        except KeyError:
            pass

    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(catalog_mstar, cigale_mstar, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(7,13, 300), np.linspace(7, 13, 300), color='r')
    ax_main.set(xlabel=r"$m_\star$ from catalog", ylabel=r"$m_\star$ from CIGALE", xlim=(7,13), ylim=(7,13))

    ax_yDist.hist(cigale_mstar, bins=200, orientation='horizontal', align='mid')
    ax_xDist.hist(catalog_mstar, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()


def plot_bpt_diag(write_to_file=False, loii_range=None):
    """
    Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)

    Cuts:
    -SNR > 3 for all 4 lines

    :return: None
    """

    snr_lim = SNR_LIM

    print("making bpt diagram...")

    nii = FSF.catalog['NII_6584_FLUX'][FSF_BGS_MASK]
    nii_snr = nii/(np.sqrt(1/FSF.catalog['NII_6584_FLUX_IVAR'][FSF_BGS_MASK]))
    ha = FSF.catalog['HALPHA_FLUX'][FSF_BGS_MASK]
    ha_snr = ha/(np.sqrt(1/FSF.catalog['HALPHA_FLUX_IVAR'][FSF_BGS_MASK]))
    oiii = FSF.catalog['OIII_5007_FLUX'][FSF_BGS_MASK]
    oiii_snr = oiii/(np.sqrt(1/FSF.catalog['OIII_5007_FLUX_IVAR'][FSF_BGS_MASK]))
    hb = FSF.catalog['HBETA_FLUX'][FSF_BGS_MASK]
    hb_snr = hb/(np.sqrt(1/FSF.catalog['HBETA_FLUX_IVAR'][FSF_BGS_MASK]))

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    zero_mask = generate_combined_mask(nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0)

    full_mask = generate_combined_mask(zero_mask, nii_snr > snr_lim, ha_snr > snr_lim, oiii_snr > snr_lim, hb_snr > snr_lim, FSF.catalog['OII_SUMMED_SNR'][FSF_BGS_MASK] > snr_lim)


    # Getting the highest and lowest OII luminosity for the high SNR (>3) sources to use as the limits for the color bar. Otherwise its all basically just one color
    oii_lum = FSF.catalog['OII_COMBINED_LUMINOSITY'][FSF_BGS_MASK]

    if loii_range:
        full_mask = generate_combined_mask(full_mask, oii_lum >= loii_range[0], oii_lum < loii_range[1])

    x_for_line_1 = np.log10(np.logspace(-5,.049,300))
    hii_agn_line = 0.61/(x_for_line_1 - 0.05) + 1.3

    x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
    composite_line_2 = 0.61/(x_for_line_2 - 0.47) + 1.19

    x_for_line_3 = np.linspace(-.13,2,100)
    agn_line_3 = 2.144507*x_for_line_3 + 0.465028

    if write_to_file:
        tids = FSF.catalog['TARGETID'][FSF_BGS_MASK]
        #this goes through all the tids and writes them to a file if they are in the "agn" region
        with open('possible_agn_tids.txt', 'w') as f:
            for i in range(len(tids[full_mask])):
                x = np.log10(nii[full_mask][i]/ha[full_mask][i])
                y = np.log10(oiii[full_mask][i]/hb[full_mask][i])
                if y > 0.61/(x - 0.47) + 1.19 and y > 2.144507*x + 0.465028:
                    f.write(f"{tids[full_mask][i]}\n")

    f, ax = plt.subplots()
    plt.scatter(np.log10(nii[full_mask]/ha[full_mask]), np.log10(oiii[full_mask]/hb[full_mask]), marker='.', alpha=0.3, c=oii_lum[full_mask], vmax=41.5, vmin=39)
    plt.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='k')
    plt.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r')
    plt.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b')
    plt.text(-1.2, -0.9, "H II", fontweight='bold')
    plt.text(-.21, -1.1, "Composite", fontweight='bold')
    plt.text(-1.0, 1.5, "AGN", fontweight='bold')
    plt.text(0.45, -0.8, "Shocks", fontweight='bold')
    plt.text(0.005, 1.005, f'total: {sum(full_mask)}, snr $>$ {snr_lim}',
          horizontalalignment='left',
          verticalalignment='bottom',
          transform=ax.transAxes)
    plt.xlim(-2, 1)
    plt.ylim(-1.5, 2)
    plt.colorbar(label=r"$\log{L_{[OII]}}$ [erg s$^{-1}$]")
    plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
    plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')
    if loii_range:
        plt.savefig(f'figures/bpt_bgs_sv3_loii=({loii_range[0], loii_range[1]}).png',dpi=800)
    else:
        plt.savefig('figures/bpt_bgs_sv3.png',dpi=800)
    plt.show()


def plot_bpt_hist(loii_range=None):
    """
    Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)

    Cuts:
    -SNR > 3 for all 4 lines

    :return: None
    """

    snr_lim = SNR_LIM

    print("making bpt histogram...")

    nii = FSF.catalog['NII_6584_FLUX'][FSF_BGS_MASK]
    nii_snr = nii/(np.sqrt(1/FSF.catalog['NII_6584_FLUX_IVAR'][FSF_BGS_MASK]))
    ha = FSF.catalog['HALPHA_FLUX'][FSF_BGS_MASK]
    ha_snr = ha/(np.sqrt(1/FSF.catalog['HALPHA_FLUX_IVAR'][FSF_BGS_MASK]))
    oiii = FSF.catalog['OIII_5007_FLUX'][FSF_BGS_MASK]
    oiii_snr = oiii/(np.sqrt(1/FSF.catalog['OIII_5007_FLUX_IVAR'][FSF_BGS_MASK]))
    hb = FSF.catalog['HBETA_FLUX'][FSF_BGS_MASK]
    hb_snr = hb/(np.sqrt(1/FSF.catalog['HBETA_FLUX_IVAR'][FSF_BGS_MASK]))

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    zero_mask = generate_combined_mask(nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0)

    full_mask = generate_combined_mask(zero_mask, nii_snr > snr_lim, ha_snr > snr_lim, oiii_snr > snr_lim, hb_snr > snr_lim, FSF.catalog['OII_SUMMED_SNR'][FSF_BGS_MASK] > snr_lim)


    # Getting the highest and lowest OII luminosity for the high SNR (>3) sources to use as the limits for the color bar. Otherwise its all basically just one color
    oii_lum = FSF.catalog['OII_COMBINED_LUMINOSITY_LOG'][FSF_BGS_MASK]
    #hisnr_oii_lum = oii_lum[FSF.catalog['OII_SUMMED_SNR'][FSF_BGS_MASK] > 3]

    if loii_range:
        full_mask = generate_combined_mask(full_mask, oii_lum >= loii_range[0], oii_lum < loii_range[1])


    x_for_line_1 = np.log10(np.logspace(-5,.049,300))
    hii_agn_line = 0.61/(x_for_line_1 - 0.05) + 1.3

    x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
    composite_line_2 = 0.61/(x_for_line_2 - 0.47) + 1.19

    x_for_line_3 = np.linspace(-.13,2,100)
    agn_line_3 = 2.144507*x_for_line_3 + 0.465028

    f, ax = plt.subplots()
    plt.hist2d(np.log10(nii[full_mask]/ha[full_mask]), np.log10(oiii[full_mask]/hb[full_mask]), bins=100, range=((-2, 1), (-1.5, 2)), norm=mpl.colors.LogNorm())
    plt.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='w')
    plt.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r')
    plt.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b')
    plt.text(-1.2, -0.9, "H II", fontweight='bold', color='w')
    plt.text(-.21, -1.1, "Composite", fontweight='bold', color='w')
    plt.text(-1.0, 1.5, "AGN", fontweight='bold', color='w')
    plt.text(0.45, -0.8, "Shocks", fontweight='bold', color='w')
    plt.text(0.005, 1.005, f'total: {sum(full_mask)}, snr $>$ {snr_lim}',
          horizontalalignment='left',
          verticalalignment='bottom',
          transform=ax.transAxes)

    plt.xlim(-2, 1)
    plt.ylim(-1.5, 2)
    plt.colorbar(label=r"count")
    plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
    plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')
    if loii_range:
        plt.savefig(f'figures/bpt_hist_bgs_sv3_loii=({loii_range[0], loii_range[1]}).png',dpi=800)
    else:
        plt.savefig('figures/bpt_hist_bgs_sv3.png',dpi=800)
    plt.show()


def compare_electron_density():

    doublet_snr_mask = generate_combined_mask(generate_oii_snr_mask(), generate_sii_snr_mask(), FSF.catalog['OII_DOUBLET_RATIO'] > 0.3839, FSF.catalog['OII_DOUBLET_RATIO'] < 1.4558, FSF.catalog['SII_DOUBLET_RATIO'] > 0.4375, FSF.catalog['SII_DOUBLET_RATIO'] < 1.4484)
    print(sum(doublet_snr_mask))

    n_oii = np.log10(calc_electron_density_oii(doublet_snr_mask))
    n_sii = np.log10(calc_electron_density_sii(doublet_snr_mask))

    median_n_oii = np.median(n_oii)
    mean_n_oii = np.average(n_oii)
    stdev_n_oii = np.std(n_oii)

    median_n_sii = np.median(n_sii)
    mean_n_sii = np.average(n_sii)
    stdev_n_sii = np.std(n_sii)

    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(n_oii, n_sii, marker='+', alpha=0.1)
    ax_main.set(xlabel=r"$\log{n_e}$ from [OII] (cm$^{-3}$)", ylabel=r"$\log{n_e}$ from [SII] (cm$^{-3}$)", ylim=(1.8, 4.6), xlim=(1.8, 4.6))
    ax_main.text(.01, .01, fr"SNR $\geq$ {SNR_LIM}", verticalalignment='bottom', transform=ax_main.transAxes)

    ax_xDist.hist(n_oii, bins=200, orientation='vertical', align='mid')
    ax_xDist.text(.015, .97, f"N: {len(n_oii)}" + "\n" + r"Median: {:.2f}".format(median_n_oii) + "\n" + r"$\sigma$: {:.2f}".format(stdev_n_oii), verticalalignment='top',
                  transform=ax_xDist.transAxes)
    ax_yDist.hist(n_sii, bins=200, orientation='horizontal', align='mid')
    ax_yDist.text(.03, .985, f"N: {len(n_sii)}" + "\n" + r"Median: {:.2f}".format(median_n_sii) + "\n" + r"$\sigma$: {:.2f}".format(stdev_n_sii), verticalalignment='top',
                  transform=ax_yDist.transAxes)

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.savefig(f'figures/electron_density_comparison_{SNR_LIM}.png', dpi=600)
    plt.show()


def hist_electron_density():

    doublet_snr_mask = generate_combined_mask(generate_oii_snr_mask(), generate_sii_snr_mask(),
                                              FSF.catalog['OII_DOUBLET_RATIO'] > 0.3839,
                                              FSF.catalog['OII_DOUBLET_RATIO'] < 1.4558,
                                              FSF.catalog['SII_DOUBLET_RATIO'] > 0.4375,
                                              FSF.catalog['SII_DOUBLET_RATIO'] < 1.4484)

    n_oii = calc_electron_density_oii(doublet_snr_mask)
    n_sii = calc_electron_density_sii(doublet_snr_mask)

    plt.subplot(1, 2, 1)
    plt.hist(n_oii[n_oii < 4000], bins=60, label='N([OII])')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(n_sii[n_sii < 4000], bins=60, label='N([SII])')
    plt.legend()
    plt.title(f"SNR>{SNR_LIM}")
    plt.show()


def ne_vs_mass():

    cigale_results = read_cigale_results()
    cigale_mstar = {cigale_results['id'][i]: cigale_results['bayes.stellar.m_star'][i] for i in range(len(cigale_results['id']))}

    doublet_snr_mask = generate_combined_mask(generate_oii_snr_mask(), FSF.catalog['OII_DOUBLET_RATIO'] > 0.3839, FSF.catalog['OII_DOUBLET_RATIO'] < 1.4558)

    tids = FSF.catalog['TARGETID'][doublet_snr_mask]
    n_oii = np.log10(calc_electron_density_oii(doublet_snr_mask))

    mstar_list = []
    ne_list = []

    for i, tid in enumerate(tids):
        try:
            m = cigale_mstar[tid]
            mstar_list.append(m)
            ne_list.append(n_oii[i])
        except KeyError:
            pass
    if len(mstar_list) != len(ne_list):
        print("something went wrong when combining the lists")
        return 0

    plt.scatter(np.log10(mstar_list), ne_list, marker='+', alpha=0.2)
    plt.xlim(6, 11.9)
    plt.ylim(0.2, 5.4)
    plt.xlabel(r"$\log{M_{\star}/M_\odot}$")
    plt.ylabel(r"$\log{n_e}$ from [OII] ratio")
    plt.show()


def sfr_vs_mstar():

    mass = np.log10(CC.catalog['MSTAR'][FSF_BGS_MASK])
    sfr = np.log10(CC.catalog['SFR'][FSF_BGS_MASK])

    print(len(mass), len(sfr))
    print(len(np.isfinite(mass)), len(np.isfinite(sfr)))

    fin = generate_combined_mask(np.isfinite(mass), np.isfinite(sfr))
    mass = mass[fin]
    sfr = sfr[fin]

    print(len(mass), len(sfr))

    plt.hist2d(mass, sfr, bins=100, norm=mpl.colors.LogNorm())
    plt.xlabel(r"$\log{M_\star/M_\odot}$")
    plt.ylabel(r"$\log{M_\odot/ \textrm{yr}}$")
    plt.show()

"""
def calc_gr_color(full_mask=FSF_BGS_MASK):

    # these fluxes are in nmgy
    g_mag = FSF.catalog['ABSMAG10_DECAM_G'][full_mask]
    r_mag = FSF.catalog['ABSMAG10_DECAM_R'][full_mask]

    # Set flat cosmology
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    # Calculate luminosity distance in Mpc
    D_l_unit = cosmo.luminosity_distance(z)
    D_l = D_l_unit.value  # Mpc
    # print(D_l_unit, D_l)
    # Calculate magnitudes from the fluxes
    # fluxes are in nanomaggies

    # conversion to magnitudes from https://www.legacysurvey.org/dr9/description/#photometry
    # then convert to vega mag
    # from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
    w1_mag_AB = -2.5 * np.log10(w1_flux) + 22.5
    w1_mag_vega = -2.5 * np.log10(w1_flux) + 22.5 - 2.699
    w2_mag_vega = -2.5 * np.log10(w2_flux) + 22.5 - 3.339

    # color is difference between  magnitudes
    col = w1_mag_vega - w2_mag_vega
"""

def calc_aperture():

    tids = DR9.catalog['TARGETID']
    r_flux = DR9.catalog['FLUX_R']
    r_flux_aper = DR9.catalog['FIBERFLUX_R']

    aper_ratio = np.divide(r_flux, r_flux_aper)  # Element-wise division
    aper_dict = dict(zip(tids, aper_ratio))

    return aper_dict


def plot_aper_z():

    tids = FSF.catalog['TARGETID'][FSF_BGS_MASK]
    redshifts = FSF.catalog['Z'][FSF_BGS_MASK]
    ap_dict = calc_aperture()
    aper_ratios = [ap_dict[key] for key in tids]

    plt.scatter(redshifts, np.log10(aper_ratios), marker='+', alpha=0.03)
    plt.xlabel("Z")
    plt.ylabel(r"$\log{F_{r, proj}/F_{r, aper}}$")
    plt.savefig('fall_figs/aperture_vs_z.png')
    plt.show()


def aperture_correct_ha():

    # Calculate extinction-corrected H-alpha using H-beta

    halpha_snr_mask = FSF.catalog['HALPHA_AMP'] * np.sqrt(FSF.catalog['HALPHA_AMP_IVAR']) > SNR_LIM
    hbeta_snr_mask = FSF.catalog['HBETA_AMP'] * np.sqrt(FSF.catalog['HBETA_AMP_IVAR']) > SNR_LIM
    full_mask = generate_combined_mask(FSF_BGS_MASK, halpha_snr_mask, hbeta_snr_mask)

    E_beta_alpha = 2.5 * np.log10(
        2.86 / (FSF.catalog['HALPHA_FLUX'][full_mask] / FSF.catalog['HBETA_FLUX'][full_mask]))

    EBV = E_beta_alpha / (k_lambda_2000(6563) - k_lambda_2000(4861))

    H_alpha_flux_int = FSF.catalog['HALPHA_FLUX'][full_mask] * 10 ** (0.4 * k_lambda_2000(6563) * EBV)

    # Pull in the aperture ratio dictonary - this is for the r-band aperture ratio

    tids = FSF.catalog['TARGETID'][full_mask]
    ap_dict = calc_aperture()

    # Now construct the g-r to Ha/r arrays to fit the curve

    # First, pull in fiber fluxes and tractor fluxes

    dr9_tids = DR9.catalog['TARGETID']
    r_fiber_flux = DR9.catalog['FIBERFLUX_R']
    g_fiber_flux = DR9.catalog['FIBERFLUX_G']
    r_model_flux = DR9.catalog['FLUX_R']
    g_model_flux = DR9.catalog['FLUX_G']

    # Convert all to magnitudes and calculate colors, create dictionaries

    r_fiber_mag = -2.5 * np.log10(r_fiber_flux)
    g_fiber_mag = -2.5 * np.log10(g_fiber_flux)
    gr_fiber_color = g_fiber_mag - r_fiber_mag
    fiber_color_dict = dict(zip(dr9_tids, gr_fiber_color))

    r_model_mag = -2.5 * np.log10(r_model_flux)
    g_model_mag = -2.5 * np.log10(g_model_flux)
    gr_model_color = g_model_mag - r_model_mag
    model_color_dict = dict(zip(dr9_tids, gr_model_color))

    # Create a dictionary for the r-band as well

    r_flux_dict = dict(zip(dr9_tids, r_fiber_flux))

    # Match all the colors from DR9 with the Ha data from fastspecfit

    f_ratio = []
    fiber_gr_array = []
    model_gr_array = []
    tid_array = []

    for i, tid in enumerate(tids):
        try:
            rat = r_flux_dict[tid]
            f_ratio.append(H_alpha_flux_int[i]/rat)
            fiber_gr_array.append(fiber_color_dict[tid])
            model_gr_array.append(model_color_dict[tid])
            tid_array.append(tid)
        except KeyError:
            pass

    f_ratio = np.array(f_ratio)
    fiber_gr_array = np.array(fiber_gr_array)
    model_gr_array = np.array(model_gr_array)
    r_rat = []

    # Perform a linear fit to the Ha/r (fiber) vs g-r (fiber) plot

    linear_fit = np.polyfit(fiber_gr_array, np.log10(f_ratio), 1)
    slope = linear_fit[0]
    print(f"slope: {slope}")

    # Pulling from all the different dictionaries, calculate the aperture-corrected H alpha

    F_ha_appcorr_array = []

    for i, tid in enumerate(tid_array):
        try:
            r_ratio = ap_dict[tid]
            r_rat.append(r_ratio)
            F_ha_appcorr = H_alpha_flux_int[i] * r_ratio * 10 ** (slope * (model_gr_array[i] - fiber_gr_array[i]))
            #print(H_alpha_flux_int[i], F_ha_appcorr, r_ratio, slope, model_gr_array[i] - fiber_gr_array[i])

            F_ha_appcorr_array.append(F_ha_appcorr)
        except KeyError:
            # This should never happen - all these dictionaries should be complete by this point.
            print("Something's wrong")
            pass

    #plt.hist(r_rat, bins=10000)
    #plt.xlim(0,25)
    #plt.xlabel(r"$f_{r,tractor}/f_{r,ap}$")
    #plt.show()

    ha_appcorr_dict = dict(zip(tid_array, np.array(F_ha_appcorr_array)))

    return ha_appcorr_dict

    """
    #plt.scatter(gr_array, np.log10(f_ratio), marker='.', alpha=0.01)
    plt.hist2d(gr_array, np.log10(f_ratio), bins=100)
    plt.xlim(0.1, 1.4)
    plt.ylim(-7.5, 5)
    plt.xlabel("g - r (from fiber fluxes)")
    plt.ylabel(r"$\log(F_{H\alpha}/F_{r, fiber})$")
    plt.show()
    """


def plot_halpha_ratio():
    halpha_snr_mask = FSF.catalog['HALPHA_AMP'] * np.sqrt(FSF.catalog['HALPHA_AMP_IVAR']) > SNR_LIM
    hbeta_snr_mask = FSF.catalog['HBETA_AMP'] * np.sqrt(FSF.catalog['HBETA_AMP_IVAR']) > SNR_LIM
    fsf_full_mask = generate_combined_mask(FSF_BGS_MASK, halpha_snr_mask, hbeta_snr_mask)

    E_beta_alpha = 2.5 * np.log10(
        2.86 / (FSF.catalog['HALPHA_FLUX'][fsf_full_mask] / FSF.catalog['HBETA_FLUX'][fsf_full_mask]))

    EBV = E_beta_alpha / (k_lambda_2000(6563) - k_lambda_2000(4861))

    H_alpha_flux_int = FSF.catalog['HALPHA_FLUX'][fsf_full_mask] * 10 ** (0.4 * k_lambda_2000(6563) * EBV)

    #g_mag = FSF.catalog['ABSMAG01_SDSS_G'][full_mask]
    #r_mag = FSF.catalog['ABSMAG01_SDSS_R'][full_mask]
    #gr_color = g_mag - r_mag

    redshifts = FSF.catalog['Z'][fsf_full_mask]
    tids = FSF.catalog['TARGETID'][fsf_full_mask]
    #ap_dict = calc_aperture()
    #aper_ratios = [ap_dict[key] for key in tids]

    dr9_tids = DR9.catalog['TARGETID']
    r_fiber_flux = -2.5 * np.log10(DR9.catalog['FIBERFLUX_R'])
    g_fiber_flux = -2.5 * np.log10(DR9.catalog['FIBERFLUX_G'])

    f_ratio = []
    gr_array =[]

    for i, tid in enumerate(tids):
        try:
            rat = r_flux_dict[tid]
            f_ratio.append(H_alpha_flux_int[i]/rat)
            gr_array.append(gr_color[i])
        except KeyError:
            pass

    f_ratio = np.array(f_ratio)
    gr_array = np.array(gr_array)

    plt.scatter(gr_array, np.log10(f_ratio), marker='.', alpha=0.01)
    plt.xlim(-0.2, 1.5)
    plt.ylim(-7.5, 5)
    plt.xlabel("g - r")
    plt.ylabel(r"$\log{F_{H\alpha, int}/F_{r, fiber}}$")
    plt.show()


def ident_bad_fits():
    results = read_cigale_results()
    bad_fits = results['id'][results['best.reduced_chi_square'] > 10]
    #print(bad_fits)
    #with open('/Users/simonwright/Desktop/bad_fits.txt', 'w') as f:
    #    for i in bad_fits:
    #        f.write(str(i)+'\n')
    generate_cigale_input_table(custom_tids=bad_fits)


def chi_vs_flux():
    results = read_cigale_results()
    tids = results['id']
    chi2 = results['best.reduced_chi_square']

    W4Flux = DR9.catalog['FLUX_W4']
    W3Flux = DR9.catalog['FLUX_W3']
    W2Flux = DR9.catalog['FLUX_W2']
    W1Flux = DR9.catalog['FLUX_W1']
    GFlux = DR9.catalog['FLUX_G']
    RFlux = DR9.catalog['FLUX_R']
    ZFlux = DR9.catalog['FLUX_Z']
    LSTid = DR9.catalog['TARGETID']

    print("making ")

    w4_dict = dict(zip(LSTid, W4Flux))
    w3_dict = dict(zip(LSTid, W3Flux))
    w2_dict = dict(zip(LSTid, W2Flux))
    w1_dict = dict(zip(LSTid, W1Flux))
    g_dict = dict(zip(LSTid, GFlux))
    r_dict = dict(zip(LSTid, RFlux))
    z_dict = dict(zip(LSTid, ZFlux))

    chi_list = []
    flux_list = []

    print("starting loop")
    for i, tid in enumerate(tids):
        try:
            w4_flux = w4_dict[tid]
            w3_flux = w3_dict[tid]
            w2_flux = w2_dict[tid]
            w1_flux = w1_dict[tid]
            g_flux = g_dict[tid]
            r_flux = r_dict[tid]
            z_flux = z_dict[tid]

            chi = chi2[i]

            if chi > 10:
                neg_flag = 0
                for i in [w4_flux, w3_flux, w2_flux, w1_flux, g_flux, r_flux, z_flux]:
                    if i < 0:
                        neg_flag = 1
                if neg_flag == 0:
                    print(f"{tid} has chi squared {chi} with no negative fluxes.")



            #flux_list.append(flux)
            #chi_list.append(chi2[i])
        except KeyError:
            pass


def plot_half_light_radius():
    hl_radius = DR9.catalog['SHAPE_R'][DR9_BGS_MASK]
    plt.hist(hl_radius, bins=100)
    plt.yscale('log')
    plt.xlabel('Half-light radius (")')
    plt.show()


def main():

    #fsf = FSFCatalog()
    #zpix = ZPIXCatalog()
    #primary_program_dict = zpix.generate_primary_dict()
    #fsf.add_primary_to_table(primary_program_dict)

    global FSF_BGS_MASK, DR9_BGS_MASK, LSS, FSF, DR9, CC, SNR_LIM
    SNR_LIM = 3
    # The FSF_BGS_MASK contains the cuts made at the catalog level, z <= .4, and removes failed redshifts
    # There are no SNR cuts in place as of now, as those are made on a per-line basis.
    LSS, FSF, DR9, FSF_BGS_MASK, DR9_BGS_MASK = generate_bgs_mask()
    #CC = CombinedCatalog()

    #analyze_chi2_cigale()

    #ident_bad_fits()

    #plot_aper_z()
    #plot_halpha_ratio()

    #chi_vs_flux()
    #aperture_correct_ha()
    #plot_half_light_radius()
    #plot_bpt_hist()
    #plot_sii_rat_vs_stellar_mass()
    #plot_oii_rat_vs_stellar_mass()
    #compare_electron_density()

    #generate_cigale_input_table(custom_tids=[39627733935853845])
    #generate_cigale_input_table()
    #plot_k()
    #compare_stellar_mass()
    compare_sfr()
    #compare_cigale_sfr_vs_catalog()
    #compare_stellar_mass_vs_catalog()
    #for snr in (3, 5, 7, 10, 15, 20):
    #    SNR_LIM = snr
    #    compare_electron_density()


    #for i in range(0,11):
    #    SNR_LIM = i
    #    hist_electron_density()



if __name__ == '__main__':
    main()
