import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['text.usetex'] = True
#import smplotlib

import seaborn as sns

import numpy as np

from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import pandas as pd

from utility_scripts import get_lum, generate_combined_mask
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

        # reads in the entire BGS_ANY catalog. Quality cuts are already implemented.
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

    # confirming that the fsf tid list is not fully unique
    #print(len(fsf.catalog['TARGETID'][fsf_bgs_mask]), len(np.unique(fsf.catalog['TARGETID'][fsf_bgs_mask])))
    #u, c = np.unique(fsf.catalog['TARGETID'][fsf_bgs_mask], return_counts=True)
    #print(len(u[c > 1]))
    #print(u[c > 1])

    dr9_bgs_mask = np.isin(dr9.catalog['TARGETID'], fsf.catalog['TARGETID'][fsf_bgs_mask])

    #print(sum(fsf_bgs_mask), sum(dr9_bgs_mask))
    #u, c = np.unique(dr9.catalog['TARGETID'][dr9_bgs_mask], return_counts=True)
    #print(len(u[c > 1]))
    #print(u[c > 1])

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

    #low_snr_mask = [a_i and not b_i for a_i, b_i in zip(FSF_BGS_MASK, high_snr_mask)]


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


def plot_oii_rat_vs_stellar_mass(mass_range=False):
    """
    Plots the OII doublet ratio vs the stellar mass with a histogram of the ratios
    Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

    :param mass_range: tuple with 2 floats: low and high end of the mass range to plot, respectively
    :return:
    """

    full_mask = generate_oii_snr_mask()

    oii_rat = FSF.catalog['OII_DOUBLET_RATIO'][full_mask]
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
    sp = ax_main.scatter(stellar_mass[plot_mask], oii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3, vmax=41.5, vmin=39)
    fig.colorbar(sp, ax=axs, label=r"$L_{[OII]}$", location='top')
    ax_main.text(0.005, 1.005, f'total: {sum(plot_mask)}, snr $>$ {SNR_LIM}',
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax_main.transAxes)
    ax_main.set(xlabel=r"$\log{M_{\star}}$", ylabel="$\lambda 3726 / \lambda 3729$", ylim=(0, 2))

    ax_yDist.hist(oii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
    ax_yDist.set(xlabel='count')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    if mass_range:
        plt.savefig('figures/oii_ratio_vs_mass_loiicol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
    else:
        plt.savefig("figures/oii_ratio_vs_mass_loiicol.png", dpi=800)
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
    fig.colorbar(sp, ax=axs, label=r"$L_{[SII]}$", location='top')
    ax_main.text(0.005, 1.005, f'total: {sum(plot_mask)}, snr $>$ {SNR_LIM}',
             horizontalalignment='left',
             verticalalignment='bottom',
             transform=ax_main.transAxes)
    ax_main.set(xlabel=r"$\log{M_{\star}}$", ylabel="$\lambda 6716 / \lambda 6731$", ylim=(0, 2))

    ax_yDist.hist(sii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
    ax_yDist.set(xlabel='count')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    if mass_range:
        plt.savefig('figures/sii_ratio_vs_mass_lsiicol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
    else:
        plt.savefig("figures/sii_ratio_vs_mass_lsiicol.png", dpi=800)
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


def snr_cut_test_plot():
    oii_snr_mask = generate_oii_snr_mask()

    bgs_oii_tids = FSF.catalog['TARGETID'][oii_snr_mask]

    for i in bgs_oii_tids:
        spec = Spectrum(targetid=i)
        spec.check_for_files()
        spec.plot_spectrum(foldstruct="spectra/snr2test/")


def calc_SFR_Halpha_old_WRONG():

    # THIS IS GARBAGE AND WRONG. DO NOT USE.

    # this version cuts on h beta only. Not sure which to use
    hbeta_snr_mask = FSF.catalog['HBETA_AMP'] * np.sqrt(FSF.catalog['HBETA_AMP_IVAR']) > SNR_LIM
    combined_mask = generate_combined_mask(FSF_BGS_MASK, hbeta_snr_mask)

    unextincted_halpha_flux = FSF.catalog['HBETA_FLUX'][combined_mask] * 2.86
    redshifts = FSF.catalog['Z'][combined_mask]

    unextincted_halpha_lum = np.ones(len(unextincted_halpha_flux)) * -1
    ctime = CustomTimer(len(unextincted_halpha_flux), "Halpha luminosity")
    for i, (flux, z) in enumerate(zip(unextincted_halpha_flux, redshifts)):
        unextincted_halpha_lum[i] = get_lum(flux, z)
        ctime.update_time(i)

    # using the table from Kennicutt 2012
    halpha_sfr_log = np.log10(unextincted_halpha_lum) - 41.27
    # using the method from Kennicutt 1998 (as listed in https://arxiv.org/pdf/2312.00300 sect 3.3)
    halpha_sfr = unextincted_halpha_lum * 7.9E-42


    plt.hist(halpha_sfr_log, bins=60)
    plt.xlabel(r"$\log(\dot{M}_\star)$ from H alpha ($M_\odot$/yr)")
    plt.show()
    #this definitely seems wrong...


def k_lambda(wavelength):
    # Wavelength is in angstroms - convert to microns
    wavelength = wavelength * 1e-4

    if wavelength <= 2.2000 and wavelength > .6300:
        k = 1.17 * (-1.1857 + (1.040 / wavelength)) + 1.78
    elif wavelength >= .1200:
        k = 1.17 * (-2.156 + (1.509 / wavelength) - (0.198 / wavelength**2) + (0.011 / wavelength**3)) + 1.78
    else:
        print(wavelength, "outside wavelength range")
        return 0

    return k


def calc_SFR_Halpha():
    halpha_snr_mask = FSF.catalog['HALPHA_AMP'] * np.sqrt(FSF.catalog['HALPHA_AMP_IVAR']) > SNR_LIM
    hbeta_snr_mask = FSF.catalog['HBETA_AMP'] * np.sqrt(FSF.catalog['HBETA_AMP_IVAR']) > SNR_LIM
    full_mask = generate_combined_mask(FSF_BGS_MASK, halpha_snr_mask, hbeta_snr_mask)

    E_beta_alpha = -2.5 * np.log10(2.86 / (FSF.catalog['HALPHA_FLUX'][full_mask] / FSF.catalog['HBETA_FLUX'][full_mask]))

    print(E_beta_alpha)

    EBV = E_beta_alpha / (k_lambda(4861) - k_lambda(6563))

    H_alpha_flux_int = FSF.catalog['HALPHA_FLUX'][full_mask] * 10 ** (0.4 * k_lambda(6563) * EBV)

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

    # these fluxes are in nmgy, in AB system
    w1_flux = w1
    w2_flux = w2

    # Set flat cosmology
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    # Calculate luminosity distance in Mpc
    D_l = cosmo.luminosity_distance(z).value  # Mpc
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

    #print(w1_mag, w2_mag)
    # Calculate absolute magnitudes using luminosity distance
    abs_mag_w1_AB = w1_mag_AB - 5 * np.log10(D_l * 1e6 / 10) # convert D_l to pc
    abs_mag_w1_vega = abs_mag_w1_AB - 2.699


    # calculate luminosity using absolute magnitude
    # vega mag of sun is 3.24
    # AB mag of sun is 5.92
    log_lw1 = -0.4 * (abs_mag_w1_vega - 3.24)

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
    mstar = np.zeros(len(tids))

    for i, (tid, flux1, flux2, z) in enumerate(zip(tids, fw1, fw2, redshifts)):
        if z < 0:
            print("redshift error")
            return 1
        mstar[i] = calc_mstar_WISE_color(flux1, flux2, z)
    wise_mstar = pd.DataFrame(list(zip(tids, mstar)), columns=['TARGETID', 'MSTAR'])
    wise_mstar = wise_mstar.drop_duplicates()
    return wise_mstar


def nmgy_to_mjy(flux):
    return flux * (10 ** (-(48.6 + 22.5)/2.5)) * (10 ** 23) * 10 ** 3


def generate_cigale_input_table():

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
            write_str = ""
            if emm_dict[tid][0] in (9010, 9012):
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


def read_cigale_results():
    cigale_dir = os.path.expanduser('~') + '/Documents/school/research/cigale'

    results_1 = pd.read_table(f"{cigale_dir}/9010/out/results.txt", header=0, sep='\s+')
    results_2 = pd.read_table(f"{cigale_dir}/9011/out/results.txt", header=0, sep='\s+')

    cigale_results = pd.concat([results_1, results_2], ignore_index=True, sort=False)

    return cigale_results


def compare_sfr():
    cigale_results = read_cigale_results()

    os.path.expanduser('~') + '/Documents/school/research/cigale'

    cigale_sfr_results = {cigale_results['id'][i]: cigale_results['bayes.sfh.sfr'][i] for i in range(len(cigale_results['id']))}
    halpha_sfr_table = calc_SFR_Halpha()
    cigale_sfr = []
    halpha_sfr = []

    for i, tid in enumerate(halpha_sfr_table['TARGETID']):
        try:
            cm = np.log10(cigale_sfr_results[tid])
            wm = halpha_sfr_table['LOG_SFR'][i]
            cigale_sfr.append(cm)
            halpha_sfr.append(wm)
            if len(cigale_sfr) != len(halpha_sfr):
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
    sp = ax_main.scatter(halpha_sfr, cigale_sfr, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(-10,10, 100), np.linspace(-10, 10, 100), color='r')
    ax_main.set(xlabel=r"SFR from H$\alpha$ [log(M$_\odot$/yr]", ylabel=r"SFR from CIGALE [log(M$_\odot$/yr]", xlim=(-6,3), ylim=(-6,3))

    ax_yDist.hist(cigale_sfr, bins=200, orientation='horizontal', align='mid')
    ax_xDist.hist(halpha_sfr, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()



def compare_stellar_mass():
    cigale_results = read_cigale_results()

    #os.path.expanduser('~') + '/Documents/school/research/cigale'

    cigale_mstar_results = {cigale_results['id'][i]: cigale_results['bayes.stellar.m_star'][i] for i in range(len(cigale_results['id']))}
    wise_mstar_table = all_WISE_mstars()
    cigale_mstar = []
    wise_mstar = []

    for i, tid in enumerate(wise_mstar_table['TARGETID']):
        try:
            cm = np.log10(cigale_mstar_results[tid])
            wm = wise_mstar_table['MSTAR'][i]
            cigale_mstar.append(cm)
            wise_mstar.append(wm)
            if len(cigale_mstar) != len(wise_mstar):
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
    sp = ax_main.scatter(wise_mstar, cigale_mstar, marker='+', alpha=0.05)
    ax_main.plot(np.linspace(7,13, 300), np.linspace(7, 13, 300), color='r')
    ax_main.set(xlabel=r"$m_\star$ from WISE color", ylabel=r"$m_\star$ from CIGALE", xlim=(7,13), ylim=(7,13))

    ax_yDist.hist(cigale_mstar, bins=200, orientation='horizontal', align='mid')
    ax_xDist.hist(wise_mstar, bins=200, orientation='vertical', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.show()


    #mstar_scatter = np.histogram2d(wise_mstar['MSTAR'], cigale_mstar)#, bins=100)
    #plt.imshow(mstar_scatter)
    #plt.xlabel(r"$\log(m_\star)$ from WISE1")
    #plt.ylabel(r"$\log(m_\star)$ from CIGALE")
    #plt.title("Stellar mass determination comparison")
    #plt.show()


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
    #hisnr_oii_lum = oii_lum[FSF.catalog['OII_SUMMED_SNR'][FSF_BGS_MASK] > 3]

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
    oii_lum = FSF.catalog['OII_COMBINED_LUMINOSITY'][FSF_BGS_MASK]
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
    plt.hist2d(np.log10(nii[full_mask]/ha[full_mask]), np.log10(oiii[full_mask]/hb[full_mask]), bins=100, range=((-2, 1), (-1.5, 2)), vmax=300)
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

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(4, 4)
    ax_main = plt.subplot(gs[1:4, :3])
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    plt.subplots_adjust(wspace=.0, hspace=.0)#, top=0.95)
    axs = [ax_main, ax_yDist]#, ax_xDist]
    sp = ax_main.scatter(n_oii, n_sii, marker='+', alpha=0.1)
    ax_main.set(xlabel=r"$\log{n_e}$ from [OII] (cm$^{-1}$)", ylabel=r"$\log{n_e}$ from [SII] (cm$^{-1}$)", ylim=(1.8, 4.6), xlim=(1.8, 4.6))

    ax_xDist.hist(n_oii, bins=200, orientation='vertical', align='mid')
    ax_yDist.hist(n_sii, bins=200, orientation='horizontal', align='mid')

    ax_yDist.invert_xaxis()
    ax_yDist.yaxis.tick_right()

    ax_xDist.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.savefig('figures/electron_density_comparison.png', dpi=600)
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


def main():

    #fsf = FSFCatalog()
    #zpix = ZPIXCatalog()
    #primary_program_dict = zpix.generate_primary_dict()
    #fsf.add_primary_to_table(primary_program_dict)

    global FSF_BGS_MASK, DR9_BGS_MASK, LSS, FSF, DR9, SNR_LIM
    SNR_LIM = 3
    # The FSF_BGS_MASK contains the cuts made at the catalog level, z <= .4, and removes failed redshifts
    # There are no SNR cuts in place as of now, as those are made on a per-line basis.
    LSS, FSF, DR9, FSF_BGS_MASK, DR9_BGS_MASK = generate_bgs_mask()

    #wise_m = all_WISE_mstars()

    #plt.hist(wise_m['MSTAR'], bins=100)
    #plt.xlabel(r'$m_{\star}$ ($m_{\odot}$)')
    #plt.show()

    #generate_cigale_input_table()
    #snr_cut_test_plot()
    #calc_SFR_Halpha()
    #compare_sfr()
    #compare_stellar_mass()
    #compare_stellar_mass_vs_catalog()
    #compare_electron_density()
    compare_stellar_mass()
    #luminosity_functions()
    #plot_oii_rat_vs_stellar_mass()
    #plot_sii_rat_vs_stellar_mass()
    #plot_bpt_diag()
    #plot_bpt_hist()
    #plot_loii_vs_redshift()
    #print(calc_mstar_WISE_color(496.86493, 316.6975, 0.12383684362225736)

    #for i in range(0,11):
    #    SNR_LIM = i
    #    hist_electron_density()


if __name__ == '__main__':
    main()
