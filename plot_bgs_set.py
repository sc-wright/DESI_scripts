import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['text.usetex'] = True

import seaborn as sns

import numpy as np

from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table

import pandas as pd

from utility_scripts import get_lum, generate_combined_mask

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


class FSFCat:
    def __init__(self):
        """
        The FSFCat class creates a catalog object which contains a table with data from the vacs: especially the fsf cat
        It creates several useful attributes which are useful to make many plots
        self.specprod:          str: the codename for the release ('fuji' is for the edr)
        self.specprod_dir:      str: the local directory where the data for the release is stored
        self.fsfCatalogsDir:    str: the local directory where the fsf catalogs are stored
        self.lssCatalogsDir:    str: the local directory where the lss catalogs are stored
        self.ds9CatalogsDir:    str: the local directory where the ds9 catalogs are stored
        self.fsfMeta:           astropy table: the metadata table from the fsf catalog
        self.fsfData:           astropy table: the data table from the fsf catalog
        self.lssData:           astropy table: the data table from the lss catalog
        self.bgs_mask_hiz:      boolean array: an array that can be used to mask for all 'good' bgs sv3 targets
        self.bgs_mask           boolean array: same as above but only for the expected redshift: 0 > z > .6  (this will usually be used to make plots)

        There are more attributes that get created elsewhere


        todo: find a way to rewrite this so that it takes a table object as an argument without complicating usage so that the same catalog can be stored in memory for use in several different classes. Maybe another separate class
        todo: go through all the attributes and make sure it is clear which are defined in the instantiator
        todo: figure out if we can switch to a different/smaller fsf catalog for just bgs/sv3 targets to save memory and runtime
        """

        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.specprod = 'fuji'
        self.specprod_dir = f'{my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.fsfCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2/catalogs'
        self.lssCatalogsDir = f'{my_dir}/public/edr/vac/edr/lss/v2.0/LSScats/full'
        self.ds9CatalogsDir = f'{my_dir}/public/edr/vac/edr/lsdr9-photometry/fuji/v2.1/observed-targets'

        self.fsfMeta = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji.fits', hdu=2)

        try:
            # try to just read in the modified table with the extra columns added
            print("reading in fsf table...")
            self.fsfData = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji-data-processed.fits')
        except FileNotFoundError:
            # if the modified table doesn't exist, do the necessary calculations and create the table.
            # this can take several minutes to run. It should include a progress report for the longest step, but be patient for the rest.
            print("FITS with pre-calculated values not found, generating new file...")
            # The table gets read in inside of the first step.
            self.spec_type_in_fsf()
            self.add_lum_to_table()
            snr = self.calculate_oii_snr()
            self.add_col_to_table("OII_SUMMED_SNR", snr)
            self.write_table_to_disk()

        print("reading in lss table...")
        self.lssData = Table.read(f'{self.lssCatalogsDir}/BGS_ANY_full.dat.fits')

        print("removing failed redshifts...")
        # this all the non-vetoed targetids in the bgs. fully unique
        bgs_tids = self.lssData['TARGETID'][self.lssData['ZWARN'] == 0]

        # select the targetids from the fsf catalog that are also in the BGS_ANY lss catalog
        # then remove any that did not get appropriate L([OII]) fits.
        bgs_mask = np.isin(self.fsfData['TARGETID'], bgs_tids)
        #bgs_mask = [i in bgs_tids for i in self.fsfData['TARGETID']]

        print("making filters by redshift...")
        self.bgs_mask_hiz = np.logical_and(bgs_mask, self.fsfData['OII_COMBINED_LUMINOSITY_LOG'] > 0)
        self.bgs_mask = np.logical_and(self.bgs_mask_hiz, self.fsfData['Z'] < .6)  # can change redshift limit here - .6 is supposed to be the limit but there are many sources up to .8
        # bgs_mask is a boolean array that gets applied to the fsfData array to select data from good targets

    def spec_type_in_fsf(self):
        """
        This contains modified code from the desi tutorials to add survey tracers as tracers in the table.
        It is largely redundant at this point since we have determined better ways to create a list of meaningful targets.
        It creates and then modifies the self.fsfData table to add a boolean for each tracer in a column.
        It only needs to be called if fastspec-fuji-data-processed.fits does not exist.

        :return: None
        """

        #### Making lists to split the different surveys ####

        desi_masks = {}
        desi_masks['cmx'] = cmxmask
        desi_masks['sv1'] = sv1mask
        desi_masks['sv2'] = sv2mask
        desi_masks['sv3'] = sv3mask
        desi_masks['special'] = specialmask
        surveys = list(desi_masks.keys())

        mask_colnames = {}
        mask_colnames['cmx'] = 'CMX_TARGET'
        mask_colnames['sv1'] = 'SV1_DESI_TARGET'
        mask_colnames['sv2'] = 'SV2_DESI_TARGET'
        mask_colnames['sv3'] = 'SV3_DESI_TARGET'
        mask_colnames['special'] = 'DESI_TARGET'

        tracers = ['BGS', 'ELG', 'LRG', 'QSO', 'STAR', 'SCND']

        #### Reading in the relevant tables ####

        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        specprod = 'fuji'
        fsfCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'

        print("reading in table...")
        fsfData = Table.read(f'{fsfCatalogsDir}/fastspec-fuji.fits', hdu=1)
        fsfMeta = Table.read(f'{fsfCatalogsDir}/fastspec-fuji.fits', hdu=2)

        # Initialize columns to keep track of tracers. Set to -1 so we can ensure we fill all rows
        for tracer in tracers:
            fsfData.add_column(Table.Column(data=-1 * np.ones(len(fsfMeta)), dtype=int, name=f"IS{tracer}"))

        for survey in surveys:
            print(f'Identifying targets for survey: {survey}')
            desi_mask = desi_masks[survey]
            colname = mask_colnames[survey]
            bits = {}
            if survey == 'cmx':
                bgs = desi_mask.mask('MINI_SV_BGS_BRIGHT|SV0_BGS')
                elg = desi_mask.mask('MINI_SV_ELG|SV0_ELG')
                lrg = desi_mask.mask('MINI_SV_LRG|SV0_LRG')
                qso = desi_mask.mask('MINI_SV_QSO|SV0_QSO|SV0_QSO_Z5')
                starbitnames = 'STD_GAIA|SV0_STD_FAINT|SV0_STD_BRIGHT|STD_TEST|STD_CALSPEC|STD_DITHER|' \
                               + 'SV0_MWS_CLUSTER|SV0_MWS_CLUSTER_VERYBRIGHT|SV0_MWS|SV0_WD|BACKUP_BRIGHT|' \
                               + 'BACKUP_FAINT|M31_STD_BRIGHT|M31_H2PN|M31_GC|M31_QSO|M31_VAR|M31_BSPL|M31_M31cen|' \
                               + 'M31_M31out|ORI_STD_BRIGHT|ORI_QSO|ORI_ORI|ORI_HA|M33_STD_BRIGHT|M33_H2PN|M33_GC|' \
                               + 'M33_QSO|M33_M33cen|M33_M33out|SV0_MWS_FAINT|STD_DITHER_GAIA|STD_FAINT|STD_BRIGHT'
                star = desi_mask.mask(starbitnames)
                sec = 2 ** 70  # secondaries don't exist in cmx, so set it to above the 63rd bit
            else:
                bgs = desi_mask.mask('BGS_ANY')
                elg = desi_mask.mask('ELG')
                lrg = desi_mask.mask('LRG')
                qso = desi_mask.mask('QSO')
                sec = desi_mask.mask('SCND_ANY')
                star = desi_mask.mask('MWS_ANY|STD_FAINT|STD_WD|STD_BRIGHT')

            survey_selection = (fsfMeta[
                                    'SURVEY'] == survey)  # creates a mask of the desired survey to slap on top of the whole set
            survey_subset = fsfMeta[survey_selection]

            ## See if redrock thought it was a galaxy, star, or qso - this cannot be done with the fsf, no 'spectype' key
            GALTYPE = (survey_subset['SPECTYPE'] == 'GALAXY')
            STARTYPE = (survey_subset['SPECTYPE'] == 'STAR')
            QSOTYPE = (survey_subset['SPECTYPE'] == 'QSO')

            print(colname)

            ## BGS
            PASSES_BIT_SEL = ((survey_subset[colname] & bgs) > 0)
            fsfData['ISBGS'][survey_selection] = (PASSES_BIT_SEL & GALTYPE)

            ## ELG
            PASSES_BIT_SEL = ((survey_subset[colname] & elg) > 0)
            fsfData['ISELG'][survey_selection] = (PASSES_BIT_SEL & GALTYPE)

            ## LRG
            PASSES_BIT_SEL = ((survey_subset[colname] & lrg) > 0)
            fsfData['ISLRG'][survey_selection] = (PASSES_BIT_SEL & GALTYPE)

            ## QSO
            PASSES_BIT_SEL = ((survey_subset[colname] & qso) > 0)
            fsfData['ISQSO'][survey_selection] = (PASSES_BIT_SEL & QSOTYPE)

            ## STAR
            PASSES_BIT_SEL = ((survey_subset[colname] & star) > 0)
            fsfData['ISSTAR'][survey_selection] = (PASSES_BIT_SEL & STARTYPE)

            ## Secondaries
            PASSES_BIT_SEL = ((survey_subset[colname] & sec) > 0)
            fsfData['ISSCND'][survey_selection] = (PASSES_BIT_SEL)

            fsfMeta.remove_column(colname)

        for tracer in tracers:
            col = f"IS{tracer}"
            print(f"For {tracer}: {np.sum(fsfData[col] < 0):,} not set")
            if np.sum(fsfData[col] < 0) == 0:
                fsfData[col] = Table.Column(data=fsfData[col], name=col, dtype=bool)

        self.fsfData = fsfData

    def add_lum_to_table(self):
        """
        This calculates the [OII] luminosity and adds it in a new column to the table that already exists in memory.
        The luminosity column is called "OII_COMBINED_LUMINOSITY_LOG" which isn't a good name but it's too much trouble
        to change it now.
        It only needs to be called if fastspec-fuji-data-processed.fits does not exist.

        :return: None
        """
        # This adds the oii luminosity to the table (in memory)
        # It only needs to be called if fastspec-fuji-data-processed.fits does not exist

        self.fsfData.add_column(Table.Column(data=-1 * np.ones(len(self.fsfData)), dtype=float, name=f"OII_COMBINED_LUMINOSITY_LOG"))

        oII6Flux = np.array(self.fsfData['OII_3726_FLUX'])
        oII9Flux = np.array(self.fsfData['OII_3729_FLUX'])
        oIICombinedFlux = oII6Flux + oII9Flux
        redshift = np.array(self.fsfData['Z'])
        npix = np.array(self.fsfData['OII_3726_NPIX']) + np.array(self.fsfData['OII_3729_NPIX'])
        dataLength = len(oIICombinedFlux)

        t = time.time()
        lastFullSecElapsed = int(time.time() - t)

        for i in range(dataLength):
            if npix[i] > 1:
                flux = oIICombinedFlux[i]
                if flux > 0:
                    oIILum = np.log10(get_lum(flux, redshift[i]))
                    self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][i] = oIILum

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

    def calculate_oii_snr(self):
        """
        This calculates the SNR of the [OII] flux for every entry in the catalog.
        It calculates noise by square rooting the reciprocal of the "_AMP_IVAR" entry in the model for each line,
        and then adding them in quadrature.
        It calculates signal by adding the "_FLUX" entry in the model for the two lines.
        This only needs to be run if fastspec-fuji-data-processed.fits does not exist.

        todo: the "_AMP_IVAR" is inverse variance of the amplitude, not the flux. Is there a better way to do this?

        :return: an array of floats with the SNR for the combined [OII] flux for every entry in the catalog
        """


        noise1 = np.array(self.fsfData['OII_3726_AMP_IVAR'])
        noise1 = np.sqrt(1 / noise1)
        noise2 = np.array(self.fsfData['OII_3729_AMP_IVAR'])
        noise2 = np.sqrt(1 / noise2)
        noise = np.sqrt(noise1 ** 2 + noise2 ** 2)

        oII6Flux = np.array(self.fsfData['OII_3726_FLUX'])
        oII9Flux = np.array(self.fsfData['OII_3729_FLUX'])
        oIICombinedFlux = oII6Flux + oII9Flux

        snr = oIICombinedFlux / noise

        return snr

    def calculate_oii_rat_snr(self, snrlim, plot=False):
        """
        This calculates the snr for the oii ratio itself. Call it before
        1. Takes the inverse square root of the '_AMP_IVAR' for each line to get the uncertainty
        2. Divide the '_AMP' by the uncertainty for each line and then add these fractional uncertainties in quadrature
        3. The SNR then becomes 1/net uncertainty
        This is a naive way to calculate the snr, because the two inverse variances are coupled. This vastly overestimates the uncertainty.
        A better way will be implemented soon

        Creates a new attribute:
        self.snr_bgs_mask: boolean array: Can be used in place of the bgs_mask if we want to only access targets above a certain snr (set by snrlim)

        todo: Either make this run in the init by default so the snr is always available, or add it as another column to the table
        todo: Come up with and implement a better way to calculate SNR for the ratio, which will repalce this method

        :param snrlim: int/float: The minimum snr to cut by in the self.snr_bgs_mask
        :param plot: bool: if True, makes and saves a histogram of all the snr
        :return: None
        """
        # this is a huge overestimation since it treats the uncertainties as independent
        # just a top level look and a way to cut the absolutely most unreliable sources
        oii_3726_lineamp = self.fsfData['OII_3726_AMP']
        oii_3726_uncert = np.sqrt(1 / self.fsfData['OII_3726_AMP_IVAR'])
        oii_3729_lineamp = self.fsfData['OII_3729_AMP']
        oii_3729_uncert = np.sqrt(1 / self.fsfData['OII_3729_AMP_IVAR'])

        net_uncert = np.sqrt((oii_3726_uncert / oii_3726_lineamp) ** 2 + (oii_3729_uncert / oii_3729_lineamp) ** 2)

        oii_rat = self.fsfData['OII_DOUBLET_RATIO']
        #oii_rat_filt = oii_rat[self.bgs_mask]
        #snr_3726 = oii_3726_lineamp/oii_3726_uncert
        #snr_3729 = oii_3729_lineamp/oii_3729_uncert

        ratio_snr = 1/net_uncert
        self.snr_bgs_mask = np.logical_and(self.bgs_mask, ratio_snr > snrlim)

        if plot:
            plt.hist(ratio_snr[self.bgs_mask], bins=100, range=(0,25))
            plt.xlabel("SNR for [OII] doublet ratio in BGS (underestimate)")
            plt.savefig("naive_uncoupled_snr_hist.png")
            plt.show()

    def write_table_to_disk(self):
        """
        This takes the current self.fsfData table and writes it to a new fits file so calculations don't need to be remade
        The new fits file has just the one data table, the metadata (in hdu 2 of the original fits file) is lost.
        The metadata can still be read from the original fits file, as the rows are still matched.
        If the processed file is missing or needs to be remade, this is the last method to run to do that.

        :return: None
        """
        # This writes the current version of the table as it exists in memory to the disk
        # as "fastspec-fuji-data-processed.fits"
        # It only needs to be called if the processed table doesn't exist yet

        ogname = self.fsfCatalogsDir + "/fastspec-fuji-data-processed.fits"
        bakname = ogname + ".bak"

        print("Writing table...")
        try:
            print("renaming old table...")
            os.rename(ogname, bakname)
            print("writing new table to disk...")
            self.fsfData.write(self.fsfCatalogsDir + "/fastspec-fuji-data-processed.fits")
        except:
            print("old table not found, writing tale...")
            self.fsfData.write(self.fsfCatalogsDir + "/fastspec-fuji-data-processed.fits")

    def add_col_to_table(self, colstr, data):
        """
        This takes an array of any type (data) and adds it to the self.fsfData table (in memory) with the column name 'colstr'
        The array must be the same length as the other columns in the table

        :param colstr: str: the name of the column to add (CAPS by convention)
        :param data: any array: The data to add as a column. The array must be the same length as all the other columns in the table
        :return: None
        """

        print(f"Adding column {colstr} to table...")

        self.fsfData.add_column(Table.Column(data=-1 * np.ones(len(self.fsfData)), dtype=float, name=colstr))

        for i, v in enumerate(data):
            self.fsfData[colstr][i] = v

        # Optionally write out the whole table to disk again.
        # Generally keep this disabled so the table from disk always has known columns
        # write_table_to_disk(table)

    def plot_lum_vs_redshift(self, zrange = False):
        """
        Creates two new boolean arrays for the OII flux SNR and then plots the [OII] luminosity vs redshift, coloring
        dots with SNR < 3 differently.
        If no argument is given for zrange, it plots the expected range for BGS of 0 < z < .6

        :param zrange: tuple with 2 floats: low and high end of the redshift range to plot, respectively
        :return: None
        """
        snr_hi_mask = self.fsfData['OII_SUMMED_SNR'] >= 3
        snr_lo_mask = self.fsfData['OII_SUMMED_SNR'] < 3

        lum_hi = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask & snr_hi_mask]
        redshift_hi = self.fsfData['Z'][self.bgs_mask & snr_hi_mask]
        #snr = self.fsfData['OII_SUMMED_SNR'][self.bgs_mask]
        lum_lo = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask & snr_lo_mask]
        redshift_lo = self.fsfData['Z'][self.bgs_mask & snr_lo_mask]
        plt.scatter(redshift_hi, lum_hi, marker='.', alpha=0.3, label=r'SNR $\geq 3$')
        plt.scatter(redshift_lo, lum_lo, marker='.', alpha=0.3, label=r'SNR $< 3$')
        if zrange:
            plt.xlim(zrange[0], zrange[1])
        plt.title(r"$L_{[OII]}$ vs redshift")
        plt.xlabel("z")
        plt.ylabel(r"$L_{[OII]}$ (combined)")
        plt.legend()
        if zrange:
            plt.savefig(f'figures/bgs_loii_vs_redshift_z={zrange}.png')
        else:
            plt.savefig(f'figures/bgs_loii_vs_redshift_allz.png')
        plt.show()
        #plt.clf()

    def plot_oii_rat_vs_stellar_mass(self, mass_range=False):
        """
        Plots the OII doublet ratio vs the stellar mass with a histogram of the ratios
        Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

        :param mass_range: tuple with 2 floats: low and high end of the mass range to plot, respectively
        :return:
        """

        snr_lim = 7

        full_mask = generate_combined_mask([self.bgs_mask, self.fsfData['OII_SUMMED_SNR'] > snr_lim])

        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][full_mask]
        stellar_mass = self.fsfData['LOGMSTAR'][full_mask]
        colr = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][full_mask]

        if mass_range:
            plot_mask = generate_combined_mask([stellar_mass >= mass_range[0], stellar_mass < mass_range[1]])
        else:
            plot_mask = np.ones(len(stellar_mass), dtype=bool)

        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(2, 4)
        ax_main = plt.subplot(gs[0:2, :3])
        ax_yDist = plt.subplot(gs[0:2, 3], sharey=ax_main)
        plt.subplots_adjust(wspace=.0, top=0.99)
        axs = [ax_main, ax_yDist]
        sp = ax_main.scatter(stellar_mass[plot_mask], oii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3, vmax=41.5, vmin=39)
        fig.colorbar(sp, ax=axs, label=r"$L_{[OII]}$", location='top')
        ax_main.text(0.005, 1.005, f'total: {sum(plot_mask)}, snr $>$ {snr_lim}',
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
            plt.savefig("figures/oii_ratio_vs_mass_loiicol.png")
        plt.show()

    def plot_oii_rat_vs_stellar_mass_loii_bin(self, loii_range):
        """
        Plots the OII doublet ratio vs the stellar mass with a histogram of the ratios
        Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

        :param loii_range: tuple with 2 floats: low and high end of the luminosity range to plot, respectively
        :return:
        """

        snr_lim = 7

        full_mask = generate_combined_mask([self.bgs_mask, self.fsfData['OII_SUMMED_SNR'] > snr_lim])

        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][full_mask]
        stellar_mass = self.fsfData['LOGMSTAR'][full_mask]
        colr = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][full_mask]


        plot_mask = generate_combined_mask([colr >= loii_range[0], colr < loii_range[1]])

        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(2, 4)
        ax_main = plt.subplot(gs[0:2, :3])
        ax_yDist = plt.subplot(gs[0:2, 3], sharey=ax_main)
        plt.subplots_adjust(wspace=.0, top=0.99)
        axs = [ax_main, ax_yDist]
        sp = ax_main.scatter(stellar_mass[plot_mask], oii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3, vmax=42, vmin=39)
        fig.colorbar(sp, ax=axs, label=r"$L_{[OII]}$", location='top')
        ax_main.text(0.005, 1.005, f'total: {sum(plot_mask)}, snr $>$ {snr_lim}, ${loii_range[0]} \leq$ ' +  r'$\log{L_{[OII]}}$' + f' $< {loii_range[1]}$',
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=ax_main.transAxes)
        ax_main.set(xlabel=r"$\log{M_{\star}}$", ylabel="$\lambda 3726 / \lambda 3729$", ylim=(0, 2))

        ax_yDist.hist(oii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
        ax_yDist.set(xlabel='count')

        ax_yDist.invert_xaxis()
        ax_yDist.yaxis.tick_right()

        plt.savefig('figures/oii_ratio_vs_mass_loiicol_loii=({:.1f}'.format(loii_range[0]) + ',{:.1f}'.format(loii_range[1]) + ').png')

        plt.show()


    def plot_oii_rat_vs_stellar_mass_loii(self, loii_range):
        """
        Plots the OII doublet ratio vs the stellar mass with a histogram of the ratios
        Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

        :param loii_range: tuple with 2 floats: low and high end of the luminosity range to plot, respectively
        :return:
        """

        snr_lim = 7

        full_mask = generate_combined_mask([self.bgs_mask, self.fsfData['OII_SUMMED_SNR'] > snr_lim])

        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][full_mask]
        stellar_mass = self.fsfData['LOGMSTAR'][full_mask]
        colr = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][full_mask]


        plot_mask = generate_combined_mask([colr >= loii_range[0], colr < loii_range[1]])

        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(2, 4)
        ax_main = plt.subplot(gs[0:2, :3])
        ax_yDist = plt.subplot(gs[0:2, 3], sharey=ax_main)
        plt.subplots_adjust(wspace=.0, top=0.99)
        axs = [ax_main, ax_yDist]
        sp = ax_main.scatter(stellar_mass[plot_mask], oii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3, vmax=42, vmin=39)
        fig.colorbar(sp, ax=axs, label=r"$L_{[OII]}$", location='top')
        ax_main.text(0.005, 1.005, f'total: {sum(plot_mask)}, snr $>$ {snr_lim}, ${loii_range[0]} \leq$ ' +  r'$\log{L_{[OII]}}$' + f' $< {loii_range[1]}$',
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=ax_main.transAxes)
        ax_main.set(xlabel=r"$\log{M_{\star}}$", ylabel="$\lambda 3726 / \lambda 3729$", ylim=(0, 2))

        ax_yDist.hist(oii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
        ax_yDist.set(xlabel='count')

        ax_yDist.invert_xaxis()
        ax_yDist.yaxis.tick_right()

        plt.savefig('figures/oii_ratio_vs_mass_loiicol_loii=({:.1f}'.format(loii_range[0]) + ',{:.1f}'.format(loii_range[1]) + ').png')

        plt.show()


    def plot_oii_rat_vs_stellar_mass_old(self, mass_range=False, snr_lim=False, plt_col='redshift'):
        """
        Plots the OII doublet ratio vs the stellar mass with a histogram of the ratios
        Note: Stellar mass may be overestimated in v3.2 of the fastspecfit catalog

        :param mass_range: tuple with 2 floats: low and high end of the mass range to plot, respectively
        :param snr_lim: float or bool: if given, only plots line ratios with snrs above this value
        :return:
        """

        if snr_lim:
            self.calculate_oii_rat_snr(snr_lim, plot=False)

            oii_rat = self.fsfData['OII_DOUBLET_RATIO'][self.snr_bgs_mask]
            stellar_mass = self.fsfData['LOGMSTAR'][self.snr_bgs_mask]
            if plt_col == 'redshift':
                colr = self.fsfData['Z'][self.snr_bgs_mask]
            if plt_col == 'loii':
                colr = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.snr_bgs_mask]
        else:
            oii_rat = self.fsfData['OII_DOUBLET_RATIO'][self.bgs_mask]
            stellar_mass = self.fsfData['LOGMSTAR'][self.bgs_mask]
            if plt_col == 'redshift':
                colr = self.fsfData['Z'][self.bgs_mask]
            if plt_col == 'loii':
                colr = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask]



        if mass_range:
            plot_mask = generate_combined_mask([stellar_mass >= mass_range[0], stellar_mass < mass_range[1]])
        else:
            plot_mask = np.ones(len(stellar_mass), dtype=bool)

        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(2, 4)
        ax_main = plt.subplot(gs[0:2,:3])
        ax_yDist = plt.subplot(gs[0:2,3], sharey=ax_main)
        plt.subplots_adjust(wspace=.0, top=0.99)
        axs = [ax_main, ax_yDist]

        if plt_col == 'redshift':
            sp = ax_main.scatter(stellar_mass[plot_mask], oii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3,
                                 vmax=0.6, vmin=0.0)
            fig.colorbar(sp, ax=axs, label="Z", location='top')
        elif plt_col == 'loii':
            sp = ax_main.scatter(stellar_mass[plot_mask], oii_rat[plot_mask], c=colr[plot_mask], marker='.', alpha=0.3,
                                 vmax=41.5, vmin=39)
            fig.colorbar(sp, ax=axs, label=r"$L_{[OII]}$", location='top')

        ax_main.set(xlabel=r"$\log{M_{\star}}$", ylabel="$\lambda 3726 / \lambda 3729$", ylim=(0, 2))

        ax_yDist.hist(oii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
        ax_yDist.set(xlabel='count')

        ax_yDist.invert_xaxis()
        ax_yDist.yaxis.tick_right()

        if mass_range:
            if plt_col=='redshift':
                plt.savefig('figures/oii_ratio_vs_mass_zcol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
            elif plt_col == 'loii':
                plt.savefig('figures/oii_ratio_vs_mass_loiicol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
        else:
            if plt_col == 'redshift':
                plt.savefig("figures/oii_ratio_vs_mass_zcol.png")
            elif plt_col == 'loii':
                plt.savefig("figures/oii_ratio_vs_mass_loiicol.png")
        plt.show()


    def plot_oii_rat_vs_oii_flux_old(self):
        """
        Plots the OII amplitude ratio vs the total oii flux with redshift as a color axis

        :return: None
        """

        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][self.bgs_mask]
        oii_flux = self.fsfData['OII_3726_FLUX'][self.bgs_mask] + self.fsfData['OII_3729_FLUX'][self.bgs_mask]
        redshift = self.fsfData['Z'][self.bgs_mask]

        plt.scatter(np.log10(oii_flux), oii_rat, c=redshift, marker='.', alpha=0.3)
        plt.xlabel(r"$\log{(F_{\lambda 3726} + F_{\lambda 3729})}$ ($10^{-17}$ erg cm$^{-2}$")
        plt.ylabel("$\lambda 3726 / \lambda 3729$")
        plt.title("[OII] Doublet Ratio vs Combined [OII] Flux")
        plt.colorbar(label="Z")
        plt.savefig("figures/oii_ratio_vs_oii_flux.png")
        plt.show()
        plt.clf()

    def plot_oii_rat_vs_oii_flux(self, mass_range=False, redshift_range=False, snr_lim=1, plt_col='redshift'):

        self.calculate_oii_rat_snr(snr_lim, plot=False)

        if snr_lim:
            local_mask = self.snr_bgs_mask
        else:
            local_mask = self.bgs_mask

        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][local_mask]
        loii = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][local_mask]
        redshift = self.fsfData['Z'][local_mask]
        stellar_mass = self.fsfData['LOGMSTAR'][local_mask]

        if mass_range:
            plot_mask = generate_combined_mask([stellar_mass >= mass_range[0], stellar_mass < mass_range[1]])
        elif redshift_range:
            plot_mask = generate_combined_mask([redshift >= redshift_range[0], redshift < redshift_range[1]])
        else:
            plot_mask = np.ones(len(redshift), dtype=bool)

        fig = plt.figure(figsize=(8, 10))
        gs = GridSpec(2, 4)
        ax_main = plt.subplot(gs[0:2,:3])
        ax_yDist = plt.subplot(gs[0:2,3], sharey=ax_main)
        plt.subplots_adjust(wspace=.0, top=0.99)
        axs = [ax_main, ax_yDist]

        if plt_col == 'redshift':
            sp = ax_main.scatter(loii[plot_mask], oii_rat[plot_mask], c=redshift[plot_mask], marker='.', alpha=0.3,
                                 vmax=0.6, vmin=0.0)
            fig.colorbar(sp, ax=axs, label="Z", location='top')
        elif plt_col == 'loii':
            sp = ax_main.scatter(loii[plot_mask], oii_rat[plot_mask], c=stellar_mass[plot_mask], marker='.', alpha=0.3,
                                 vmax=41.5, vmin=39)
            fig.colorbar(sp, ax=axs, label=r"$\log{M_{\star}}$", location='top')

        ax_main.set(xlabel=r"$L_{[OII]}$", ylabel="$\lambda 3726 / \lambda 3729$", ylim=(0, 2))

        ax_yDist.hist(oii_rat[plot_mask], bins=100, orientation='horizontal', align='mid')
        ax_yDist.set(xlabel='count')

        ax_yDist.invert_xaxis()
        ax_yDist.yaxis.tick_right()

        if mass_range:
            if plt_col=='redshift':
                plt.savefig('figures/oii_ratio_vs_loii_zcol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
            elif plt_col == 'mass':
                plt.savefig('figures/oii_ratio_vs_loii_masscol_m=({:.1f}'.format(mass_range[0]) + ',{:.1f}'.format(mass_range[1]) + ').png')
        elif redshift_range:
            if plt_col == 'redshift':
                plt.savefig('figures/oii_ratio_vs_loii_zcol_m=({:.1f}'.format(redshift_range[0]) + ',{:.1f}'.format(
                    redshift_range[1]) + ').png')
            elif plt_col == 'mass':
                plt.savefig('figures/oii_ratio_vs_loii_masscol_m=({:.1f}'.format(redshift_range[0]) + ',{:.1f}'.format(
                    redshift_range[1]) + ').png')
        else:
            if plt_col == 'redshift':
                plt.savefig("figures/oii_ratio_vs_loii_zcol.png")
            elif plt_col == 'mass':
                plt.savefig("figures/oii_ratio_vs_loii_masscol.png")
        plt.show()


    def plot_oii_lum_vs_color(self, zrange = False):
        """
        Plots the total oii luminosity vs the g-r color using the SDSS g and r colors

        :param zrange: bool or tuple with 2 numbers: if given, low and high end of redshift range to plot, respectively. Otherwise plots bgs_mask range
        :return: None
        """

        if zrange:
            loz_mask = self.fsfData['Z'] >= zrange[0]
            hiz_mask = self.fsfData['Z'] < zrange[1]
            z_mask = np.logical_and(loz_mask, hiz_mask)
            full_mask = np.logical_and(z_mask, self.bgs_mask)
        else:
            full_mask = self.bgs_mask

        full_mask = generate_combined_mask([full_mask, self.fsfData['OII_SUMMED_SNR'] > 3])

        r_band = self.fsfData['ABSMAG01_SDSS_R'][full_mask]
        g_band = self.fsfData['ABSMAG01_SDSS_G'][full_mask]
        colors = g_band - r_band
        oii_luminosity = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][full_mask]
        redshift = self.fsfData['Z'][full_mask]
        if zrange:
            plt.scatter(colors, oii_luminosity, c=redshift, marker='.', alpha=0.3)
        else:
            plt.scatter(colors, oii_luminosity, c=redshift, marker='.', alpha=0.3, vmax=0.4)
        plt.xlabel("g-r")
        plt.ylabel(r"$L_{[OII]}$ (combined)")
        plt.title(r'$L_{[OII]}$ vs color')
        plt.xlim(-.9,2)
        plt.ylim(36,43)
        plt.colorbar(label="Z")
        if zrange:
            plt.savefig('figures/oii_luminosity_vs_g-r_color_z=({:.1f}'.format(zrange[0]) + ',{:.1f}'.format(zrange[1]) + ').png')
        else:
            plt.savefig(f'figures/oii_luminosity_vs_g-r_color.png')

        plt.show()
        plt.clf()

    def plot_oii_rat_vs_redshift(self, zrange=False):
        """
        Plots oii ratio vs redshift

        todo: add snr limit to this plot

        :param zrange: bool or tuple with 2 numbers: if given, low and high end of redshift range to plot, respectively. Otherwise plots bgs_mask range
        :return: None
        """
        self.calculate_oii_rat_snr(1, plot=False)
        if zrange:
            loz_mask = self.fsfData['Z'] >= zrange[0]
            hiz_mask = self.fsfData['Z'] < zrange[1]
            z_mask = np.logical_and(loz_mask, hiz_mask)
            full_mask = np.logical_and(z_mask, self.snr_bgs_mask)
        else:
            full_mask = self.snr_bgs_mask


        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][full_mask]
        redshift = self.fsfData['Z'][full_mask]
        mass = self.fsfData['LOGMSTAR'][full_mask]

        plt.scatter(redshift, oii_rat, c=mass, marker='.', alpha=0.3)
        plt.xlabel('Redshift')
        plt.ylabel("$\lambda 3726 / \lambda 3729$")
        plt.colorbar(label=r"$\log{M_\star}$")
        plt.title("[OII] doublet ratio vs redshift")
        if zrange:
            plt.savefig(f'figures/oii_ratio_vs_redshift_z={zrange}.png')
        else:
            plt.savefig('figures/oii_ratio_vs_redshift.png')
        plt.show()
        plt.clf()

    def plot_bpt_diag(self, write_to_file=False, loii_range=None, snr_lim = 3):
        """
        Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)

        Cuts:
        -SNR > 3 for all 4 lines

        :return: None
        """

        print("making bpt diagram...")

        nii = self.fsfData['NII_6584_FLUX'][self.bgs_mask]
        nii_snr = nii/(np.sqrt(1/self.fsfData['NII_6584_FLUX_IVAR'][self.bgs_mask]))
        ha = self.fsfData['HALPHA_FLUX'][self.bgs_mask]
        ha_snr = ha/(np.sqrt(1/self.fsfData['HALPHA_FLUX_IVAR'][self.bgs_mask]))
        oiii = self.fsfData['OIII_5007_FLUX'][self.bgs_mask]
        oiii_snr = oiii/(np.sqrt(1/self.fsfData['OIII_5007_FLUX_IVAR'][self.bgs_mask]))
        hb = self.fsfData['HBETA_FLUX'][self.bgs_mask]
        hb_snr = hb/(np.sqrt(1/self.fsfData['HBETA_FLUX_IVAR'][self.bgs_mask]))

        # todo: add snr calculation

        # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
        zero_mask = generate_combined_mask([nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0])

        full_mask = generate_combined_mask([zero_mask, nii_snr > snr_lim, ha_snr > snr_lim, oiii_snr > snr_lim, hb_snr > snr_lim, self.fsfData['OII_SUMMED_SNR'][self.bgs_mask] > snr_lim])


        # Getting the highest and lowest OII luminosity for the high SNR (>3) sources to use as the limits for the color bar. Otherwise its all basically just one color
        oii_lum = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask]
        #hisnr_oii_lum = oii_lum[self.fsfData['OII_SUMMED_SNR'][self.bgs_mask] > 3]

        if loii_range:
            full_mask = generate_combined_mask([full_mask, oii_lum >= loii_range[0], oii_lum < loii_range[1]])


        x_for_line_1 = np.log10(np.logspace(-5,.049,300))
        hii_agn_line = 0.61/(x_for_line_1 - 0.05) + 1.3

        x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
        composite_line_2 = 0.61/(x_for_line_2 - 0.47) + 1.19

        x_for_line_3 = np.linspace(-.13,2,100)
        agn_line_3 = 2.144507*x_for_line_3 + 0.465028

        if write_to_file:
            tids = self.fsfData['TARGETID'][self.bgs_mask]
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
        plt.colorbar(label=r"$\log{L_{[OII]}}$")
        plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
        plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')
        if loii_range:
            plt.savefig(f'figures/bpt_bgs_sv3_loii=({loii_range[0], loii_range[1]}).png',dpi=800)
        else:
            plt.savefig('figures/bpt_bgs_sv3.png',dpi=800)
        plt.show()



    def plot_bpt_diag_luminosity_bin(self, snr_lim = 3):
        """
        Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)

        Cuts:
        -SNR > 3 for all 4 lines and oii lum

        :return: None
        """

        print("making bpt diagram...")

        nii = self.fsfData['NII_6584_FLUX'][self.bgs_mask]
        nii_snr = nii/(np.sqrt(1/self.fsfData['NII_6584_FLUX_IVAR'][self.bgs_mask]))
        ha = self.fsfData['HALPHA_FLUX'][self.bgs_mask]
        ha_snr = ha/(np.sqrt(1/self.fsfData['HALPHA_FLUX_IVAR'][self.bgs_mask]))
        oiii = self.fsfData['OIII_5007_FLUX'][self.bgs_mask]
        oiii_snr = oiii/(np.sqrt(1/self.fsfData['OIII_5007_FLUX_IVAR'][self.bgs_mask]))
        hb = self.fsfData['HBETA_FLUX'][self.bgs_mask]
        hb_snr = hb/(np.sqrt(1/self.fsfData['HBETA_FLUX_IVAR'][self.bgs_mask]))

        # todo: add snr calculation

        # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
        zero_mask = generate_combined_mask([nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0])

        # Including only galaxies where all lines have SNR > 3 (by default)
        full_mask = generate_combined_mask([zero_mask, nii_snr > snr_lim, ha_snr > snr_lim, oiii_snr > snr_lim, hb_snr > snr_lim, self.fsfData['OII_SUMMED_SNR'][self.bgs_mask] > snr_lim])

        oii_lum = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask]

        x_for_line_1 = np.log10(np.logspace(-5,.049,300))
        hii_agn_line = 0.61/(x_for_line_1 - 0.05) + 1.3

        x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
        composite_line_2 = 0.61/(x_for_line_2 - 0.47) + 1.19

        x_for_line_3 = np.linspace(-.13,2,100)
        agn_line_3 = 2.144507*x_for_line_3 + 0.465028

        #fig = plt.figure(figsize=(16, 8))
        fig, axo = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 8))
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
        plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')

        for ax, loii in zip(axs, np.arange(39, 42, 0.5)):
            axis_mask = generate_combined_mask([full_mask, oii_lum >= loii, oii_lum < loii + 0.5])
            sp = ax.scatter(np.log10(nii[axis_mask]/ha[axis_mask]), np.log10(oiii[axis_mask]/hb[axis_mask]), marker='.', alpha=0.3, c=oii_lum[axis_mask], vmax=42, vmin=39)
            ax.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='k')
            ax.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r')
            ax.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b')
            ax.set_xlim(-2, 1)
            ax.set_ylim(-1.5, 2)
            #ax.text(-1.2, -0.9, "H II", fontweight='bold')
            #ax.text(-.21, -1.1, "Composite", fontweight='bold')
            #ax.text(-1.0, 1.5, "AGN", fontweight='bold')
            #ax.text(0.45, -0.8, "Shocks", fontweight='bold')
            ax.text(0.01, 0.98, f'total: {sum(axis_mask)}, snr $>$ {snr_lim}, ${loii} \leq$ ' +  r'$\log{L_{[OII]}}$' + f' $< {loii + 0.5}$',
                  horizontalalignment='left',
                  verticalalignment='top',
                  transform=ax.transAxes)
            #ax.set_xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
            #ax.set_ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')

        #ax = fig.add_subplot(111, frameon=False)
        #ax.spines['top'].set_color('none')
        #ax.spines['bottom'].set_color('none')
        #ax.spines['left'].set_color('none')
        #ax.spines['right'].set_color('none')
        #ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        #ax.set_xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
        #ax.set_ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')
        #plt.colorbar(sp, ax=axo, label=r"$\log{L_{[OII]}}$")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'figures/bpt_bgs_sv3_loii.png',dpi=800)
        plt.show()



    def plot_bpt_hist(self, loii_range=None, snr_lim = 3):
        """
        Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)

        Cuts:
        -SNR > 3 for all 4 lines

        :return: None
        """

        print("making bpt histogram...")

        nii = self.fsfData['NII_6584_FLUX'][self.bgs_mask]
        nii_snr = nii/(np.sqrt(1/self.fsfData['NII_6584_FLUX_IVAR'][self.bgs_mask]))
        ha = self.fsfData['HALPHA_FLUX'][self.bgs_mask]
        ha_snr = ha/(np.sqrt(1/self.fsfData['HALPHA_FLUX_IVAR'][self.bgs_mask]))
        oiii = self.fsfData['OIII_5007_FLUX'][self.bgs_mask]
        oiii_snr = oiii/(np.sqrt(1/self.fsfData['OIII_5007_FLUX_IVAR'][self.bgs_mask]))
        hb = self.fsfData['HBETA_FLUX'][self.bgs_mask]
        hb_snr = hb/(np.sqrt(1/self.fsfData['HBETA_FLUX_IVAR'][self.bgs_mask]))

        # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
        zero_mask = generate_combined_mask([nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0])

        full_mask = generate_combined_mask([zero_mask, nii_snr > snr_lim, ha_snr > snr_lim, oiii_snr > snr_lim, hb_snr > snr_lim, self.fsfData['OII_SUMMED_SNR'][self.bgs_mask] > snr_lim])


        # Getting the highest and lowest OII luminosity for the high SNR (>3) sources to use as the limits for the color bar. Otherwise its all basically just one color
        oii_lum = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask]
        #hisnr_oii_lum = oii_lum[self.fsfData['OII_SUMMED_SNR'][self.bgs_mask] > 3]

        if loii_range:
            full_mask = generate_combined_mask([full_mask, oii_lum >= loii_range[0], oii_lum < loii_range[1]])


        x_for_line_1 = np.log10(np.logspace(-5,.049,300))
        hii_agn_line = 0.61/(x_for_line_1 - 0.05) + 1.3

        x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
        composite_line_2 = 0.61/(x_for_line_2 - 0.47) + 1.19

        x_for_line_3 = np.linspace(-.13,2,100)
        agn_line_3 = 2.144507*x_for_line_3 + 0.465028

        f, ax = plt.subplots()
        plt.hist2d(np.log10(nii[full_mask]/ha[full_mask]), np.log10(oiii[full_mask]/hb[full_mask]), bins=100, range=((-2, 1), (-1.5, 2)), vmax=300)
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
        plt.colorbar(label=r"count")
        plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
        plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')
        if loii_range:
            plt.savefig(f'figures/bpt_hist_bgs_sv3_loii=({loii_range[0], loii_range[1]}).png',dpi=800)
        else:
            plt.savefig('figures/bpt_hist_bgs_sv3.png',dpi=800)
        plt.show()



    def plot_bpt_hist_with_loii(self, snr_lim = 3):
        """
        Plots line ratios in bpt-style diagram with AGN/HII separator lines from Kewley et al. (2001) and Kauffmann et al. (2003)

        Cuts:
        -SNR > 3 for all 4 lines

        :return: None
        """

        print("making bpt histogram...")

        # Pulling all lines and calculating their snr
        nii = self.fsfData['NII_6584_FLUX'][self.bgs_mask]
        nii_snr = nii/(np.sqrt(1/self.fsfData['NII_6584_FLUX_IVAR'][self.bgs_mask]))
        ha = self.fsfData['HALPHA_FLUX'][self.bgs_mask]
        ha_snr = ha/(np.sqrt(1/self.fsfData['HALPHA_FLUX_IVAR'][self.bgs_mask]))
        oiii = self.fsfData['OIII_5007_FLUX'][self.bgs_mask]
        oiii_snr = oiii/(np.sqrt(1/self.fsfData['OIII_5007_FLUX_IVAR'][self.bgs_mask]))
        hb = self.fsfData['HBETA_FLUX'][self.bgs_mask]
        hb_snr = hb/(np.sqrt(1/self.fsfData['HBETA_FLUX_IVAR'][self.bgs_mask]))

        # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
        zero_mask = generate_combined_mask([nii != 0.0, ha != 0.0, oiii != 0.0, hb != 0.0])

        # implementing snr limits
        full_mask = generate_combined_mask([zero_mask, nii_snr > snr_lim, ha_snr > snr_lim, oiii_snr > snr_lim, hb_snr > snr_lim, self.fsfData['OII_SUMMED_SNR'][self.bgs_mask] > snr_lim])

        oii_lum = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask]

        # creating lines to use on the plot
        x_for_line_1 = np.log10(np.logspace(-5,.049,300))
        hii_agn_line = 0.61/(x_for_line_1 - 0.05) + 1.3

        x_for_line_2 = np.log10(np.logspace(-5, 0.46, 300))
        composite_line_2 = 0.61/(x_for_line_2 - 0.47) + 1.19

        x_for_line_3 = np.linspace(-.13,2,100)
        agn_line_3 = 2.144507*x_for_line_3 + 0.465028


        # Creating pandas df
        ratios = pd.DataFrame(list(zip(np.log10(nii[full_mask]/ha[full_mask]), np.log10(oiii[full_mask]/hb[full_mask]), oii_lum)), columns=['x', 'y', 'lum'])


        # Plotting
        f, ax = plt.subplots()

        #coordinates['']

        ratios['x_bin'] = pd.cut(ratios['x'], bins=100)
        ratios['y_bin'] = pd.cut(ratios['y'], bins=100)

        grouped = ratios.groupby(['x_bin', 'y_bin'], as_index=False)['lum'].median()
        data = grouped.pivot(index='y_bin', columns='x_bin', values='lum')
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(data, ax=ax, xticklabels=10, yticklabels=10)


        #plt.hist2d(np.log10(nii[full_mask]/ha[full_mask]), np.log10(oiii[full_mask]/hb[full_mask]), bins=100, range=((-2, 1), (-1.5, 2)), vmax=300)
        plt.plot(x_for_line_1, hii_agn_line, linestyle='dashed', color='k')
        plt.plot(x_for_line_2, composite_line_2, linestyle='dotted', color='r')
        plt.plot(x_for_line_3, agn_line_3, linestyle='dashdot', color='b')
        #plt.text(-1.2, -0.9, "H II", fontweight='bold')
        #plt.text(-.21, -1.1, "Composite", fontweight='bold')
        #plt.text(-1.0, 1.5, "AGN", fontweight='bold')
        #plt.text(0.45, -0.8, "Shocks", fontweight='bold')
        ax.text(0.005, 1.005, f'total: {sum(full_mask)}, snr $>$ {snr_lim}',
              horizontalalignment='left',
              verticalalignment='bottom',
              transform=ax.transAxes)

        #ax.set_xticks(np.arange(-1.5, 1.5, 0.3))
        #ax.set_yticks(np.arange(-2.2, 0.8, 0.3))

        #plt.xlim(-2, 1)
        #plt.ylim(-1.5, 2)
        #plt.colorbar(label=r"$\log{L_{[OII]}}$")
        plt.xlabel(r'$\log([N II]_{\lambda 6584} / H\alpha)$')
        plt.ylabel(r'$\log([O III]_{\lambda 5007} / H\beta)$')
        ax.invert_yaxis()

        plt.savefig('figures/bpt_hist_lum_color_bgs_sv3.png',dpi=800)
        plt.show()

    def generate_all_figs(self):
        """
        Just runs all the plotting routines to generate all-new plots.
        Probably missing some now

        todo: Once all the plotting routines are written, make sure they are all represented here

        :return: None
        """
        self.plot_lum_vs_redshift()
        for i in np.arange(0,.8,.1):
            self.plot_lum_vs_redshift(zrange=(i, i+.1))
        self.plot_oii_lum_vs_color()
        self.plot_oii_rat_vs_stellar_mass()
        self.plot_oii_rat_vs_oii_flux()
        self.plot_oii_rat_vs_redshift()

    def report_all_zero_rat(self):
        """
        Quick script to make a list of all the TARGETIDs with an OII doublet ratio less than 0.01 (there are none
        exactly equal to zero) so that we can investigate their spectra
        Writes a text file to the hard drive in the working directory (of the Pycharm project

        :return: None
        """
        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][self.bgs_mask]
        zoii = self.fsfData['TARGETID'][self.bgs_mask]
        zoii = zoii[oii_rat < 0.01]
        with open('zero_rat_tids.txt', 'w') as f:
            for tid in zoii:
                f.write(f"{tid}\n")


if __name__ == '__main__':
    catalog = FSFCat()
    #catalog.calculate_oii_rat_snr(1)  # the argument is the naive snr calculation to use
    #catalog.generate_all_figs()
    #for i in np.arange(0,.5, .1):
    #    catalog.plot_oii_lum_vs_color(zrange=[i,i+.1])
    #catalog.plot_oii_rat_vs_redshift()
    #catalog.plot_oii_rat_vs_stellar_mass()
    #for i in np.arange(7,12):
    #    catalog.plot_oii_rat_vs_stellar_mass(mass_range=[i, i+1])
    #for i in np.arange(39, 42, .5):
    #    catalog.plot_oii_rat_vs_stellar_mass_loii_bin([i, i + .5])

    catalog.plot_bpt_hist_with_loii()

    #catalog.report_all_zero_rat()
    #catalog.plot_bpt_diag(write_to_file=False)
    #oii_lum = catalog.fsfData['OII_COMBINED_LUMINOSITY_LOG'][catalog.bgs_mask]
    #tid = catalog.fsfData['TARGETID'][catalog.bgs_mask]
    #print(tid[oii_lum < 32.5])

    #catalog.plot_bpt_diag()
    #catalog.plot_bpt_diag_luminosity_bin()
    #for i in np.arange(39, 42, 0.5):
    #    catalog.plot_bpt_diag(loii_range=[i, i+0.5])

    #catalog.plot_bpt_hist()