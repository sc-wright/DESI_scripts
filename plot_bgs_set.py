import os

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import numpy as np

from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table

import pandas as pd

from utility_scripts import get_lum

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
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.specprod = 'fuji'
        self.specprod_dir = f'{my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.fsfCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2/catalogs'
        self.lssCatalogsDir = f'{my_dir}/public/edr/vac/edr/lss/v2.0/LSScats/full'
        self.ds9CatalogsDir = f'{my_dir}/public/edr/vac/edr/lsdr9-photometry/fuji/v2.1/observed-targets'

        self.fsfMeta = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji.fits', hdu=2)

        try:
            print("reading in fsf table...")
            self.fsfData = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji-data-processed.fits')
        except FileNotFoundError:
            print("FITS with pre-calculated values not found, generating new file...")
            self.spec_type_in_fsf()
            self.add_lum_to_table()
            snr = self.calculate_oii_snr()
            self.add_col_to_table("OII_SUMMED_SNR", snr)
            self.write_table_to_disk()

        self.fsfMeta = Table.read(f'{self.fsfCatalogsDir}/fastspec-fuji.fits', hdu=2)

        print("reading in lss table...")
        self.lssData = Table.read(f'{self.lssCatalogsDir}/BGS_ANY_full.dat.fits')

        # this all the non-vetoed targetids in the bgs. fully unique
        bgs_tids = self.lssData['TARGETID'][self.lssData['ZWARN'] == 0]

        # select the targetids from the fsf catalog that are also in the BGS_ANY lss catalog
        # then remove any that did not get appropriate L([OII]) fits.
        bgs_mask = [i in bgs_tids for i in self.fsfData['TARGETID']]
        self.bgs_mask_hiz = np.logical_and(bgs_mask, self.fsfData['OII_COMBINED_LUMINOSITY_LOG'] > 0)
        self.bgs_mask = np.logical_and(self.bgs_mask_hiz, self.fsfData['Z'] < .8)
        # bgs_mask is a boolean array that gets applied to the fsfData array to select data from good targets

    def spec_type_in_fsf(self):

        # This adds the survey tracer (BGS, ELG, LRG, QSO, STAR, SCND) to the table (in memory)
        # It only needs to be called if fastspec-fuji-data-processed.fits does not exist

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
        # This calculates the SNR of the OII flux.
        # It only needs to be called if fastspec-fuji-data-processed.fits does not exist

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

    def write_table_to_disk(self):
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

        print(f"Adding column {colstr} to table...")

        self.fsfData.add_column(Table.Column(data=-1 * np.ones(len(self.fsfData)), dtype=float, name=colstr))

        for i, v in enumerate(data):
            self.fsfData[colstr][i] = v

        # write_table_to_disk(table)

    def plot_lum_vs_redshift(self, zrange = False):
        """
        :param zrange:
        optional parameter zrange can be a two-value tuple with the low and high end of the redshift range you want to
        show in the plot.
        :return:
        returns nothing. Saves and shows the plots.
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
        plt.title(r"L_{[OII]} vs redshift")
        plt.xlabel("z")
        plt.ylabel(r"$L_{[OII]}$ (combined)")
        plt.legend()
        if zrange:
            plt.savefig(f'figures/bgs_loii_vs_redshift_z={zrange}.png')
        else:
            plt.savefig(f'figures/bgs_loii_vs_redshift_allz.png')
        plt.show()

    def plot_oii_rat_vs_stellar_mass(self):
        oii_rat = self.fsfData['OII_DOUBLET_RATIO'][self.bgs_mask]
        stellar_mass = self.fsfData['LOGMSTAR'][self.bgs_mask]
        redshift = self.fsfData['Z'][self.bgs_mask]

        print(oii_rat)
        print(stellar_mass)
        print(redshift)

        plt.scatter(stellar_mass, oii_rat, c=redshift, marker='.', alpha=0.3)
        plt.xlabel(r"$\log{M_{\star}}$")
        plt.ylabel("$\lambda 3726 / \lambda 3729$")
        plt.title("[OII] Doublet Ratio vs Stellar Mass")
        plt.colorbar(label="Z")
        plt.savefig("figures/oii_ratio_vs_mass.png")
        plt.show()

    def plot_oii_rat_vs_oii_flux(self):
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

    def plot_oii_lum_vs_color(self, zrange = False):

        if zrange:
            loz_mask = self.fsfData['Z'] >= zrange[0]
            hiz_mask = self.fsfData['Z'] < zrange[1]
            z_mask = np.logical_and(loz_mask, hiz_mask)
            full_mask = np.logical_and(z_mask, self.bgs_mask)
        else:
            full_mask = self.bgs_mask
        r_band = self.fsfMeta['FLUX_R'][full_mask] / self.fsfMeta['MW_TRANSMISSION_R'][full_mask]
        g_band = self.fsfMeta['FLUX_G'][full_mask] / self.fsfMeta['MW_TRANSMISSION_G'][full_mask]
        colors = g_band - r_band
        oii_luminosity = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][full_mask]
        redshift = self.fsfData['Z'][full_mask]

        plt.scatter(colors, oii_luminosity, c=redshift, marker='.', alpha=0.3)
        plt.xlabel("g-r [nmgy]")
        plt.ylabel(r"$L_{[OII]}$ (combined) [erg]")
        plt.title(r'$L_{[OII]}$ vs color')
        plt.colorbar(label="Z")
        if zrange:
            plt.savefig(f'figures/oii_luminosity_vs_g-r_color_z={zrange}.png')
            plt.xlim(-1000, 50)
            plt.ylim(36, 42)
            plt.savefig(f'figures/oii_luminosity_vs_g-r_color_z={zrange}[zoomed].png')

        else:
            plt.savefig(f'figures/oii_luminosity_vs_g-r_color.png')
            plt.xlim(-1000, 50)
            plt.ylim(36, 42)
            plt.savefig(f'figures/oii_luminosity_vs_g-r_color[zoomed].png')

        #plt.show()

    def plot_oii_lum_vs_color_old(self, zrange = False):
        print("reading in photometry table...")
        self.ds9Data = Table.read(f'{self.ds9CatalogsDir}/targetphot-sv3-fuji.fits')

        # Covnert the astropy Table to pandas for ease of table merging
        print('converting tables to pandas...')
        names = [name for name in self.fsfData.colnames if len(self.fsfData[name].shape) <= 1]
        fsfPd = self.fsfData[names].to_pandas()
        names = [name for name in self.ds9Data.colnames if len(self.ds9Data[name].shape) <= 1]
        ds9Pd = self.ds9Data[names].to_pandas()
        ds9Colors = ds9Pd[['TARGETID', 'FLUX_G', 'FLUX_R']]

        print('merging tables...')
        # Since the g and r columns are identical for any given targetid, we can just drop all the duplicate targetids
        ds9Unique = ds9Pd.drop_duplicates(subset=['TARGETID'])
        merged_color = pd.merge(fsfPd[self.bgs_mask], ds9Unique, on='TARGETID', how='left')

        """
        bgs_tids = self.fsfData['TARGETID'][self.bgs_mask]
        print("processing photometric data to make color plot...")
        color = np.zeros(len(bgs_tids))
        percent = 0
        for i, tid in enumerate(bgs_tids):
            targs = self.ds9Data['TARGETID'] == tid
            g = np.unique(self.ds9Data['FLUX_G'][targs])
            r = np.unique(self.ds9Data['FLUX_R'][targs])
            if len(g) > 1 and len(r) > 1:
                print(f"more than 1 entry for tid {tid}")
                print(len(g), len(r))
                print(g, r)
            else:
                color[i] = g[0] - r[0]
            prog = (i*100)//len(bgs_tids)
            if prog > percent:
                percent = prog
                print(f'\r{prog}% done...', end='', flush=True)
        """
        #oii_luminosity = self.fsfData['OII_COMBINED_LUMINOSITY_LOG'][self.bgs_mask]
        print(f'bgs mask: {len(self.bgs_mask)}')
        print(f"color mask: {len(merged_color['OII_COMBINED_LUMINOSITY_LOG'])}")
        #color_mask = merged_color['FLUX_G'] > 0 & merged_color['FLUX_R'] > 0 & self.bgs_mask

        oii_luinosity = merged_color['OII_COMBINED_LUMINOSITY_LOG']
        color = merged_color['FLUX_G'] - merged_color['FLUX_R']
        redshift = self.fsfData['Z'][self.bgs_mask]

        print('plotting...')
        plt.scatter(color, oii_luminosity, c=redshift, marker='.', alpha=0.3)
        plt.xlabel("g-r")
        plt.ylabel(r"$L_{[OII]}$ (combined)")
        plt.title(r'$L_{[OII]}$ vs color')
        plt.colorbar(label="Z")
        if zrange:
            plt.xlim(zrange[0], zrange[1])
            plt.savefig(f'figures/oii_luminosity_vs_g-r_color_z={zrange}.png')
        else:
            plt.savefig(f'figures/oii_luminosity_vs_g-r_color.png')
        plt.show()


    def generate_all_figs(self):
        self.plot_lum_vs_redshift(zrange=(0,.8))
        for i in np.arange(0,.8,.1):
            self.plot_lum_vs_redshift(zrange=(i, i+.1))
        self.plot_oii_rat_vs_stellar_mass()
        self.plot_oii_rat_vs_oii_flux()


if __name__ == '__main__':
    catalog = FSFCat()
    #catalog.generate_all_figs()
    catalog.plot_oii_lum_vs_color()