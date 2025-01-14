import os
import mks

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table

from utility_scripts import get_lum, generate_combined_mask

plt.rcParams['text.usetex'] = True

import time
import hashlib
import glob
import random

import pandas as pd

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
        self.fssCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2/catalogs'
        try:
            print("reading in table...")
            self.fsfData = Table.read(f'{self.fssCatalogsDir}/fastspec-fuji-data-processed.fits')
        except:
            print("FITS with pre-calculated values not found, generating new file...")
            self.spec_type_in_fsf()
            self.add_lum_to_table()
            snr = self.calculate_oii_snr()
            self.add_col_to_table("OII_SUMMED_SNR", snr)
            self.write_table_to_disk()

    def spec_type_in_fsf(self):

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
        fssCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'

        print("reading in table...")
        fsfData = Table.read(f'{fssCatalogsDir}/fastspec-fuji.fits', hdu=1)
        fsfMeta = Table.read(f'{fssCatalogsDir}/fastspec-fuji.fits', hdu=2)

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

            ## See if redrock thought it was a galaxy, star, or qso - this cannot be done with the fss, no 'spectype' key
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
        self.fsfMeta = fsfMeta

    def add_lum_to_table(self):
        # table should be the fsf table

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

    def write_table_to_disk(self):
        ogname = self.fssCatalogsDir + "/fastspec-fuji-data-processed.fits"
        bakname = ogname + ".bak"

        print("Writing table...")
        try:
            print("renaming old table...")
            os.rename(ogname, bakname)
            print("writing new table to disk...")
            self.fsfData.write(self.fssCatalogsDir + "/fastspec-fuji-data-processed.fits")
        except:
            print("old table not found, writing tale...")
            self.fsfData.write(self.fssCatalogsDir + "/fastspec-fuji-data-processed.fits")

    def calculate_oii_snr(self):

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

    def add_col_to_table(self, colstr, data):

        print(f"Adding column {colstr} to table...")

        self.fsfData.add_column(Table.Column(data=-1 * np.ones(len(self.fsfData)), dtype=float, name=colstr))

        for i, v in enumerate(data):
            self.fsfData[colstr][i] = v

        # write_table_to_disk(table)


class Spectrum:
    def __init__(self, targetid='random', zpix_cat=None):

        #print("reading in table...")
        self.basedir = os.path.expanduser('~') + '/Documents/school/research'
        self.my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        self.survey = 'sv3'
        self.program = 'bright'
        self.specprod = 'fuji'
        self.specprod_dir = f'{self.my_dir}/public/edr/spectro/redux/{self.specprod}'
        self.fastspec_dir = f'{self.my_dir}/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2'
        self.healpix_dir = self.specprod_dir + '/healpix'
        if zpix_cat is None:
            self.zpix_cat = Table.read(f'{self.specprod_dir}/zcatalog/zall-pix-{self.specprod}.fits', hdu="ZCATALOG")
        else:
            self.zpix_cat = zpix_cat
        #self.fsf_models = Table.read(f'{self.fastspec_dir}/fastspec-fuji.fits', hdu=3)

        if targetid == 'random':
            bgs_tgtmask = sv3mask["BGS_ANY"]
            #targetid = random.choice(self.zpix_cat['TARGETID'][self.zpix_cat['SV3_BGS_TARGET'] > 0])
            targetid = random.choice(self.zpix_cat['TARGETID'][(self.zpix_cat["SV3_DESI_TARGET"] & bgs_tgtmask != 0)])# & self.zpix_cat["ZWARN"] > 0)])

            self.targetid = targetid
            print(f"targetid {targetid}")
            self.check_for_files()  # if plotting a random spectrum, we're assuming we need to check if the files exist
        else:
            self.targetid = targetid
            print(f"targetid {targetid}")

        #self.files_checked = False
        #self.check_for_files()

    def plot_spectrum(self, foldstruct="spectra/", display_plot=False):
        """
        return 6 means there were multiple matching targetids found in the selected fastspecfit file

        :param foldstruct:
        :return:
        """

        selected_tgts = self.zpix_cat['TARGETID'] == self.targetid

        zcat_sel = self.zpix_cat[selected_tgts]

        survey_col = zcat_sel['SURVEY'].astype(str)
        program_col = zcat_sel['PROGRAM'].astype(str)
        hpx_col = zcat_sel['HEALPIX']
        redshift_col = zcat_sel['Z']

        is_primary = zcat_sel['ZCAT_PRIMARY']

        for survey, program, hpx, redshift in zip(survey_col[is_primary], program_col[is_primary], hpx_col[is_primary], redshift_col[is_primary]):
            # This loop should only run once, as there should be only one primary spectrum for a given tid
            print(survey, program, hpx, redshift)
            tgt_dir = f'{self.healpix_dir}/{survey}/{program}/{hpx // 100}/{hpx}'
            zFact = redshift + 1

            # Filename
            coadd_filename = f'coadd-{survey}-{program}-{hpx}.fits'
            specfile = f'{tgt_dir}/{coadd_filename}'

            spec = read_spectra(specfile).select(targets=self.targetid)

            coadd_spec = coadd_cameras(spec)
            bands = coadd_spec.bands[0]

            fastfile = self.fastspec_dir + f'/healpix/{survey}/{program}/{hpx // 100}/{hpx}/fastspec-{survey}-{program}-{hpx}.fits.gz'
            meta = Table(fitsio.read(fastfile, 'METADATA'))
            data = Table(fitsio.read(fastfile, 'FASTSPEC'))

            #zwarn is currently never set
            zflag = False

            fsf_target_mask = data['TARGETID'] == self.targetid

            if sum(fsf_target_mask) > 1:
                print("more than one matching targetid was found in the fastspec table. please check the data.")
                return 6
            oii_rat = data['OII_DOUBLET_RATIO'][fsf_target_mask][0]
            oii_snr_26 = data['OII_3726_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['OII_3726_AMP_IVAR'][fsf_target_mask][0])
            oii_snr_29 = data['OII_3729_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['OII_3729_AMP_IVAR'][fsf_target_mask][0])

            #oii_vshift = data['OII_3726_VSHIFT'][fsf_target_mask][0]
            #oii_zshift = oii_vshift*1000/mks.c + 1

            sii_rat = data['SII_DOUBLET_RATIO'][fsf_target_mask][0]
            sii_snr_16 = data['SII_6716_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['SII_6716_AMP_IVAR'][fsf_target_mask][0])
            sii_snr_31 = data['SII_6731_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['SII_6731_AMP_IVAR'][fsf_target_mask][0])

            #print(data['SII_6716_AMP'][fsf_target_mask][0], data['SII_6716_AMP_IVAR'][fsf_target_mask][0])

            neiii_snr = data['NEIII_3869_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['NEIII_3869_AMP_IVAR'][fsf_target_mask][0])
            hb_snr = data['HBETA_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['HBETA_AMP_IVAR'][fsf_target_mask][0])
            oiii59_snr = data['OIII_4959_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['OIII_4959_AMP_IVAR'][fsf_target_mask][0])
            oiii07_snr = data['OIII_5007_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['OIII_5007_AMP_IVAR'][fsf_target_mask][0])
            nii48_snr = data['NII_6548_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['NII_6548_AMP_IVAR'][fsf_target_mask][0])
            ha_snr = data['HALPHA_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['HALPHA_AMP_IVAR'][fsf_target_mask][0])
            nii84_snr = data['NII_6584_MODELAMP'][fsf_target_mask][0] * np.sqrt(data['NII_6584_AMP_IVAR'][fsf_target_mask][0])

            models, hdr = fitsio.read(fastfile, 'MODELS', header=True)
            models = models[meta['TARGETID'] == self.targetid]
            modelwave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1']) * hdr['CDELT1']

            mw_transmission_spec = dust_transmission(coadd_spec.wave[bands], meta['EBV'][meta['TARGETID'] == self.targetid])

            line_names = [r'$[OII]$', r'$[OIII]$', r'$[OIII]$', r'$H\alpha$', r'$H\beta$', r'$H\gamma$',
                          r'$H\delta$', r'$[SII]$', r'$[SII]$', r'$CaII H$', r'$CaII K$', r'$[NII]$',
                          r'$[NII]$', r'$[NeIII]$']
            line_vals = [3728.48, 4960.2937, 5008.2383, 6564.613, 4862.683, 4341.684, 4102.892, 6718.2913, 6732.6705, 3933, 3968, 6549.8578, 6585.2696, 3869.8611]

            #lines = {line_names[i]: line_vals[i] for i in range(len(line_names))}

            spec_lo = 3500
            spec_hi = 7000

            nev_limit = 15
            oii_limit = 8
            hb_limit = 10
            oiii_limit = 35
            nii_limit = 25
            sii_limit = 20

            # this is the wl to use as the center of the subfig
            oii_line = (3727.0919 + 3729.8750) / 2
            sii_line = (6718.2913 + 6732.6705) / 2
            nii_line = (6549.8578 + 6564.613 + 6585.2696) / 3 # average of the 3 lines showing in that subfig
            oiii_line = (4960.2937 + 5008.2383) / 2
            hb_line = 4862.683
            nev_line = 3869.8611

            # buff determines how much space to put above and below the top of the lines
            buff = 0.15

            full_spec = coadd_spec.flux['brz'][0] / mw_transmission_spec

            # get maxima and minima for full plot
            full_left_lim = coadd_spec.wave['brz'] / zFact > (spec_lo)
            full_right_lim = coadd_spec.wave['brz'] / zFact < (spec_hi)
            full_xlims = np.logical_and(full_left_lim, full_right_lim)
            full_y_top = max(full_spec[full_xlims])
            full_y_bottom = min(full_spec[full_xlims])
            full_y_range = full_y_top - full_y_bottom
            full_y_top += full_y_range*buff
            full_y_bottom -= full_y_range*buff

            # get maxima and minima for oii subplot
            oii_left_lim = coadd_spec.wave['brz'] / zFact > (oii_line-oii_limit)
            oii_right_lim = coadd_spec.wave['brz'] / zFact < (oii_line+oii_limit)
            oii_xlims = np.logical_and(oii_left_lim, oii_right_lim)
            try:
                oii_y_top = max(full_spec[oii_xlims])
                oii_y_bottom = min(full_spec[oii_xlims])
            except ValueError:
                oii_y_top = 1
                oii_y_bottom = 0
            oii_y_range = oii_y_top - oii_y_bottom
            oii_y_top += oii_y_range*buff
            oii_y_bottom -= oii_y_range*buff

            # get maxima and minima for sii subplot
            sii_left_lim = coadd_spec.wave['brz'] / zFact > (sii_line-sii_limit)
            sii_right_lim = coadd_spec.wave['brz'] / zFact < (sii_line+sii_limit)
            sii_xlims = np.logical_and(sii_left_lim, sii_right_lim)
            try:
                sii_y_top = max(full_spec[sii_xlims])
                sii_y_bottom = min(full_spec[sii_xlims])
            except ValueError:
                sii_y_top = 1
                sii_y_bottom = 0
            sii_y_range = sii_y_top - sii_y_bottom
            sii_y_top += sii_y_range*buff
            sii_y_bottom -= sii_y_range*buff

            # get maxima and minima for nii subplot
            nii_left_lim = coadd_spec.wave['brz'] / zFact > (nii_line-nii_limit)
            nii_right_lim = coadd_spec.wave['brz'] / zFact < (nii_line+nii_limit)
            nii_xlims = np.logical_and(nii_left_lim, nii_right_lim)
            try:
                nii_y_top = max(full_spec[nii_xlims])
                nii_y_bottom = min(full_spec[nii_xlims])
            except ValueError:
                nii_y_top = 1
                nii_y_bottom = 0
            nii_y_range = nii_y_top - nii_y_bottom
            nii_y_top += nii_y_range*buff
            nii_y_bottom -= nii_y_range*buff

            #get maxima and minima for oiii subplot
            oiii_left_lim = coadd_spec.wave['brz'] / zFact > (oiii_line-oiii_limit)
            oiii_right_lim = coadd_spec.wave['brz'] / zFact < (oiii_line+oiii_limit)
            oiii_xlims = np.logical_and(oiii_left_lim, oiii_right_lim)
            try:
                oiii_y_top = max(full_spec[oiii_xlims])
                oiii_y_bottom = min(full_spec[oiii_xlims])
            except ValueError:
                oiii_y_top = 1
                oiii_y_bottom = 0
            oiii_y_range = oiii_y_top - oiii_y_bottom
            oiii_y_top += oiii_y_range*buff
            oiii_y_bottom -= oiii_y_range*buff

            # get maxima and minima for hb subplot
            hb_left_lim = coadd_spec.wave['brz'] / zFact > (hb_line-hb_limit)
            hb_right_lim = coadd_spec.wave['brz'] / zFact < (hb_line+hb_limit)
            hb_xlims = np.logical_and(hb_left_lim, hb_right_lim)
            try:
                hb_y_top = max(full_spec[hb_xlims])
                hb_y_bottom = min(full_spec[hb_xlims])
            except ValueError:
                hb_y_top = 1
                hb_y_bottom = 0
            hb_y_range = hb_y_top - hb_y_bottom
            hb_y_top += hb_y_range*buff
            hb_y_bottom -= hb_y_range*buff

            nev_left_lim = coadd_spec.wave['brz'] / zFact > (nev_line-nev_limit)
            nev_right_lim = coadd_spec.wave['brz'] / zFact < (nev_line+nev_limit)
            nev_xlims = np.logical_and(nev_left_lim, nev_right_lim)
            try:
                nev_y_top = max(full_spec[nev_xlims])
                nev_y_bottom = min(full_spec[nev_xlims])
            except ValueError:
                nev_y_top = 1
                nev_y_bottom = 0
            nev_y_range = nev_y_top - nev_y_bottom
            nev_y_top += nev_y_range*buff
            nev_y_bottom -= nev_y_range*buff


            plt.figure(figsize=(12, 9))
            ax1 = plt.subplot(3, 3, (1, 3))
            ax2 = plt.subplot(3, 3, 4)
            ax3 = plt.subplot(3, 3, 5)
            ax4 = plt.subplot(3, 3, 6)
            ax5 = plt.subplot(3, 3, 7)
            ax6 = plt.subplot(3, 3, 8)
            ax7 = plt.subplot(3, 3, 9)
            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            # PLOTTING FULL SPECTRUM
            ax1.plot(coadd_spec.wave['brz']/zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='maroon', alpha=0.5)
            #plot the models
            ax1.plot(modelwave / zFact, models[0, 0, :], label='Stellar Continuum Model', ls='-', color='blue')
            ax1.plot(modelwave / zFact, models[0, 1, :], label='Smooth Continuum Correction', ls='--', color='k')

            #ax1.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)), color='k', lw=2.0)

            ax1.set_xlim([spec_lo, spec_hi])
            ax1.set_ylim([full_y_bottom, full_y_top])
            lastline = 0
            vertpos = 0.90
            trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
            for line, name in sorted(zip(line_vals, line_names)):
                if line - lastline < 70:
                    vertpos -=.06
                else:
                    vertpos = 0.90
                ax1.axvline(x = line, linestyle='dashed', lw = 0.8, alpha=0.4)
                ax1.text(line+8, vertpos, name,
                         horizontalalignment='left',
                         verticalalignment='center',
                         fontsize=8, transform=trans)
                lastline = line
            if zflag:
                ax1.text(0.005, 1.01, f'targetid: {self.targetid}, z = {zFact-1}*, survey = {survey}, program = {program}',
                         horizontalalignment='left',
                         verticalalignment='bottom',
                         transform=ax1.transAxes)
            else:
                ax1.text(0.005, 1.01, f'targetid: {self.targetid}, z = {zFact-1}, survey = {survey}, program = {program}',
                         horizontalalignment='left',
                         verticalalignment='bottom',
                         transform=ax1.transAxes)
            ax1.set_xlabel(r'$\lambda_{rest}$')
            ax1.legend(fontsize=8, loc='lower right')



            # plotting oii spectrum
            ax2.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4)
            #ax2.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax2.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red', linewidth=1)
            ax2.set_xlim([oii_line-oii_limit, oii_line+oii_limit])
            ax2.set_ylim([oii_y_bottom, oii_y_top])
            ax2.text(0.995, 0.975, f'[O II]/[O II]\n' + 'ratio: {0:.4f}\n'.format(oii_rat) + 'SNR: {0:.1f}'.format(oii_snr_26) + ', {0:.1f}'.format(oii_snr_29),
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax2.transAxes)

            ax2.axvline(3727.0919, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax2.axvline(3729.8750, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax2.set_xlabel(r'$\lambda_{rest}$')



            # plotting nev spectrum
            ax3.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4)
            #ax3.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax3.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red', linewidth=1)
            ax3.set_xlim([nev_line-nev_limit, nev_line+nev_limit])
            ax3.set_ylim([nev_y_bottom, nev_y_top])
            ax3.text(0.995, 0.975, f'[Ne III]\n' + 'SNR: {0:.1f}'.format(neiii_snr),
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax3.transAxes)
            ax3.axvline(3869.8611, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax3.set_xlabel(r'$\lambda_{rest}$')


            # plotting hb spectrum
            ax4.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4)
            #ax4.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax4.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red', linewidth=1)
            ax4.set_xlim([hb_line-hb_limit, hb_line+hb_limit])
            ax4.set_ylim([hb_y_bottom, hb_y_top])
            ax4.text(0.995, 0.975, r'$H \beta$' + '\nSNR: {0:.1f}'.format(hb_snr),
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax4.transAxes)
            ax4.axvline(4862.683, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax4.set_xlabel(r'$\lambda_{rest}$')


            # plotting oiii spectrum
            ax5.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4)
            #ax5.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax5.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red', linewidth=1)
            ax5.set_xlim([oiii_line - oiii_limit, oiii_line + oiii_limit])
            ax5.set_ylim([oiii_y_bottom, oiii_y_top])
            ax5.text(0.995, 0.975, '[O III]/[O III]\n' + 'SNR: {0:.1f}'.format(oiii59_snr) + ', {0:.1f}'.format(oiii07_snr),
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax5.transAxes)
            ax5.axvline(4960.2937, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax5.axvline(5008.2383, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax5.set_xlabel(r'$\lambda_{rest}$')


            # plotting nii spectrum
            ax6.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4)
            #ax6.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax6.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red', linewidth=1)

            ax6.set_xlim([nii_line-nii_limit, nii_line+nii_limit])
            ax6.set_ylim([nii_y_bottom, nii_y_top])
            ax6.text(0.995, 0.975, fr'[N II]/H$\alpha$/[N II]' + '\nSNR: {0:.1f}'.format(nii48_snr) + ', {0:.1f}'.format(ha_snr) + ', {0:.1f}'.format(nii84_snr),
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax6.transAxes)
            ax6.axvline(6549.8578, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax6.axvline(6564.613, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax6.axvline(6585.2696, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax6.set_xlabel(r'$\lambda_{rest}$')


            # plotting sii spectrum
            ax7.plot(coadd_spec.wave['brz'] / zFact, coadd_spec.flux['brz'][0] / mw_transmission_spec,
                     color='black', alpha=0.4)
            #ax7.plot(coadd_spec.wave['brz'] / zFact, convolve(coadd_spec.flux['brz'][0], Gaussian1DKernel(5)),
                     #color='k', lw=1.0)
            ax7.plot(modelwave / zFact, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red', linewidth=1)
            ax7.set_xlim([sii_line - sii_limit, sii_line + sii_limit])
            ax7.set_ylim([sii_y_bottom, sii_y_top])
            ax7.text(0.995, 0.975, f'[SII]/[SII]\n' + '{0:.4f}\n'.format(sii_rat) + 'SNR: {0:.1f}'.format(sii_snr_16) + ', {0:.1f}'.format(sii_snr_31),
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax7.transAxes)
            ax7.axvline(6718.2913, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax7.axvline(6732.6705, linestyle='dashed', lw = 0.8, alpha=0.4)
            ax7.set_xlabel(r'$\lambda_{rest}$')
            """
            try:
                plt.savefig(f'{foldstruct}spectrum_{self.targetid}.png', dpi=800)
            except FileNotFoundError:
                os.mkdir(fold)
            """
            plt.savefig(f'{foldstruct}spectrum_{self.targetid}.png', dpi=800)
            if display_plot:
                plt.show()

    def fetch_models(self):
        # This is a test function that is not actually used.
        # Remove this once the spectrum display tool is fully functional
        zcat_sel = self.zpix_cat[self.zpix_cat['TARGETID'] == self.targetid]
        hpx_col = zcat_sel['HEALPIX']
        healpix = np.unique(hpx_col)
        if len(healpix) == 1: # if there's only one healpix number (which should be the case)
            healpix = healpix[0]
        else: # if there's more than one, print the numbers and exit with error code
            print(f"healpix: {healpix}")
            return 2
        print(f'healpix num: {healpix}')
        specfile = f'{self.healpix_dir}/{self.survey}/{self.program}/{healpix // 100}/{healpix}/coadd-{self.survey}-{self.program}-{healpix}.fits'
        fastfile = self.fastspec_dir + f'/healpix/{self.survey}/{self.program}/{healpix // 100}/{healpix}/fastspec-{self.survey}-{self.program}-{healpix}.fits.gz'


        meta = Table(fitsio.read(fastfile, 'METADATA'))
        fast = Table(fitsio.read(fastfile, 'FASTSPEC'))

        models, hdr = fitsio.read(fastfile, 'MODELS', header=True)

        models = models[meta['TARGETID'] == self.targetid]
        print(len(models))

        print(f'models len: {len(models)}')
        print(f'hdr len: {len(hdr)}')
        modelwave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1']) * hdr['CDELT1']

        print(f'modelwave len: {len(modelwave)}')
        print(modelwave)

        #spec = read_spectra(specfile).select(targets=meta['TARGETID'])
        spec = read_spectra(specfile).select(targets=self.targetid)
        coadd_spec = coadd_cameras(spec)
        bands = coadd_spec.bands[0]

        #print(meta.columns)



        mw_transmission_spec = dust_transmission(coadd_spec.wave[bands], meta['EBV'][meta['TARGETID'] == self.targetid])


        #print(coadd_spec.flux[bands].flatten())
        #print(len(coadd_spec.flux[bands][meta['TARGETID'] == self.targetid]))
        #print(coadd_spec.flux[bands][meta['TARGETID'] == self.targetid])

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(coadd_spec.wave[bands], coadd_spec.flux[bands].flatten() / mw_transmission_spec,
                 color='gray', alpha=0.7, label='Data')
        ax1.plot(modelwave, models[0, 0, :], label='Stellar Continuum Model', ls='-', color='blue')
        ax1.plot(modelwave, models[0, 1, :], label='Smooth Continuum Correction', ls='--', color='k')
        ax1.set_ylim(-2.5, 7.5)
        ax1.legend(fontsize=8, loc='upper right')

        ax2.plot(coadd_spec.wave[bands], coadd_spec.flux[bands].flatten() / mw_transmission_spec,
                 color='gray', alpha=0.7, label='Data')
        ax2.plot(modelwave, np.sum(models, axis=1).flatten(), label='Final Model', ls='-', color='red')
        ax2.legend(fontsize=8, loc='upper left')
        ax2.set_xlabel(r'Observed-frame Wavelength ($\AA$)')

        fig.subplots_adjust(hspace=0.05, top=0.95, right=0.95)
        fig.text(0.05, 0.5, r'Flux Density ($10^{-17}~{\rm erg}~{\rm s}^{-1}~{\rm cm}^{-2}~\AA^{-1}$)',
                 ha='center', va='center', rotation='vertical')

        plt.show()


    def check_for_files(self):
        zcat_sel = self.zpix_cat[self.zpix_cat['TARGETID'] == self.targetid]

        survey_col = zcat_sel['SURVEY'].astype(str)
        program_col = zcat_sel['PROGRAM'].astype(str)
        hpx_col = zcat_sel['HEALPIX']

        spectra_path_prefix = self.healpix_dir
        fsf_path_prefix = self.fastspec_dir + f'/healpix'
        local_path_prefix_list = [spectra_path_prefix, fsf_path_prefix]

        spectra_web_path_prefix = f'/public/edr/spectro/redux/{self.specprod}/healpix'
        fsf_web_path_prefix = f'/public/edr/vac/edr/fastspecfit/{self.specprod}/v3.2/healpix'
        web_path_prefix_list = [spectra_web_path_prefix, fsf_web_path_prefix]

        for local_path_prefix, web_path_prefix in zip(local_path_prefix_list, web_path_prefix_list):
            for survey, program, healpix in zip(survey_col, program_col, hpx_col):
                local_path = local_path_prefix + f'/{survey}/{program}/{healpix // 100}/{healpix}'
                web_path = web_path_prefix + f'/{survey}/{program}/{healpix // 100}/{healpix}'

                try:  # try to get the path to the hash file
                    hashfile_path = glob.glob(local_path + '/*.sha256sum')[0]
                except IndexError:  # if it does not exist, download it and get the path
                    print(f"hash file for {survey}/{program}/{healpix} not found, downloading hash file...")
                    #print(f'wget -q -r -nH --no-parent -e robots=off --reject="index.html*" -A.sha256sum --directory-prefix={self.my_dir} https://data.desi.lbl.gov{web_path}/')
                    print(f"downloading data from https://data.desi.lbl.gov{web_path}/")
                    os.system(f'wget -q -r -nH --no-parent -e robots=off --reject="index.html*" -A.sha256sum --directory-prefix={self.my_dir} https://data.desi.lbl.gov{web_path}/')
                    hashfile_path = glob.glob(local_path + '/*.sha256sum')[0]
                    print("hash file successfully downloaded.")

                df = pd.read_csv(hashfile_path, sep='\s+', header=None)
                file_names = df[1]
                hashes = df[0]

                for file_name, hash in zip(file_names, hashes):
                    #print(file_name, hash)
                    file_path = local_path + '/' + file_name
                    file_exists = False
                    hash_good = False
                    fail_counter = 0

                    while not file_exists or not hash_good:
                        if fail_counter > 3:
                            print("failed to download and successfully verify file. the file may be corrupted on the server. ending session.")
                            return 1
                        file_exists = os.path.isfile(file_path)
                        if file_exists:
                            print(f"{file_name} exists. verifying...", end=" ")
                            hashed_file = self.hash_file(file_path)
                            hash_good = hashed_file.hexdigest() == hash
                            if hash_good:
                                print("file is good.")
                            if not hash_good:
                                print("the file could not be verified. delete file and redownload? (y/N)")
                                accept = str(input())
                                if accept == 'y' or accept == 'Y':
                                    os.remove(file_path)
                                else:
                                    print("ending session.")
                                    return 1
                        if not file_exists:
                            print(f"{file_name} does not exist on this machine. Downloading file...", end=" ")
                            os.system(f'wget -r -q -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={self.my_dir} https://data.desi.lbl.gov{web_path}/' + file_name)
                            print("download complete. verifying...", end=" ")
                            hashed_file = self.hash_file(file_path)
                            hash_good = hashed_file.hexdigest() == hash
                            if hash_good:
                                print("file is good.")
                                file_exists = True
                            else:
                                print("the file could not be verified. something may have gone wrong with the download. delete file and try again? (y/N)")
                                accept = str(input())
                                if accept == 'y' or accept == 'Y':
                                    os.remove(file_path)
                                else:
                                    print("ending session.")
                                    return 1
                                print(f"removing {file_path}")
                                fail_counter += 1

        return 0


    def hash_file(self, file_path):
        BUF_SIZE = 65536

        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        return sha256

def spec_plot():
    #targetid = 39627806480531653
    #targetid = 39627746007056970

    #with open('possible_agn_tids.txt', 'r') as f:
    #    tid_list = f.readlines()
    #for i in range(len(tid_list)//50):
    #    tid = int(tid_list[i])
    #    spec = Spectra(targetid=tid)
    #    spec.check_for_files()
    #    spec.plot_spectrum(foldstruct="spectra/possible_agn/")

    spec = Spectrum(targetid=39627734053293709)
    spec.check_for_files()
    spec.plot_spectrum()  # this makes the plot and saves it
    #spec.gen_qa_fig()
    #spec.fetch_models()

def cigale_failed_target_plot():
    from unfit_tids import tids

    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
    specprod = 'fuji'
    specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'

    zpix = Table.read(f'{specprod_dir}/zcatalog/zall-pix-{specprod}.fits', hdu="ZCATALOG")


    for tid in tids[:100]:
        spec = Spectrum(targetid=tid, zpix_cat=zpix)
        spec.check_for_files()
        spec.plot_spectrum(foldstruct="spectra/cigale_failed/")


def main():
    #make_plots()
    spec_plot()
    #cigale_failed_target_plot()



if __name__ == '__main__':
    main()




