import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['text.usetex'] = True

import numpy as np

import astropy
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import pandas as pd

from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from spectrum_plot import Spectrum

import time


class CustomCatalog:
    def __init__(self):
        self.ccat_dir = os.path.expanduser('~') + '/Documents/school/research/customcatalog'

        try:
            # reads in the entire custom catalog
            self.catalog = Table.read(f'{self.ccat_dir}/custom-bgs-fuji-cat.fits')

            # If you need to run any scripts again for testing or output purposes, put them below.
            # This prevents having to recalculate the whole file for small tests.

            #self.correct_ha_full()

        except FileNotFoundError:
            #t = input("Custom catalog not found. Generate new catalog? (Y/n): ")
            t = 'y'  # for testing, we'll leave this
            if t == 'y' or t == 'Y':
                self.generate_new_table()
            else:
                return 4

        self.bgs_mask = generate_combined_mask(self.catalog['ZWARN'] == 0, self.catalog['Z'] <= 0.4)

        #print(self.catalog[self.bgs_mask])
        #print(len(self.catalog['TARGETID']))
        #print(len(self.catalog['TARGETID'][self.bgs_mask]))

    def generate_new_table(self):
        """
        generates new table by opening other catalogs, matching targets, and reconstructing new data structures
        :return: catalog object
        """

        self.catalog = Table()

        self.read_FSF_catalog()
        self.read_LSS_catalog()
        self.read_DR9_catalog()
        self.read_aperture_catalog()
        # The sga catalog is no longer being used in this project.
        #self.read_SGA_catalog()
        #return 0


        self.add_calculated_values()

        self.catalog.write(f'{self.ccat_dir}/custom-bgs-fuji-cat.fits')

    def add_calculated_values(self):
        #self.balmer_correction()
        #quit()

        print("calculating oii luminosity...")
        self.catalog['OII_LUMINOSITY'] = self.calc_oii_luminosity()

        print("calculating mstar from wise color...")
        self.catalog['MSTAR_WISE'] = self.add_WISE_mstars()

        print("calculating sfr from h alpha...")
        # First do balmer correction (record extinction)
        sfr_ebv = self.balmer_correction()
        #quit()
        self.catalog['HALPHA_BALMER'] = sfr_ebv[0]
        self.catalog['EBV'] = sfr_ebv[1]
        self.catalog['A_HALPHA'] = sfr_ebv[2]
        # Then do aperture correction
        self.catalog['HALPHA_BALMER_APERTURE'] = self.aperture_correct_ha()
        # Now calculate SFR using Balmer- and aperture-corrected Halpha
        self.catalog['SFR_HALPHA'] = self.add_full_corrected_sfr()
        # And calculate SFR using only Balmer-corrected Halpha
        self.catalog['SFR_APERTURE'] = self.add_aperture_corrected_SFR()
        # Finally calculate SFR surface density using only Balmer-corrected Halpha SFR
        self.catalog['SFR_SD'] = self.add_sfr_surface_density()

        print("parsing and adding cigale values...")
        cig = self.parse_cigale_results()
        self.catalog['MSTAR_CIGALE'] = cig[1]
        self.catalog['SFR_CIGALE'] = cig[0]

        print("calculating electron densities...")
        self.catalog['NE_OII'] = self.add_electron_density_oii()
        self.catalog['NE_SII'] = self.add_electron_density_sii()

        print("adding metallicity...")
        self.catalog['METALLICITY_O3N2'] = self.add_metallicity()

        print("done.")

    def read_FSF_catalog(self):

        # First open the two hdus of the catalog in two new objects

        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        specprod = 'fuji'
        specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'
        fsfCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'

        #fsfMeta = Table.read(f'{fsfCatalogsDir}/fastspec-fuji.fits', hdu=2)
        fsfCatalog = Table.read(f'{fsfCatalogsDir}/fastspec-fuji-processed.fits')

        # Create a mask that can be placed over this catalog.
        # When adding a new column from this catalog, you only need to apply this mask before adding.
        duplicate_mask = generate_combined_mask(fsfCatalog['SURVEY'] == 'sv3', fsfCatalog['ISPRIMARY'])

        column_list = ['TARGETID', 'Z', 'OII_DOUBLET_RATIO', 'SII_DOUBLET_RATIO',
                       'OII_3726_MODELAMP', 'OII_3726_AMP', 'OII_3726_AMP_IVAR',
                       'OII_3726_FLUX', 'OII_3726_FLUX_IVAR', 'OII_3726_NPIX',
                       'OII_3729_MODELAMP', 'OII_3729_AMP', 'OII_3729_AMP_IVAR',
                       'OII_3729_FLUX', 'OII_3729_FLUX_IVAR', 'OII_3729_NPIX',
                       'HBETA_MODELAMP', 'HBETA_AMP', 'HBETA_AMP_IVAR', 'HBETA_FLUX', 'HBETA_FLUX_IVAR',
                       'OIII_4959_MODELAMP', 'OIII_4959_AMP', 'OIII_4959_AMP_IVAR', 'OIII_4959_FLUX', 'OIII_4959_FLUX_IVAR',
                       'OIII_5007_MODELAMP', 'OIII_5007_AMP', 'OIII_5007_AMP_IVAR', 'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR',
                       'NII_6548_MODELAMP', 'NII_6548_AMP', 'NII_6548_AMP_IVAR', 'NII_6548_FLUX', 'NII_6548_FLUX_IVAR',
                       'HALPHA_MODELAMP', 'HALPHA_AMP', 'HALPHA_AMP_IVAR', 'HALPHA_FLUX', 'HALPHA_FLUX_IVAR',
                       'NII_6584_MODELAMP', 'NII_6584_AMP', 'NII_6584_AMP_IVAR', 'NII_6584_FLUX', 'NII_6584_FLUX_IVAR',
                       'SII_6716_MODELAMP', 'SII_6716_AMP', 'SII_6716_AMP_IVAR', 'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                       'SII_6731_MODELAMP', 'SII_6731_AMP', 'SII_6731_AMP_IVAR', 'SII_6731_FLUX', 'SII_6731_FLUX_IVAR',
                       'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_R', 'ABSMAG01_SDSS_Z',
                       'KCORR01_SDSS_G', 'KCORR01_SDSS_R', 'KCORR01_SDSS_Z'
                       ]

        for col in column_list:
            self.catalog[col] = fsfCatalog[col][duplicate_mask]

    def read_LSS_catalog(self):
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        lssCatalogsDir = f'{my_dir}/public/edr/vac/edr/lss/v2.0/LSScats/full'

        # reads in the entire BGS_ANY catalog. Quality cuts are already implemented.
        lssCatalog = Table.read(f'{lssCatalogsDir}/BGS_ANY_full.dat.fits')

        # The TARGETIDs in the LSS catalog are completely unique.
        # print(len(lssCatalog['TARGETID']), len(np.unique(lssCatalog['TARGETID'])))

        joined_table = astropy.table.join(self.catalog, lssCatalog, 'TARGETID', 'left')

        self.catalog['ZWARN'] = joined_table['ZWARN']

        #print(len(self.catalog['TARGETID']), len(self.catalog['TARGETID'][self.catalog['ZWARN'] == 0]))

    def read_DR9_catalog(self):
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        specprod = 'fuji'
        dr9CatalogsDir = f'{my_dir}/public/edr/vac/edr/lsdr9-photometry/{specprod}/v2.1/observed-targets'

        # reads in the entire edr catalog
        dr9Catalog = Table.read(f'{dr9CatalogsDir}/targetphot-sv3-{specprod}.fits')

        #print(len(dr9Catalog['TARGETID']), len(np.unique(dr9Catalog['TARGETID'])))

        column_list = ['MASKBITS', 'RELEASE', 'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR', 'EBV', 'REF_CAT',
                       'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z',
                       'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
                       'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4',
                       'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3', 'FLUX_IVAR_W4',
                       'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4',
                       'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z', 'SHAPE_R', 'RELEASE', 'BRICKNAME'
                       ]

        """
        # This checks whether any of the duplicate targetids have different values in any of the columns we are taking.
        # They do not, so we will not be concerned with duplicates and can simply take the first appearance of the target
        # This does not need to be run every time, but can be uncommented to check again.

        # Convert the Astropy table to a pandas DataFrame
        names = [name for name in dr9Catalog.colnames if len(dr9Catalog[name].shape) <= 1]
        df = dr9Catalog[names].to_pandas()
        #df = dr9Catalog.to_pandas()

        # Find duplicate TARGETIDs
        duplicate_groups = df[df.duplicated(subset=['TARGETID'], keep=False)]

        # Group by TARGETID
        grouped = duplicate_groups.groupby('TARGETID')
        print("starting test")

        # Iterate through each group
        for target_id, group in grouped:
            for column in column_list:
                # Check if all values in the column are identical
                if group[column].nunique() > 1:
                    print(f"TARGETID {target_id} has differing values in column '{column}'.")

        return 0
        """

        # Removing duplicates to prep for joining tables

        # Identify the indices of the first occurrence of each TARGETID
        unique_indices = []
        seen = set()

        for i, targetid in enumerate(dr9Catalog['TARGETID']):
            if targetid not in seen:
                unique_indices.append(i)
                seen.add(targetid)

        dr9Catalog_unique = dr9Catalog[unique_indices]

        # Now join the tables
        joined_table = astropy.table.join(self.catalog, dr9Catalog_unique, 'TARGETID', 'left')
        #print(len(self.catalog['TARGETID']))
        #print(len(joined_table['TARGETID']))

        # and add the desired new columns
        for col in column_list:
            self.catalog[col] = joined_table[col]


    def read_aperture_catalog(self):
        print("working on aperture flux matching...")
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        apertureCatalogsDir = f'{my_dir}/otherdata/legacysurvey'

        apertureCatalog = Table.read(f'{apertureCatalogsDir}/apflux_data.fits')

        column_list = ['APFLUX_G', 'APFLUX_R', 'APFLUX_Z', 'APFLUX_IVAR_G', 'APFLUX_IVAR_R', 'APFLUX_IVAR_Z']

        joined_table = astropy.table.join(self.catalog, apertureCatalog, 'TARGETID', 'left')

        for col in column_list:
            self.catalog[col] = joined_table[col]

    def read_SGA_catalog(self):

        # This is deprecated - we are no longer using the SGA catalog for anything in this project

        print("working on sga matching...")
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        lssCatalogsDir = f'{my_dir}/otherdata/sga2020'

        # reads in the entire BGS_ANY catalog. Quality cuts are already implemented.
        sgaCatalog = Table.read(f'{lssCatalogsDir}/SGA-2020.fits', hdu=2)

        # These are all the ra and decs from the sga
        sga_ra = sgaCatalog['RA']
        sga_dec = sgaCatalog['DEC']

        # These are all the sources in the custom catalog that are also in sga
        sga_mask = generate_combined_mask(self.catalog['REF_CAT'] == 'L3', self.catalog['ZWARN'] == 0)
        snr_lim = 5
        ha_snr_mask = self.catalog['HALPHA_AMP'] * np.sqrt(self.catalog['HALPHA_AMP_IVAR']) > snr_lim
        hb_snr_mask = self.catalog['HBETA_AMP'] * np.sqrt(self.catalog['HBETA_AMP_IVAR']) > snr_lim
        snr_mask = generate_combined_mask(ha_snr_mask, hb_snr_mask, self.catalog['ZWARN'] == 0, self.catalog['Z'] <= 0.4)
        full_mask = generate_combined_mask(sga_mask, snr_mask)
        cc_ra = self.catalog['RA'][full_mask]
        cc_dec = self.catalog['DEC'][full_mask]

        print(len(self.catalog['ZWARN'] == 0))
        print(len(cc_ra))

        matches = np.where(
            (sgaCatalog['RA'][:, None] == cc_ra) &
            (sgaCatalog['DEC'][:, None] == cc_dec)
        )

        # Extract the row indices from the table that match
        matching_indices = matches[0]

        print(matching_indices)

        sga_ra = sgaCatalog['RA'][matching_indices]
        sga_dec = sgaCatalog['DEC'][matching_indices]
        sga_apflux_g = sgaCatalog['APFLUX_G'][matching_indices]
        sga_apflux_r = sgaCatalog['APFLUX_R'][matching_indices]
        sga_apflux_z = sgaCatalog['APFLUX_Z'][matching_indices]

        matches = np.where(
            (self.catalog['RA'][:, None] == sga_ra) &
            (self.catalog['DEC'][:, None] == sga_dec)
        )

        matching_indices = matches[0]

        length = len(self.catalog['TARGETID'])
        cc_apflux_g = np.array([np.zeros(8) for _ in range(length)])
        cc_apflux_r = np.array([np.zeros(8) for _ in range(length)])
        cc_apflux_z = np.array([np.zeros(8) for _ in range(length)])

        cc_apflux_g[matching_indices] = sga_apflux_g
        cc_apflux_r[matching_indices] = sga_apflux_r
        cc_apflux_z[matching_indices] = sga_apflux_z

        self.catalog['APFLUX_G'] = cc_apflux_g
        self.catalog['APFLUX_R'] = cc_apflux_r
        self.catalog['APFLUX_Z'] = cc_apflux_z

    def calc_oii_luminosity(self):
        # table should be the fsf table


        oII6Flux = np.array(self.catalog['OII_3726_FLUX'])
        oII9Flux = np.array(self.catalog['OII_3729_FLUX'])
        oIICombinedFlux = oII6Flux + oII9Flux
        redshift = np.array(self.catalog['Z'])
        npix = np.array(self.catalog['OII_3726_NPIX']) + np.array(self.catalog['OII_3729_NPIX'])
        dataLength = len(oIICombinedFlux)
        oii_luminosity = np.zeros(dataLength)

        c = CustomTimer(dataLength, calcstr="OII luminosity")

        for i in range(dataLength):
            if npix[i] > 1:
                flux = oIICombinedFlux[i]
                if flux > 0:
                    oIILum = np.log10(get_lum(flux, redshift[i]))
                    oii_luminosity[i] = oIILum

            c.update_time(i)

        return oii_luminosity

    def calc_mstar_WISE_color(self, w1, w2, z):
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

        # Calculate absolute magnitudes using luminosity distance
        # convert D_l to pc
        abs_mag_w1_AB = w1_mag_AB - 5 * np.log10(D_l * 1e6 / 10)  # - 2.5*np.log10(1.2) # last term is for debugging
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

    def add_WISE_mstars(self):

        tids = self.catalog['TARGETID']
        fw1 = self.catalog['FLUX_W1'] / self.catalog['MW_TRANSMISSION_W1']
        fw2 = self.catalog['FLUX_W2'] / self.catalog['MW_TRANSMISSION_W2']
        redshifts = self.catalog['Z']
        mstar_array = np.ones(len(tids)) * -1

        for i, (tid, flux1, flux2, z) in enumerate(zip(tids, fw1, fw2, redshifts)):
            # There are some sources without good redshift fits, with z=-1.
            # Those sources will not have stellar masses calculated, leaving mstar as -1
            if z >= 0:
                mstar_array[i] = self.calc_mstar_WISE_color(flux1, flux2, z)

        mstar_array[np.where(np.isnan(mstar_array))] = np.nan
        mstar_array[np.where(np.isinf(mstar_array))] = np.nan

        return mstar_array

    def add_electron_density_oii(self):
        """
        This calculation comes from the formula determined in Sanders+16
        :return:
        """
        a = 0.3771
        b = 2468
        c = 638.4

        R = 1 / self.catalog['OII_DOUBLET_RATIO']
        R[np.where(R < 0.3839)] = np.nan
        R[np.where(R > 1.4558)] = np.nan
        n = (c * R - a * b) / (a - R)

        return n

    def add_electron_density_sii(self):

        a = 0.4315
        b = 2107
        c = 627.1

        R = 1 / self.catalog['SII_DOUBLET_RATIO']
        R[np.where(R < 0.4375)] = np.nan
        R[np.where(R > 1.4484)] = np.nan
        n = (c * R - a * b) / (a - R)
        return n

    def k_lambda_2001(self, wavelength):
        # From
        # Wavelength is in angstroms - convert to microns
        wl = wavelength * 1e-4

        if wl <= 2.2000 and wl > .6300:
            k = 1.17 * (-1.1857 + (1.040 / wl)) + 1.78
        elif wl >= .1200:
            k = 1.17 * (-2.156 + (1.509 / wl) - (0.198 / wl ** 2) + (0.011 / wl ** 3)) + 1.78
        else:
            print(wavelength, "outside wavelength range")
            return 0

        return k

    def k_lambda_2000(self, wavelength):
        # From
        # Wavelength is in angstroms - convert to microns
        wl = wavelength * 1e-4

        if wl <= 2.2000 and wl > .6300:
            k = 2.659 * (-1.857 + (1.040 / wl)) + 4.05
        elif wl >= .1200:
            k = 2.659 * (-2.156 + (1.509 / wl) - (0.198 / (wl ** 2)) + (0.011 / (wl ** 3))) + 4.05
        else:
            print(wavelength, "outside wavelength range")
            return 0

        return k

    def balmer_correction(self):
        # We calculate the extinction corrected value for every source. We can then filter by SNR later as we choose

        # Calculate extinction-corrected H-alpha using H-beta
        E_beta_alpha = 2.5 * np.log10(2.86 / (self.catalog['HALPHA_FLUX'] / self.catalog['HBETA_FLUX']))
        EBV = E_beta_alpha / (self.k_lambda_2000(6563) - self.k_lambda_2000(4861))
        EBV_s = EBV * 0.44  # this comes from Calzetti+2000
        A_halpha = self.k_lambda_2000(6563) * EBV_s
        #print(sum(np.array(A_halpha < 0, dtype=bool)))
        print(A_halpha)

        # It should be unphysical for A(Ha) to be less than 0
        # We go ahead and turn anything less than 0 into zero, making the balmer correction 1
        bad_a = np.where(A_halpha < 0)
        A_halpha[bad_a] = 0

        # There are also cases where the value could not be calculated (usually due to zeros in the catalog)
        # We remove those here and replace them with nans.
        EBV_s[np.where(np.isnan(EBV_s))] = np.nan
        EBV_s[np.where(np.isinf(EBV_s))] = np.nan

        A_halpha[np.where(np.isnan(A_halpha))] = np.nan
        A_halpha[np.where(np.isinf(A_halpha))] = np.nan

        # Finally calculate the correction and apply it
        correction = 10 ** (0.4 * A_halpha)
        ha_balmer_corrected = self.catalog['HALPHA_FLUX'] * correction

        return ha_balmer_corrected, EBV_s, A_halpha


    def aperture_correct_ha(self):

        tids = self.catalog['TARGETID']

        # These tids all have 0 r or g flux inside their fibers, so they are excluded.
        badtids = [39627811886993159, 39627811975070271, 39632941231374432, 39632951331261223, 39632981840625924,
                   39633290046473376, 39633301228490393, 39633315501703526, 39633315510095801, 39633425245670472,
                   39633421902807811, 39633453863405816]

        bgs_mask = generate_combined_mask(self.catalog['ZWARN'] == 0, self.catalog['SHAPE_R'] * 1.5 < 7, ~np.isin(tids, badtids))

        # We calculate the extinction corrected value for every source. We can then filter by SNR later as we choose

        ha_balmer_corrected = self.catalog['HALPHA_BALMER']
        # ha balmer corrected has full catalog length

        ha_balmer_corrected = ha_balmer_corrected[bgs_mask]
        # now ha balmer corrected has bgs mask length

        # Now we find the relationship between ha flux ratio and color and fit for a slope.
        r_ap_fluxes = self.catalog['APFLUX_G'][bgs_mask]
        g_ap_fluxes = self.catalog['APFLUX_R'][bgs_mask]

        # These all have length from applied mask
        r_fiber_flux = np.zeros(len(r_ap_fluxes))
        g_fiber_flux = np.zeros(len(r_ap_fluxes))
        r_total_flux = np.zeros(len(r_ap_fluxes))
        g_total_flux = np.zeros(len(r_ap_fluxes))

        # This will be used to deal with small objects that get unusual aperture corrections
        small_radius_flag = np.full(len(r_ap_fluxes), False)

        rad_ind = self.get_radius_indices(bgs_mask)

        # loop over all the r and g fluxes
        for i in range(len(r_ap_fluxes)):
            # The flux in the desi-aperture
            r_fiber_flux[i] = r_ap_fluxes[i][1]
            g_fiber_flux[i] = g_ap_fluxes[i][1]
            # The flux in the bigger aperture
            r_total_flux[i] = r_ap_fluxes[i][rad_ind[i]]
            g_total_flux[i] = g_ap_fluxes[i][rad_ind[i]]
            # if the bigger aperture is aperture 1 or 0, flag it as a small aperture.
            # these must necessarily have an aperture correction of 1
            if rad_ind[i] <= 1:
                small_radius_flag[i] = True

        # Convert all to magnitudes and calculate colors
        r_fiber_mag = -2.5 * np.log10(r_fiber_flux)
        g_fiber_mag = -2.5 * np.log10(g_fiber_flux)
        gr_fiber_color = g_fiber_mag - r_fiber_mag  # This is a list of every color element
        r_total_mag = -2.5 * np.log10(r_total_flux)
        g_total_mag = -2.5 * np.log10(g_total_flux)
        gr_total_color = g_total_mag - r_total_mag  # This is a list of every color element

        # Calculate ratios - r-band aperture ratio and corrected ha flux over r-band fiber flux
        # these are both mask length
        r_band_aperture_ratio = r_total_flux / r_fiber_flux
        ha_over_r_ratio = ha_balmer_corrected / r_fiber_flux

        # We definitely want to apply an snr cut before doing this fit. A high cut makes sense. Will use 5 to start
        # There are four values used in this fit: r and g fiber fluxes and ha and hb fluxes
        # The fiber fluxes have no _IVAR. For now we will cut only on Ha and Hb fluxes

        # each of these masks are catalog length
        snr_lim = 5
        ha_snr_mask = self.catalog['HALPHA_AMP'] * np.sqrt(self.catalog['HALPHA_AMP_IVAR']) > snr_lim
        hb_snr_mask = self.catalog['HBETA_AMP'] * np.sqrt(self.catalog['HBETA_AMP_IVAR']) > snr_lim

        zmask = generate_combined_mask(self.catalog['Z'] > 0.02, self.catalog['Z'] < 0.1)
        # we then make them all mask length and then combine them to become snr length.
        # snr mask gets applied to objects that are mask length
        snr_mask = generate_combined_mask(ha_snr_mask[bgs_mask], hb_snr_mask[bgs_mask], zmask[bgs_mask])

        # SNR mask is placed over both arrays.
        linear_fit = np.polyfit(gr_fiber_color[snr_mask], np.log10(ha_over_r_ratio[snr_mask]), 1)
        slope = linear_fit[0]
        print(f"slope: {slope}")

        # This is the final aperture correction including both aperture and color gradient
        ha_final_correction = r_band_aperture_ratio * 10 ** (slope * (gr_total_color - gr_fiber_color))

        # If 1.5*radius is the same size or smaller than the desi aperture,
        # the aperture correction should necessarily be one.
        # We force that here.
        ha_final_correction[small_radius_flag] = 1

        # Apply the color-correction to the already-balmer-corrected flux
        ha_flux_corrected = ha_balmer_corrected * ha_final_correction

        ha_flux = np.zeros(len(tids))
        ha_flux[bgs_mask] = ha_flux_corrected

        return ha_flux

    def add_full_corrected_sfr(self):

        h_alpha_flux = self.catalog['HALPHA_BALMER_APERTURE']

        redshifts = self.catalog['Z']

        h_alpha_lum = np.empty(len(h_alpha_flux))
        for i, (flux, z) in enumerate(zip(h_alpha_flux, redshifts)):
            h_alpha_lum[i] = get_lum(flux, z)

        # using the table from Kennicutt 2012
        halpha_sfr_log = np.log10(h_alpha_lum) - 41.27
        # using the method from Kennicutt 1998 (as listed in https://arxiv.org/pdf/2312.00300 sect 3.3)
        halpha_sfr = h_alpha_lum * 7.9E-42

        halpha_sfr_log[np.where(np.isnan(halpha_sfr_log))] = np.nan
        halpha_sfr_log[np.where(np.isinf(halpha_sfr_log))] = np.nan

        return halpha_sfr_log

    def add_aperture_corrected_SFR(self):

        h_alpha_flux = self.catalog['HALPHA_BALMER']

        redshifts = self.catalog['Z']

        h_alpha_lum = np.empty(len(h_alpha_flux))
        for i, (flux, z) in enumerate(zip(h_alpha_flux, redshifts)):
            h_alpha_lum[i] = get_lum(flux, z)

        # using the table from Kennicutt 2012
        halpha_sfr_log = np.log10(h_alpha_lum) - 41.27
        # using the method from Kennicutt 1998 (as listed in https://arxiv.org/pdf/2312.00300 sect 3.3)
        halpha_sfr = h_alpha_lum * 7.9E-42

        halpha_sfr_log[np.where(np.isnan(halpha_sfr_log))] = np.nan
        halpha_sfr_log[np.where(np.isinf(halpha_sfr_log))] = np.nan

        return halpha_sfr_log

    def add_sfr_surface_density(self):
        aperture_sfr = self.catalog['SFR_APERTURE']
        aperture_sfr = 10 ** aperture_sfr
        redshift = self.catalog['Z']
        cosmo = astropy.cosmology.FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
        distance = cosmo.angular_diameter_distance(redshift)  # this angular diameter distance in mpc for 1 radian
        #print(distance.value[:20])
        aperture_radius = 0.75  # in arcsec
        aperture_radius = aperture_radius / (60 * 60 * 360 / (2 * np.pi))  # in radians
        #print(aperture_radius)
        radius_kpc = (aperture_radius * distance.value * 1e3) # convert to kpc, use small angle approx
        #print(aperture_radius, np.tan(aperture_radius))
        #radius_kpc_smangle = (aperture_radius * distance.value * 1000) # convert to pc
        area_pc2 = radius_kpc ** 2 * np.pi
        sfrsd = np.log10(aperture_sfr / area_pc2)
        return sfrsd

    def add_metallicity(self):
        oiii_5007_flux = np.array(self.catalog['OIII_5007_FLUX'])
        oiii_5007_err_inv = np.array(np.sqrt(self.catalog['OIII_5007_FLUX_IVAR']))
        nii_6584_flux = np.array(self.catalog['NII_6584_FLUX'])
        nii_6584_err_inv = np.array(np.sqrt(self.catalog['NII_6584_FLUX_IVAR']))
        halpha_flux = np.array(self.catalog['HALPHA_FLUX'])
        halpha_flux_err_inv = np.array(np.sqrt(self.catalog['HALPHA_FLUX_IVAR']))
        hbeta_flux = np.array(self.catalog['HBETA_FLUX'])
        hbeta_flux_err_inv = np.array(np.sqrt(self.catalog['HBETA_FLUX_IVAR']))

        oiii_5007_snr = oiii_5007_flux * oiii_5007_err_inv
        nii_6584_snr = nii_6584_flux * nii_6584_err_inv
        halpha_snr = halpha_flux * halpha_flux_err_inv
        hbeta_snr = hbeta_flux * hbeta_flux_err_inv

        snr_lim = 3

        # This is just the o3n2 metallicity lines
        metallicity_mask = generate_combined_mask(oiii_5007_snr > snr_lim, nii_6584_snr > snr_lim, halpha_snr > snr_lim,
                                                  hbeta_snr > snr_lim)

        # 03N2 from Pettini & Pagel 2004
        O3N2 = np.log10((oiii_5007_flux / hbeta_flux) / (nii_6584_flux / halpha_flux))

        # From PP04
        o3n2_metallicity = 8.73 - 0.32 * O3N2

        return o3n2_metallicity


    def parse_cigale_results(self):

        cigale_dir = os.path.expanduser('~') + '/Documents/school/research/cigale'
        cigale_results = pd.read_table(f"{cigale_dir}/_full_sky/out/results.txt", header=0, sep='\s+')

        cigale_sfr = {cigale_results['id'][i]: cigale_results['bayes.sfh.sfr'][i] for i in range(len(cigale_results['id']))}
        cigale_mass = {cigale_results['id'][i]: cigale_results['bayes.stellar.m_star'][i] for i in range(len(cigale_results['id']))}

        sfr_array = np.zeros(len(self.catalog['TARGETID']))
        mass_array = np.zeros(len(self.catalog['TARGETID']))

        for i, tid in enumerate(self.catalog['TARGETID']):
            try:
                sfr = np.log10(cigale_sfr[tid])
                mass = np.log10(cigale_mass[tid])
                sfr_array[i] = sfr
                mass_array[i] = mass
            except KeyError:
                pass

        return sfr_array, mass_array

    def read_brickcats(self):
        # Once the file is generated this shouldnt need to be run again unless we need to draw something else from the brickcats.

        mask = self.catalog['ZWARN'] == 0

        tids = self.catalog['TARGETID'][mask]
        release = self.catalog['RELEASE'][mask]
        brickname = self.catalog['BRICKNAME'][mask]
        ra = self.catalog['RA'][mask]
        dec = self.catalog['DEC'][mask]

        length = len(tids)

        apflux_g_list = np.array([np.zeros(8) for _ in range(length)])
        apflux_r_list = np.array([np.zeros(8) for _ in range(length)])
        apflux_z_list = np.array([np.zeros(8) for _ in range(length)])
        apflux_g_ivar_list = np.array([np.zeros(8) for _ in range(length)])
        apflux_r_ivar_list = np.array([np.zeros(8) for _ in range(length)])
        apflux_z_ivar_list = np.array([np.zeros(8) for _ in range(length)])

        c = CustomTimer(len(tids))

        for i, (rel, brick, r, d) in enumerate(zip(release, brickname, ra, dec)):
            apflux_g, apflux_r, apflux_z, apflux_g_ivar, apflux_r_ivar, apflux_z_ivar = fetch_tractor_bricks(rel, brick, r, d)
            apflux_g_list[i] = apflux_g
            apflux_r_list[i] = apflux_r
            apflux_z_list[i] = apflux_z
            apflux_g_ivar_list[i] = apflux_g_ivar
            apflux_r_ivar_list[i] = apflux_r_ivar
            apflux_z_ivar_list[i] = apflux_z_ivar
            c.update_time(i)

        apflux_tab = Table([tids, apflux_g_list, apflux_r_list, apflux_z_list, apflux_g_ivar_list, apflux_r_ivar_list, apflux_z_ivar_list],
                           names=('TARGETID', 'APFLUX_G', 'APFLUX_R', 'APFLUX_Z', 'APFLUX_IVAR_G', 'APFLUX_IVAR_R', 'APFLUX_IVAR_Z'))

        apflux_tab.write(os.path.expanduser('~') + '/Documents/school/research/apflux_data.fits', format='fits')

        #use this later when merging with the table
        #joined_table = astropy.table.join(self.catalog, lssCatalog, 'TARGETID', 'left')
        #self.catalog['ZWARN'] = joined_table['ZWARN']

    def get_radius_indices(self, mask):

        angular_radius = self.catalog['SHAPE_R'][mask]

        apertures = [0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 7.0]

        # Scale re by 3

        # Find the index of the closest radius for each scaled value
        closest_indices = np.array([np.argmin([abs(r - s) for r in apertures]) for s in angular_radius * 3])
        # ul_indices = np.array([next(i for i, r in enumerate(apertures) if r >= s) if any(r >= s for r in apertures) else len(apertures) - 1 for s in angular_radius * 3])

        return closest_indices

def fetch_tractor_bricks(release, brickname, ra, dec):
    # Tool for read_brickcats to use

    local_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'

    dr9_path = '/cfs/cosmo/data/legacysurvey/dr9'

    dr9_web_prefix = 'https://portal.nersc.gov'

    if release in [9010, 9012]:
        hemi = 'south' # CHECK THIS IS CORRECT
    elif release in [9011]:
        hemi = 'north'

    fold = str(brickname[:3])

    brick_path = dr9_path + f'/{hemi}/tractor/{fold}'

    file_path = local_dir + brick_path
    web_path = dr9_web_prefix + brick_path
    file_name = f'tractor-{brickname}.fits'

    file_exists = os.path.isfile(f'{file_path}/{file_name}')

    if not file_exists:
        print(f"{file_name} does not exist on this machine. Downloading file from {web_path}/{file_name}\n", end=" ")
        os.system(
            f'wget -r -q -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={local_dir} {web_path}/' + file_name)

    #print(f'{local_dir}/{brick_path}/{file_name}')
    tab = Table.read(f'{local_dir}{brick_path}/{file_name}')

    point_mask = generate_combined_mask(tab['ra'] == ra, tab['dec'] == dec)
    #if sum(point_mask) != 1:
    #    print(sum(point_mask))

    apflux_g = tab['apflux_g'][point_mask]
    apflux_r = tab['apflux_r'][point_mask]
    apflux_z = tab['apflux_z'][point_mask]
    apflux_g_ivar = tab['apflux_ivar_g'][point_mask]
    apflux_r_ivar = tab['apflux_ivar_r'][point_mask]
    apflux_z_ivar = tab['apflux_ivar_z'][point_mask]

    return apflux_g, apflux_r, apflux_z, apflux_g_ivar, apflux_r_ivar, apflux_z_ivar


def main():
    #fetch_tractor_bricks(9010, "1493p027", 149.32612873128917)
    testcat = CustomCatalog()
    #testcat.read_brickcats()

if __name__ == '__main__':
    main()