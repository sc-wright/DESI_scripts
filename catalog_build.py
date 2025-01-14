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
        except FileNotFoundError:
            t = input("Custom catalog not found. generate new catalog? (Y/n): ")
            # t = 'y'  # for testing, we'll leave this
            if t == 'y' or t == 'Y':
                self.generate_new_table()

        self.bgs_mask = generate_combined_mask(self.catalog['ZWARN'] == 0, self.catalog['Z'] <= 0.4)

        print(self.catalog[bgs_mask])
        print(len(self.catalog['TARGETID']))
        print(len(self.catalog['TARGETID'][bgs_mask]))

    def generate_new_table(self):
        """
        generates new table by opening other catalogs, matching targets, and reconstructing new data structures
        :return: catalog object
        """

        self.catalog = Table()

        self.read_FSF_catalog()
        self.read_LSS_catalog()
        self.read_DR9_catalog()

        self.catalog.write(f'{self.ccat_dir}/custom-bgs-fuji-cat.fits')



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
                       'OII_3726_MODELAMP', 'OII_3726_AMP', 'OII_3726_AMP_IVAR', 'OII_3726_FLUX', 'OII_3726_FLUX_IVAR',
                       'OII_3729_MODELAMP', 'OII_3729_AMP', 'OII_3729_AMP_IVAR', 'OII_3729_FLUX', 'OII_3729_FLUX_IVAR',
                       'HBETA_MODELAMP', 'HBETA_AMP', 'HBETA_AMP_IVAR', 'HBETA_FLUX', 'HBETA_FLUX_IVAR',
                       'OIII_5007_MODELAMP', 'OIII_5007_AMP', 'OIII_5007_AMP_IVAR', 'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR',
                       'NII_6548_MODELAMP', 'NII_6548_AMP', 'NII_6548_AMP_IVAR', 'NII_6548_FLUX', 'NII_6548_FLUX_IVAR',
                       'HALPHA_MODELAMP', 'HALPHA_AMP', 'HALPHA_AMP_IVAR', 'HALPHA_FLUX', 'HALPHA_FLUX_IVAR',
                       'NII_6584_MODELAMP', 'NII_6584_AMP', 'NII_6584_AMP_IVAR', 'NII_6584_FLUX', 'NII_6584_FLUX_IVAR',
                       'SII_6716_MODELAMP', 'SII_6716_AMP', 'SII_6716_AMP_IVAR', 'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                       'SII_6731_MODELAMP', 'SII_6731_AMP', 'SII_6731_AMP_IVAR', 'SII_6731_FLUX', 'SII_6731_FLUX_IVAR'
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



        print(len(dr9Catalog['TARGETID']), len(np.unique(dr9Catalog['TARGETID'])))

        column_list = ['MASKBITS', 'RELEASE', 'RA', 'RA_IVAR', 'DEC', 'DEC_IVAR',
                       'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z',
                       'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
                       'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4',
                       'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3', 'FLUX_IVAR_W4',
                       'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4',
                       'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z'
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
        print(len(self.catalog['TARGETID']))
        print(len(joined_table['TARGETID']))

        # and add the desired new columns
        for col in column_list:
            self.catalog[col] = joined_table[col]


def main():
    testcat = customCatalog()

if __name__ == '__main__':
    main()