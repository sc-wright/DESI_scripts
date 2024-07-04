import os
import numpy as np

import fitsio

from astropy.io import fits
from astropy.table import Table, vstack, join
from astropy.convolution import convolve, Gaussian1DKernel

import matplotlib
import matplotlib.pyplot as plt

#-- input/output functions related to DESI spectra
import desispec.io

from desitarget.sv3.sv3_targetmask import desi_mask, bgs_mask, scnd_mask  # SV3

my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
specprod = 'fuji'
specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'
fsfCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'
lssCatalogsDir = f'{my_dir}/public/edr/vac/edr/lss/v2.0/LSScats/full'
ds9CatalogsDir = f'{my_dir}/public/edr/vac/edr/lsdr9-photometry/fuji/v2.1/observed-targets'

#fsfData = Table.read(f'{fsfCatalogsDir}/fastspec-fuji.fits', hdu=1)
fsfMeta = Table.read(f'{fsfCatalogsDir}/fastspec-fuji.fits', hdu=2)

def generate_combined_mask(masks):
    """
    Creates a new boolean array by combining every array in the masks list using 'and' logic

    :param masks: list: a list with at least one element. Each element is a boolean array of equal length
    :return: A single boolean array that is the 'and' logical combination of all the input arrays
    """
    # masks is a list with at least one element. Each element is a boolean array of equal length
    length = len(masks[0])
    full_mask = np.ones(length, dtype=bool)
    for mask in masks:
        full_mask = np.logical_and(full_mask, mask)
    return full_mask

full_mask = generate_combined_mask([fsfMeta['Z'] > 0.002,
                                    fsfMeta['Z'] < 0.025,
                                    fsfMeta['RA'] > 100,
                                    fsfMeta['RA'] < 250,
                                    fsfMeta['DEC'] > -7,
                                    fsfMeta['DEC'] < 70,
                                    fsfMeta['SURVEY'] == "sv3"])

print(sum(full_mask))

#tid = fsfMeta['TARGETID'][full_mask]
#with open('kim_edr_sample.txt', 'w') as f:
#    for i in tid:
#        f.write(f"{i}\n")