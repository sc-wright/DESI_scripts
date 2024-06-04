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


"""
# Release directory path

my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
specprod = 'fuji'
specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'
fastspec_dir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2'

#fujidata = Table( fitsio.read(os.path.join(specprod_dir, "zcatalog", "zall-pix-{}.fits".format(specprod))) )


hdul = fits.open(fastspec_dir + '/fastspec-fuji.fits')
hdul.info()
"""

my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
specprod = 'fuji'
specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'
fsfCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'
lssCatalogsDir = f'{my_dir}/public/edr/vac/edr/lss/v2.0/LSScats/full'
print("reading in fsf table...")
fsfData = Table.read(f'{fsfCatalogsDir}/fastspec-fuji-data-processed.fits')
print("reading in lss table...")
lssData = Table.read(f'{lssCatalogsDir}/BGS_ANY_full.dat.fits')

# this is a list of fully unique tids
bgs_tids = lssData['TARGETID'][lssData['ZWARN'] == 0]

# select the targetids from the fsf catalog that are also in the BGS_ANY lss catalog
bgs_mask = [i in bgs_tids for i in fsfData['TARGETID']]
bgs_mask = np.logical_and(bgs_mask, fsfData['OII_COMBINED_LUMINOSITY_LOG'] > 0)


#bgs_mask = fsfData['TARGETID'] in bgs_tids

# not all the ones marked 'ISBGS' are included here - presumably these all got vetoed or had zwarn > 0
masked_fsf = fsfData['ISBGS'][bgs_mask]
#print(sum(masked_fsf))
#rint(len(masked_fsf))
#print(sum(masked_fsf) == len(masked_fsf))

lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][bgs_mask]
redshift = fsfData['Z'][bgs_mask]
plt.plot(redshift, lum, '.', alpha=0.3)
plt.xlabel("Redshift")
plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
plt.title("[OII] luminosity redshift dependence for BGS in sv3")
#plt.savefig(f'BGS oii luminosity vs redshift.png')
plt.show()