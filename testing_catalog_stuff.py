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



# Release directory path

my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
specprod = 'fuji'
specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'

fujidata = Table( fitsio.read(os.path.join(specprod_dir, "zcatalog", "zall-pix-{}.fits".format(specprod))) )

is_sv3 = (fujidata["SURVEY"].astype(str).data == "sv3")

is_prim_sv3 = (is_sv3 & fujidata["SV_PRIMARY"])

bgs_tgtmask  = desi_mask["BGS_ANY"]
is_primary = fujidata["ZCAT_PRIMARY"]

desi_target = fujidata["DESI_TARGET"]

is_sv3_bgs = (desi_target & bgs_tgtmask != 0) | (fujidata["SV3_DESI_TARGET"] & bgs_tgtmask != 0)

is_bgs_sv3 = (fujidata["SV3_DESI_TARGET"] & bgs_tgtmask != 0)

print(is_bgs_sv3)
print(sum(is_bgs_sv3))