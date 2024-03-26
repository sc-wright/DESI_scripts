from utility_scripts import check_files, get_lum

import os
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Gaussian1DKernel

import matplotlib
import matplotlib.pyplot as plt

import time

import pandas as pd

# import DESI related modules -
from desimodel.footprint import radec2pix      # For getting healpix values
import desispec.io                             # Input/Output functions related to DESI spectra
from desispec import coaddition                # Functions related to coadding the spectra

# DESI targeting masks -
from desitarget.cmx.cmx_targetmask import cmx_mask as cmxmask
from desitarget.sv1.sv1_targetmask import desi_mask as sv1mask
from desitarget.sv2.sv2_targetmask import desi_mask as sv2mask
from desitarget.sv3.sv3_targetmask import desi_mask as sv3mask
from desitarget.targetmask import desi_mask as specialmask

def spec_type():

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

    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
    specprod = 'fuji'
    specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'

    zCatalogPath = os.path.join(specprod_dir, 'zcatalog', 'zall-pix-fuji.fits')

    zcat = Table.read(zCatalogPath)

    tracers = ['BGS', 'ELG', 'LRG', 'QSO', 'STAR', 'SCND']

    # Initialize columns to keep track of tracers. Set to -1 so we can ensure we fill all rows
    for tracer in tracers:
        zcat.add_column(Table.Column(data=-1 * np.ones(len(zcat)), dtype=int, name=f"IS{tracer}"))

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

        survey_selection = (zcat['SURVEY'] == survey)
        survey_subset = zcat[survey_selection]

        ## See if redrock thought it was a galaxy, star, or qso - this cannot be done with the fss, no 'spectype' key
        GALTYPE = (survey_subset['SPECTYPE'] == 'GALAXY')
        STARTYPE = (survey_subset['SPECTYPE'] == 'STAR')
        QSOTYPE = (survey_subset['SPECTYPE'] == 'QSO')

        ## BGS
        PASSES_BIT_SEL = ((survey_subset[colname] & bgs) > 0)
        zcat['ISBGS'][survey_selection] = (PASSES_BIT_SEL & GALTYPE)

        ## ELG
        PASSES_BIT_SEL = ((survey_subset[colname] & elg) > 0)
        zcat['ISELG'][survey_selection] = (PASSES_BIT_SEL & GALTYPE)

        ## LRG
        PASSES_BIT_SEL = ((survey_subset[colname] & lrg) > 0)
        zcat['ISLRG'][survey_selection] = (PASSES_BIT_SEL & GALTYPE)

        ## QSO
        PASSES_BIT_SEL = ((survey_subset[colname] & qso) > 0)
        zcat['ISQSO'][survey_selection] = (PASSES_BIT_SEL & QSOTYPE)

        ## STAR
        PASSES_BIT_SEL = ((survey_subset[colname] & star) > 0)
        zcat['ISSTAR'][survey_selection] = (PASSES_BIT_SEL & STARTYPE)

        ## Secondaries
        PASSES_BIT_SEL = ((survey_subset[colname] & sec) > 0)
        zcat['ISSCND'][survey_selection] = (PASSES_BIT_SEL)

        zcat.remove_column(colname)

    for tracer in tracers:
        col = f"IS{tracer}"
        print(f"For {tracer}: {np.sum(zcat[col] < 0):,} not set")
        if np.sum(zcat[col] < 0) == 0:
            zcat[col] = Table.Column(data=zcat[col], name=col, dtype=bool)

    #print(zCatTable['SUBTYPE'][:10])
    #for tracer in tracers:
    #    print(zcat[f'IS{tracer}'][:10])

    return zcat

def spec_type_in_fsf():

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

        survey_selection = (fsfMeta['SURVEY'] == survey) #creates a mask of the desired survey to slap on top of the whole set
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

    #print(zCatTable['SUBTYPE'][:10])
    #for tracer in tracers:
    #    print(zcat[f'IS{tracer}'][:10])

    return fsfData


def pull_spec_data(sepPlot=False, scatter=False, hist=False):

    """
    #### Making directory refs and reading tables #### (Don't use this anymore)

    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
    specprod = 'fuji'
    specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'

    #zCatalogPath = os.path.join(specprod_dir, 'zcatalog', 'zall-pix-fuji.fits')

    fssCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'

    print("reading in table...")

    fastSpecTable = Table.read(f'{fssCatalogsDir}/fastspec-fuji.fits', hdu=1)
    fastSpecTableMeta = Table.read(f'{fssCatalogsDir}/fastspec-fuji.fits', hdu=2)
    """

    fastSpecTable = spec_type_in_fsf()


    oII6Flux = np.array(fastSpecTable['OII_3726_FLUX'])
    oII9Flux = np.array(fastSpecTable['OII_3729_FLUX'])
    redshift = np.array(fastSpecTable['Z'])
    #targetid = np.array(fastSpecTable['TARGETID'])

    npix = np.array(fastSpecTable['OII_3726_NPIX']) + np.array(fastSpecTable['OII_3729_NPIX'])

    #for i in range(0,len(redshift)):
    #    if redshift[i] > 2:
    #        print(npix[i]) #if npix less than 2, just cut it from the sample

    combinedFlux = oII6Flux + oII9Flux

    dataNum = len(combinedFlux)


    tracers = ['BGS', 'ELG', 'LRG', 'QSO']#, 'STAR', 'SCND']
    surveyFilter = fastSpecTable['SURVEY'] == 'sv3'


    t = time.time()
    lastFullSecElapsed = int(time.time()-t)

    totalNum = 0
    for tracer in tracers:
        traceFilter = fastSpecTable[f'IS{tracer}']
        filters = np.logical_and(traceFilter, surveyFilter)
        totalNum += len(combinedFlux[filters])
    print(f"{totalNum} objects to calculate for {len(tracers)} tracers...")
    elpsNum = 0
    i = 0

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for colr, tracer in enumerate(tracers):

        # Filtering for the desired tracer and survey

        traceFilter = fastSpecTable[f'IS{tracer}']              # create mask for only the desired tracer
        filters = np.logical_and(traceFilter, surveyFilter)     # combine with the survey mask

        trcrString = f"Calculating luminosities for {tracer}... "
        dataNum = len(combinedFlux[filters])
        combinedLum = np.zeros(dataNum)
        tracerFlux = combinedFlux[filters]
        tracerRedshift = redshift[filters]
        npixTracer = npix[filters]
        elpsNum += i


        # Calculating the luminosity
        for i in range(dataNum):
            if npixTracer[i] > 1:
                combinedLum[i] = get_lum(float(tracerFlux[i]),float(tracerRedshift[i]))


        # Displaying progress through the set to check for hanging
                elapsed = time.time() - t
                fullSecElapsed = int(elapsed)
            if fullSecElapsed > lastFullSecElapsed:
                lastFullSecElapsed = fullSecElapsed
                percent = 100*(elpsNum+i+1)/(totalNum)
                #elapsed = time.time() - t
                totalTime = elapsed/(percent/100)
                remaining = totalTime - elapsed
                trString = str(int(percent)) + "% complete, approx " + str(int(remaining)//60) + "m" + str(int(remaining)%60) + "s remaining..."
                print('\r' + trcrString + trString, end='', flush=True)

        # Filtering out luminosities that could not be determined (set to zero)
        rsPlot = redshift[filters][combinedLum> 0]
        lumPlot = combinedLum[combinedLum > 0]

        # Plotting the luminosities
        if scatter:
            plt.plot(rsPlot, np.log10(lumPlot), '.', alpha=0.4, label=f'{tracer}, {len(lumPlot)} objects', color=colors[colr])

            # If seplot, make separate plots for each tracer. Otherwise, show all on one plot.
            if sepPlot:
                plt.xlabel("Redshift")
                plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
                plt.legend()
                plt.title("[OII] Luminosity redshift dependence in sv3")
                plt.savefig(f'oii luminosity for {tracer} vs redshift.png')
                plt.show()
        if hist:
            plt.hist(np.log10(lumPlot), bins=range(35,45), label=f'{tracer}', histtype='bar', stacked=True)
            plt.xlabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
            plt.title('Histogram of [OII] luminosities')
            plt.savefig(f'histogram of oii luminosity stacked {tracer}.png')
            plt.legend()
            plt.show()
    # Display total time elapsed
    tTime = time.time() - t
    print("done, ", int(tTime)//60, "minutes and ", int(tTime)%60, "seconds elapsed.")

    if scatter:
        if not sepPlot:
            plt.xlabel("Redshift")
            plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
            plt.legend()
            plt.title("[OII] Luminosity redshift dependence in sv3")
            plt.savefig(f'oii luminosity vs redshift.png')
            plt.show()
    if hist:
        plt.xlabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
        plt.title('Histogram of [OII] luminosities')
        plt.savefig('histogram of oii luminosity stacked.png')
        plt.legend()
        plt.show()



def check_rows():
    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
    specprod = 'fuji'
    specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'

    zCatalogPath = os.path.join(specprod_dir, 'zcatalog', 'zall-pix-fuji.fits')

    zcat = Table.read(zCatalogPath)

    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
    specprod = 'fuji'
    specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'

    zCatalogDir = os.path.join(specprod_dir, 'zcatalog', 'zall-pix-fuji.fits')

    fssCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'

    fss = Table.read(f'{fssCatalogsDir}/fastspec-fuji.fits')

    print(len(fss))
    print(len(zcat))




if __name__ == '__main__':
    #check_files('27257')
    #zcat = spec_type()
    pull_spec_data(hist=True)
    #check_rows()
    #fsfData = spec_type_in_fsf()
    #print(fsfData['ISBGS'][:20])
    #print(fsfData['ISELG'][:20])
    #print(fsfData['TARGETID'][fsfData['ISBGS']])
