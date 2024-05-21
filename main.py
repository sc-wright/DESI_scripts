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


    return fsfData, fsfMeta


def add_lum_to_table(table):
    #table should be the fsf table

    table.add_column(Table.Column(data=-1 * np.ones(len(table)), dtype=float, name=f"OII_COMBINED_LUMINOSITY_LOG"))

    oII6Flux = np.array(table['OII_3726_FLUX'])
    oII9Flux = np.array(table['OII_3729_FLUX'])
    oIICombinedFlux = oII6Flux + oII9Flux
    redshift = np.array(table['Z'])
    npix = np.array(table['OII_3726_NPIX']) + np.array(table['OII_3729_NPIX'])
    dataLength = len(oIICombinedFlux)


    t = time.time()
    lastFullSecElapsed = int(time.time()-t)

    for i in range(dataLength):
        if npix[i] > 1:
            flux = oIICombinedFlux[i]
            if flux > 0:
                oIILum = np.log10(get_lum(flux, redshift[i]))
                table['OII_COMBINED_LUMINOSITY_LOG'][i] = oIILum

        # Displaying progress through the set to check for hanging
        elapsed = time.time() - t
        fullSecElapsed = int(elapsed)
        if fullSecElapsed > lastFullSecElapsed:
            lastFullSecElapsed = fullSecElapsed
            percent = 100 * (i + 1) / (dataLength)
            totalTime = elapsed / (percent / 100)
            remaining = totalTime - elapsed
            trString = ("Calculating [OII] luminosity, " + str(int(percent)) + "% complete. approx "
                        + str(int(remaining) // 60) + "m" + str(int(remaining) % 60) + "s remaining...")
            print('\r' + trString, end='', flush=True)


    #write_table_to_disk(table)

def write_table_to_disk(table):

    my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
    specprod = 'fuji'
    fssCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'

    ogname = fssCatalogsDir + "/fastspec-fuji-data-processed.fits"
    bakname = ogname + ".bak"

    print("Writing table...")
    try:
        print("renaming old table...")
        os.rename(ogname, bakname)
        print("writing new table to disk...")
        table.write(fssCatalogsDir + "/fastspec-fuji-data-processed.fits")
    except:
        print("old table not found, writing tale...")
        table.write(fssCatalogsDir + "/fastspec-fuji-data-processed.fits")


def add_col_to_table(table, colstr, data):

    print(f"Adding column {colstr} to table...")

    table.add_column(Table.Column(data=-1 * np.ones(len(table)), dtype=float, name=colstr))

    for i, v in enumerate(data):
        table[colstr][i] = v

    #write_table_to_disk(table)


#THIS FUNCTION IS DEPRECATED. USE THE OTHER FUNCTIONS IN COMBINATION.
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


def make_BGS_filter(table, metaTable=None):
    # creates a binary mask to select objects in sv3 with a calculated OII luminosity of type tracer
    surveyFilter = table['SURVEY'] == 'sv3'
    traceFilter = table[f'ISBGS']  # create mask for only the desired tracer
    lumFilter = table['OII_COMBINED_LUMINOSITY_LOG'] > 0
    zFilter = table['Z'] < 0.8
    #zOKFilter = metaTable['ZWARN']

    int1Filter = np.logical_and(traceFilter, surveyFilter)  # combine with the survey mask
    int2Filter = np.logical_and(int1Filter, zFilter)
    finalFilter = np.logical_and(int2Filter, lumFilter)

    return finalFilter


def plot_lum_vs_redshift(fsfData):
    print("Plotting luminosity vs redshift...")

    BGSFilter = make_BGS_filter(fsfData)

    lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][BGSFilter]
    redshift = fsfData['Z'][BGSFilter]
    plt.plot(redshift, lum, '.', alpha = 0.3)
    plt.xlabel("Redshift")
    plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
    plt.title("[OII] luminosity redshift dependence for BGS in sv3")
    plt.savefig(f'BGS oii luminosity vs redshift.png')
    plt.show()

def plot_lum_redshift_snr_color(fsfData):
    print("Plotting SNR limit...")
    BGSFilter = make_BGS_filter(fsfData)

    lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][BGSFilter]
    redshift = fsfData['Z'][BGSFilter]
    snr = calculate_oii_snr(fsfData)[BGSFilter]

    plt.plot(redshift[snr>3], lum[snr>3], '.', alpha=0.3)
    plt.plot(redshift[snr < 3], lum[snr < 3], '.', alpha=0.3)
    plt.xlabel("Redshift")
    plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
    plt.title("[OII] luminosity redshift dependence for BGS in sv3")
    plt.savefig(f'BGS oii luminosity vs redshift w snr.png')
    plt.show()

def plot_lum_redshift_chi2(fsfData):
    BGSFilter = make_BGS_filter(fsfData) # BGS sources at z < 0.8
    chiFlagFilter = fsfData['RCHI2_LINE'] > 2 # sources with high chi2
    chiOKFilter = fsfData['RCHI2_LINE'] <= 2 #sources with low chi2
    chiFlagFilter = np.logical_and(BGSFilter, chiFlagFilter)
    chiOKFilter = np.logical_and(BGSFilter, chiOKFilter)

    lum_chiFlag = fsfData['OII_COMBINED_LUMINOSITY_LOG'][chiFlagFilter]
    redshift_chiFlag = fsfData['Z'][chiFlagFilter]

    lum_chiOK = fsfData['OII_COMBINED_LUMINOSITY_LOG'][chiOKFilter]
    redshift_chiFOK = fsfData['Z'][chiOKFilter]

    plt.scatter(redshift_chiFlag, lum_chiFlag, alpha=0.3, label=r"$\chi^2 > 2$")
    #plt.scatter(redshift_chiFOK, lum_chiOK, alpha=0.3, label=r"$\chi^2 \leq 2$")

    plt.legend()
    plt.xlabel("Redshift")
    plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
    plt.title("[OII] luminosity redshift dependence for BGS in sv3")
    plt.savefig(f'BGS oii luminosity vs redshift w chi2.png')
    plt.show()



def calculate_oii_snr(fsfData):

    noise1 = np.array(fsfData['OII_3726_AMP_IVAR'])
    noise1 = np.sqrt(1 / noise1)
    noise2 = np.array(fsfData['OII_3729_AMP_IVAR'])
    noise2 = np.sqrt(1 / noise2)
    noise = np.sqrt(noise1**2 + noise2**2)

    oII6Flux = np.array(fsfData['OII_3726_FLUX'])
    oII9Flux = np.array(fsfData['OII_3729_FLUX'])
    oIICombinedFlux = oII6Flux + oII9Flux

    snr = oIICombinedFlux/noise

    return snr

def plot_lum_stellar_mass(fsfData):
    print("Plotting luminosity vs stellar mass...")

    # Making filters:
    BGSFilter = make_BGS_filter(fsfData) # BGS sources at z < 0.8
    chiFlagFilter = fsfData['RCHI2_LINE'] > 2 # sources with high chi2
    chiOKFilter = fsfData['RCHI2_LINE'] <= 2 #sources with low chi2
    chiFlagFilter = np.logical_and(BGSFilter, chiFlagFilter)
    chiOKFilter = np.logical_and(BGSFilter, chiOKFilter)

    lum_chiFlag = fsfData['OII_COMBINED_LUMINOSITY_LOG'][chiFlagFilter]
    stellar_mass_chiFlag = fsfData['LOGMSTAR'][chiFlagFilter]
    redshift_chiFlag = fsfData['Z'][chiFlagFilter]

    lum_chiOK = fsfData['OII_COMBINED_LUMINOSITY_LOG'][chiOKFilter]
    stellar_mass_chiOK = fsfData['LOGMSTAR'][chiOKFilter]
    redshift_chiFOK = fsfData['Z'][chiOKFilter]


    plt.scatter(stellar_mass_chiFlag, lum_chiFlag, alpha=0.3, label=r"$\chi^2 > 2$")#, c=redshift_chiFlag, marker=".", cmap="bwr", label=r"$\chi^2 < 2$")
    #plt.scatter(stellar_mass_chiOK, lum_chiOK, alpha = 0.3, label=r"$\chi^2 \leq 2$")#, c = redshift_chiFOK, marker="v", cmap="bwr", label=r"$\chi^2 \geq 2$")
    #plt.colorbar()
    plt.legend()
    plt.xlabel(r'$\log{M_\star/M_\odot}$')
    plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
    plt.title("[OII] luminosity stellar mass dependence for BGS in sv3")
    plt.savefig(f'BGS oii luminosity vs stellar mass w color w chi2.png')
    plt.show()


def plot_lum_stellar_mass_redshift_color(fsfData):
    print("Plotting luminosity vs stellar mass...")

    # Making filters:
    BGSFilter = make_BGS_filter(fsfData) # BGS sources at z < 0.8
    lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][BGSFilter]
    stellar_mass = fsfData['LOGMSTAR'][BGSFilter]
    redshift = fsfData['Z'][BGSFilter]

    plt.scatter(stellar_mass, lum, alpha=0.3, c=redshift, marker=".", cmap="bwr")
    plt.colorbar(label="z")
    #plt.legend()
    plt.xlim(4.6, 12.5)
    plt.ylim(34, 43)
    plt.xlabel(r'$\log{M_\star/M_\odot}$')
    plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
    plt.title("[OII] luminosity stellar mass dependence for BGS in sv3")
    plt.savefig(f'BGS oii luminosity vs stellar mass w color.png')
    plt.show()


def plot_lum_stellar_mass_redshift_color_zslice(fsfData):
    print("Plotting luminosity vs stellar mass...")

    # Making filters:
    BGSFilter = make_BGS_filter(fsfData) # BGS sources at z < 0.8


    sliceSize = .1
    for zSlice in np.arange(0,0.8,sliceSize):
        print("Plotting histogram of luminosity (z slice z={:.1f})...".format(zSlice))
        newFilterLayer = np.logical_and(fsfData['Z'] >= zSlice, fsfData['Z'] < (zSlice + sliceSize))
        zSlicedFilter = np.logical_and(newFilterLayer, BGSFilter)

        lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][zSlicedFilter]
        stellar_mass = fsfData['LOGMSTAR'][zSlicedFilter]
        redshift = fsfData['Z'][zSlicedFilter]

        plt.scatter(stellar_mass, lum, alpha=0.3, c=redshift, cmap="bwr")
        plt.colorbar(label="z")
        #plt.legend()
        plt.xlim(4.6, 12.5)
        plt.ylim(34, 43)
        plt.xlabel(r'$\log{M_\star/M_\odot}$')
        plt.ylabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
        plt.title("[OII] luminosity stellar mass dependence\nfor BGS in sv3 for {:.1f}".format(zSlice) + r" $\leq$ z < {:.1f}".format(zSlice + sliceSize))
        plt.savefig('BGS oii luminosity vs stellar mass w color z={:.1f}.png'.format(zSlice))
        plt.show()


def plot_lum_redshift_slice(fsfData):
    BGSFilter = make_BGS_filter(fsfData)
    intFilt = fsfData['OII_COMBINED_LUMINOSITY_LOG'] > 34
    BGSFilter = np.logical_and(intFilt,BGSFilter)
    sliceSize = .1
    for zSlice in np.arange(0,0.8,sliceSize):
        print("Plotting histogram of luminosity (z slice z={:.1f})...".format(zSlice))
        newFilterLayer = np.logical_and(fsfData['Z'] >= zSlice, fsfData['Z'] < (zSlice + sliceSize))
        zSlicedFilter = np.logical_and(newFilterLayer, BGSFilter)
        lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][zSlicedFilter]
        plt.hist(lum,bins=40)
        plt.xlabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
        plt.xlim(34, 43)
        plt.ylim(1,2E4)
        plt.yscale('log')
        #plt.ylabel("Number")
        plt.title(r"[OII] luminosity for {:.1f}".format(zSlice) + r" $\leq$ z < {:.1f}".format(zSlice + sliceSize))
        plt.savefig('oii luminosity histogram z={:.1f}.png'.format(zSlice))
        plt.show()


    #filter = add_filter_layer(BGSFilter)


def plot_lum_hist(fsfData):
    print("Plotting histogram of luminosity...")
    BGSFilter = make_BGS_filter(fsfData)
    lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][BGSFilter]
    plt.hist(lum,bins=30)
    plt.xlabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
    plt.ylabel("Number")
    plt.title("[OII] luminosity")
    plt.savefig(f'oii luminosity histogram.png')
    plt.show()


def plot_lum_snr(fsfData, redslice=False):
    # Making filters:
    BGSFilter = make_BGS_filter(fsfData) # BGS sources at z < 0.8

    if redslice:
        sliceSize = .1
    else:
        sliceSize = .8
    for zSlice in np.arange(0,0.8,sliceSize):
        print("Plotting lum vs snr (z slice z={:.1f})...".format(zSlice))
        newFilterLayer = np.logical_and(fsfData['Z'] >= zSlice, fsfData['Z'] < (zSlice + sliceSize))
        zSlicedFilter = np.logical_and(newFilterLayer, BGSFilter)

        lum = fsfData['OII_COMBINED_LUMINOSITY_LOG'][zSlicedFilter]
        snr = fsfData['OII_SUMMED_SNR'][zSlicedFilter]
        redshift = fsfData['Z'][zSlicedFilter]

        SNRgt3Flag = snr>3
        nSNRgt3 = sum(SNRgt3Flag)
        print(f"# with SNR > 3: {nSNRgt3}")

        SNRle3Flag = snr <= 3
        nSNRle3 = sum(SNRle3Flag)
        print(f"# with SNR <= 3: {nSNRle3}")

        plt.scatter(lum, snr, alpha=0.3, c=redshift, cmap="bwr")
        plt.colorbar(label="z")
        #plt.legend()
        plt.xlim(37,42.5)
        plt.ylim(0.1,50)
        plt.yscale('log')
        plt.ylabel(r'SNR($L_{[OII]}$)')
        plt.xlabel(r"$\log (L_{\mathrm{[OII]}})$ [erg s$^{-1}$]")
        plt.title("[OII] luminosity vs SNR\nfor BGS in sv3 for {:.1f}".format(zSlice) + r" $\leq$ z < {:.1f}".format(zSlice + sliceSize))
        plt.savefig('BGS oii luminosity vs snr for z={:.1f}.png'.format(zSlice))
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


def main():
    #check_files('27257')
    #zcat = spec_type()
    #pull_spec_data(hist=True)
    #check_rows()
    """
    try:
        my_dir = os.path.expanduser('~') + '/Documents/school/research/desidata'
        specprod = 'fuji'
        fssCatalogsDir = f'{my_dir}/public/edr/vac/edr/fastspecfit/{specprod}/v3.2/catalogs'

        print("reading in table...")
        fsfData = Table.read(f'{fssCatalogsDir}/fastspec-fuji-data-processed.fits')
    except:
        print("FITS with pre-calculated values not found, generating new file...")
        fsfData, fsfMeta = spec_type_in_fsf()
        add_lum_to_table(fsfData)
        snr = calculate_oii_snr(fsfData)
        add_col_to_table(fsfData, "OII_SUMMED_SNR", snr)
        write_table_to_disk(fsfData)
    """

    spec_type_in_fsf()

    #plot_lum_vs_redshift(fsfData)
    #plot_lum_redshift_snr_color(fsfData)
    #plot_lum_redshift_chi2(fsfData)
    #plot_lum_redshift_slice(fsfData)
    #plot_lum_stellar_mass(fsfData)
    #plot_lum_stellar_mass_redshift_color(fsfData)
    #plot_lum_stellar_mass_redshift_color_zslice(fsfData)

    #plot_lum_snr(fsfData, redslice=True)
    #plot_lum_snr(fsfData, redslice=False)



    #plot_lum_hist(fsfData)


    #print(fsfData['ISBGS'][:20])
    #print(fsfData['ISELG'][:20])
    #print(fsfData['TARGETID'][fsfData['ISBGS']])

if __name__ == '__main__':
    main()
