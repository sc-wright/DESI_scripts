import numpy as np

from import_custom_catalog import CC
from utility_scripts import get_lum, generate_combined_mask, CustomTimer

from scipy.stats import binned_statistic_2d

global SNR_LIM
SNR_LIM = 5

def bgs_mask():
    """
    This generates a catalog-level mask with 3 criteria:
    -Selection criteria from lss catalog + zwarn = 0
    -z <= 0.4
    -effective radius  <= 7/1.5  (for purposes of aperture/color gradient correction)
    :return: boolean array (catalog length)
    """
    bgs_mask = generate_combined_mask(CC.catalog['ZWARN'] == 0, CC.catalog['Z'] <= 0.4, CC.catalog['SHAPE_R']* 1.5 <= 7)
    bgs_mask = bgs_mask.filled(False)
    return bgs_mask


def cat_hydrogen_snr_cut(snr_lim=5):
    """
    This generates a catalog-level mask with Halpha and Hbeta fluxes both above snr_lim
    :param snr_lim: what snr to cut at. defaults to 5
    :return: boolean array (catalog length)
    """

    ha_flux = CC.catalog['HALPHA_FLUX']
    ha_ivar = CC.catalog['HALPHA_FLUX_IVAR']
    hb_flux = CC.catalog['HBETA_FLUX']
    hb_ivar = CC.catalog['HBETA_FLUX_IVAR']

    ha_snr_mask = ha_flux * np.sqrt(ha_ivar) > snr_lim
    hb_snr_mask = hb_flux * np.sqrt(hb_ivar) > snr_lim

    sfr_mask = (ha_snr_mask) & (hb_snr_mask)

    return sfr_mask


def bgs_hydrogen_snr_cut(snr_lim=5):
    """
    This generates a BGS-level mask with Halpha and Hbeta fluxes both above snr_lim
    This results the same as convolving bgs mask with the above snr mask.
    :param snr_lim: what snr to cut at. defaults to 5
    :return: boolean array (bgs length)
    """
    ha_flux = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    ha_ivar = CC.catalog['HALPHA_FLUX_IVAR'][BGS_MASK]
    hb_flux = CC.catalog['HBETA_FLUX'][BGS_MASK]
    hb_ivar = CC.catalog['HBETA_FLUX_IVAR'][BGS_MASK]


    ha_snr_mask = ha_flux * np.sqrt(ha_ivar) > snr_lim
    hb_snr_mask = hb_flux * np.sqrt(hb_ivar) > snr_lim

    sfr_mask = (ha_snr_mask) & (hb_snr_mask)

    return sfr_mask


def cat_mass_cut():
    """
    This generates a catalog-level mask that cuts out the most unphysical masses reported by CIGALE.
    It should only be used for some diagnostic purposes
    :return: boolean array (catalog length)
    """

    mass = CC.catalog['MSTAR_CIGALE']

    mass_mask = mass > 1

    mass_mask = mass_mask.filled(False)

    return mass_mask


def bgs_mass_cut():
    """
    This generates a BGS-level mask that cuts out most unphysical masses reported by CIGALE.
    :return: boolean array (bgs length)
    """
    mass = CC.catalog['MSTAR_CIGALE'][BGS_MASK]

    mass_mask = mass > 1

    mass_mask = mass_mask.filled(False)

    return mass_mask


def bgs_ne_snr_cut(snr_lim=5):
    """
    Generates a float array of valid ne values and a BGS-length boolean array for those values.
    Valid ne values are those where the total snr > 5 for all lines
    :param snr_lim: Changes the required snr for electron density. Default 5
    :return: float array of ne values (BGS length), boolean array mask for ne values (BGS length)
    """

    # First calculate SNR for OII and SII in the BGS sample. Masks are BGS length
    oii_1_snr = CC.catalog['OII_3726_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['OII_3726_FLUX_IVAR'][BGS_MASK]) > snr_lim
    oii_2_snr = CC.catalog['OII_3729_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['OII_3729_FLUX_IVAR'][BGS_MASK]) > snr_lim
    oii_snr = generate_combined_mask(oii_1_snr, oii_2_snr)  # mask for oii - catalog length
    sii_1_snr = CC.catalog['SII_6716_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['SII_6716_FLUX_IVAR'][BGS_MASK]) > snr_lim
    sii_2_snr = CC.catalog['SII_6731_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['SII_6731_FLUX_IVAR'][BGS_MASK]) > snr_lim
    sii_snr = generate_combined_mask(sii_1_snr, sii_2_snr)  # mask for sii - catalog length
    # Now we & them to find the objects with high enough snr for both
    combined_snr = generate_combined_mask(oii_snr, sii_snr)  # bgs length

    # Import the ne from both OII and SII
    ne_oii = CC.catalog['NE_OII'][BGS_MASK]  # ne values, bgs length
    ne_sii = CC.catalog['NE_SII'][BGS_MASK]  # ne values, bgs length

    # The locations with values are the inverse of the masked array mask
    valid_oii_mask = ~ne_oii.mask
    valid_sii_mask = ~ne_sii.mask

    # The values are the data out of these masked arrays
    ne_oii_vals = ne_oii.data
    ne_sii_vals = ne_sii.data

    # This deals with any cases where the ratio is outside the valid range of the analytical equation from Sanders+2016
    positive_ne_oii = ne_oii_vals > 0
    positive_ne_sii = ne_sii_vals > 0

    # Require snr > 5 and positive ne values (negative values are outside the range of Sanders+2016 equation)
    ne_mask = valid_oii_mask & valid_sii_mask & combined_snr & positive_ne_oii & positive_ne_sii

    # Taking log but silencing warnings because mask will handle the undefined values
    # Save the current settings
    old_settings = np.seterr(all='ignore')
    # Taking log
    ne_oii = np.log10(ne_oii_vals)
    ne_sii = np.log10(ne_sii_vals)
    # Restore original settings
    np.seterr(**old_settings)

    # This ensures ne_mask is a fully filled boolean array rather than a masked array - easier to count
    ne_mask = ne_mask.filled(False)

    # Take the average of the two electron densities
    ne = np.array((ne_oii + ne_sii) * 0.5)

    # ne and ne_mask are both BGS length
    return ne, ne_mask


def bgs_oii_ne_snr_cut(snr_lim=5):
    # First calculate SNR for OII and SII in the BGS sample. Masks are BGS length
    oii_1_snr = CC.catalog['OII_3726_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['OII_3726_FLUX_IVAR'][BGS_MASK]) > snr_lim
    oii_2_snr = CC.catalog['OII_3729_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['OII_3729_FLUX_IVAR'][BGS_MASK]) > snr_lim
    oii_snr = generate_combined_mask(oii_1_snr, oii_2_snr)  # mask for oii - catalog length
    # We will only be using oii in this case
    combined_snr = oii_snr  # bgs length

    # Import the ne from both OII and SII
    ne_oii = CC.catalog['NE_OII'][BGS_MASK]  # ne values, bgs length

    # The locations with values are the inverse of the masked array mask
    valid_oii_mask = ~ne_oii.mask

    # The values are the data out of these masked arrays
    ne_oii_vals = ne_oii.data

    # This deals with any cases where the ratio is outside the valid range of the analytical equation from Sanders+2016
    positive_ne_oii = ne_oii_vals > 0

    # Require snr > 5 and positive ne values (negative values are outside the range of Sanders+2016 equation)
    ne_mask = valid_oii_mask & combined_snr & positive_ne_oii

    # Taking log but silencing warnings because mask will handle the undefined values
    # Save the current settings
    old_settings = np.seterr(all='ignore')
    # Taking log
    ne_oii = np.log10(ne_oii_vals)
    # Restore original settings
    np.seterr(**old_settings)

    # This ensures ne_mask is a fully filled boolean array rather than a masked array - easier to count
    ne_mask = ne_mask.filled(False)

    # Take the average of the two electron densities
    ne = ne_oii

    # ne and ne_mask are both BGS length
    return ne, ne_mask


def bgs_sii_ne_snr_cut(snr_lim=5):
    """
    Generates a float array of valid ne values and a BGS-length boolean array for those values.
    Valid ne values are those where the total snr > 5 for all lines
    :param snr_lim: Changes the required snr for electron density. Default 5
    :return: float array of ne values (BGS length), boolean array mask for ne values (BGS length)
    """

    # First calculate SNR for OII and SII in the BGS sample. Masks are BGS length
    sii_1_snr = CC.catalog['SII_6716_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['SII_6716_FLUX_IVAR'][BGS_MASK]) > snr_lim
    sii_2_snr = CC.catalog['SII_6731_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['SII_6731_FLUX_IVAR'][BGS_MASK]) > snr_lim
    sii_snr = generate_combined_mask(sii_1_snr, sii_2_snr)  # mask for sii - catalog length
    # Now we & them to find the objects with high enough snr for both
    combined_snr = sii_snr  # bgs length

    # Import the ne from both OII and SII
    ne_sii = CC.catalog['NE_SII'][BGS_MASK]  # ne values, bgs length

    # The locations with values are the inverse of the masked array mask
    valid_sii_mask = ~ne_sii.mask

    # The values are the data out of these masked arrays
    ne_sii_vals = ne_sii.data

    # This deals with any cases where the ratio is outside the valid range of the analytical equation from Sanders+2016
    positive_ne_sii = ne_sii_vals > 0

    # Require snr > 5 and positive ne values (negative values are outside the range of Sanders+2016 equation)
    ne_mask = valid_sii_mask & combined_snr & positive_ne_sii

    # Taking log but silencing warnings because mask will handle the undefined values
    # Save the current settings
    old_settings = np.seterr(all='ignore')
    # Taking log
    ne_sii = np.log10(ne_sii_vals)
    # Restore original settings
    np.seterr(**old_settings)

    # This ensures ne_mask is a fully filled boolean array rather than a masked array - easier to count
    ne_mask = ne_mask.filled(False)

    # Take the average of the two electron densities
    ne = np.array(ne_sii)

    # ne and ne_mask are both BGS length
    return ne, ne_mask


def bgs_combined_snr_mask():
    """
    This generates a BGS-length mask that includes all SNR cuts:
    -CIGALE successfully fit for a mass
    -Halpha/Hbeta lines (>5)
    -All 4 [OII] and [SII] lines (>5)
    :return: boolean array (bgs length)
    """

    sfr_mask = bgs_hydrogen_snr_cut(snr_lim=SNR_LIM)
    mass_mask = bgs_mass_cut()
    _, ne_mask = bgs_ne_snr_cut()
    bgs_complete_snr_mask = generate_combined_mask(sfr_mask, mass_mask, ne_mask)

    return bgs_complete_snr_mask



global BGS_MASK  # This is catalog length - master bgs mask
global CAT_SFR_MASK  # This is catalog length
global CAT_MASS_MASK  # This is catalog length
global BGS_SFR_MASK  # This is BGS length
global BGS_MASS_MASK  # This is BGS length
global BGS_SNR_MASK

BGS_MASK = bgs_mask()                                       # This is catalog length - master bgs mask
CAT_SFR_MASK = cat_hydrogen_snr_cut(snr_lim=SNR_LIM)        # This is catalog length
CAT_MASS_MASK = cat_mass_cut()                              # This is catalog length
BGS_SFR_MASK = bgs_hydrogen_snr_cut(snr_lim=SNR_LIM)        # This is BGS length
BGS_MASS_MASK = bgs_mass_cut()                              # This is BGS length
BGS_SNR_MASK = bgs_combined_snr_mask()                      # This is BGS length



def redshift_percentiles():

    combined_snr_mask = BGS_SNR_MASK                # Includes BGS, SFR, MASS, NE cuts

    redshift = CC.catalog['Z'][BGS_MASK]

    redshift = redshift[combined_snr_mask]

    z80 = np.percentile(redshift, 80)
    z90 = np.percentile(redshift, 90)

    z40 = np.percentile(redshift, 40)
    z50 = np.percentile(redshift, 50)

    return z40, z50, z80, z90

def redshift_complete_mask():
    """
    This generates an SNR-length mask that includes only objects above the 10% minima in the hi-z bin
    It does *not* cut on redshift explicitly.
    :return: boolean array (snr mask length)
    """

    z40, z50, z80, z90 = redshift_percentiles()

    sfr = np.array(CC.catalog['SFR_HALPHA'][BGS_MASK])
    mass = np.array(CC.catalog['MSTAR_CIGALE'][BGS_MASK])
    redshift = np.array(CC.catalog['Z'][BGS_MASK])

    sfr_snr = sfr[BGS_SNR_MASK]
    mass_snr = mass[BGS_SNR_MASK]
    redshift_snr = redshift[BGS_SNR_MASK]

    redshift_hi_mask = generate_combined_mask(redshift_snr > z80, redshift_snr <= z90)
    redshift_lo_mask = generate_combined_mask(redshift_snr > z40, redshift_snr <= z50)

    mass_hi_10 = np.percentile(mass_snr[redshift_hi_mask], 10)
    mass_lo_10 = np.percentile(mass_snr[redshift_lo_mask], 10)

    sfr_hi_10 = np.percentile(sfr_snr[redshift_hi_mask], 10)
    sfr_lo_10 = np.percentile(sfr_snr[redshift_lo_mask], 10)

    #print(f"lo-z mass: {mass_lo_10}", f"all-z mass: {mass_hi_10}")
    #print(f"lo-z sfr: {sfr_lo_10}", f"all-z sfr: {sfr_hi_10}")

    hi_z_bin = (mass > mass_hi_10) & (sfr > sfr_hi_10) & (redshift < z90) & BGS_SNR_MASK
    lo_z_bin = (mass > mass_lo_10) & (sfr > sfr_lo_10) & (redshift < z50) & BGS_SNR_MASK

    return lo_z_bin, hi_z_bin, z50, z90, mass_lo_10, mass_hi_10, sfr_lo_10, sfr_hi_10


def get_galaxy_type_mask(sample_mask=BGS_MASK):
    """
    Returns boolean mask of hii, composite, agn, and shock galaxies based on BPT region
    returned mask is length left after sample_mask is applied
    therefore this mask should be used *after* the chosen sample mask
    :param sample_mask:
    :return:
    """

    # potentially change this so instead of a flat snr cut we keep uncertainties
    # and find other ways to deal with it
    snr_lim = 3#SNR_LIM

    # Extracting line fluxes from the catalog.
    # All are BGS length
    nii = CC.catalog['NII_6584_FLUX'][BGS_MASK]
    nii_snr = nii * np.sqrt(CC.catalog['NII_6584_FLUX_IVAR'][BGS_MASK])
    ha = CC.catalog['HALPHA_FLUX'][BGS_MASK]
    oiii = CC.catalog['OIII_5007_FLUX'][BGS_MASK]
    oiii_snr = oiii * np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK])
    hb = CC.catalog['HBETA_FLUX'][BGS_MASK]

    # removing all cases where the selected line flux is zero, since log(0) and x/0 are undefined
    # all input masks are BGS length
    bpt_mask = generate_combined_mask(nii_snr > snr_lim, oiii_snr > snr_lim)

    nh = np.log10(nii / ha)  # x-axis
    oh = np.log10(oiii / hb) # y-axis

    hii_boundary = lambda x: 0.61/(x - 0.05) + 1.3          # black dashed
    agn_boundary = lambda x: 0.61 / (x - 0.47) + 1.19       # red dotted
    shock_boundary = lambda x: 2.144507*x + 0.465028        # blue dotdash

    hii_object_mask         = (oh < agn_boundary(nh)) & (oh < hii_boundary(nh))         # below both red and black lines
    agn_object_mask         = (oh > agn_boundary(nh)) & (oh > shock_boundary(nh))       # above both red and blue
    composite_object_mask   = (oh > hii_boundary(nh)) & (oh < agn_boundary(nh))         # above black and below red
    shock_object_mask       = (oh > agn_boundary(nh)) & (oh < shock_boundary(nh))       # above red and below blue

    return hii_object_mask, agn_object_mask, composite_object_mask, shock_object_mask


global LO_Z_MASK                                    # BGS-length
global HI_Z_MASK                                    # BGS-length
global Z50
global Z90
global M50
global M90
global SFR50
global SFR90

LO_Z_MASK, HI_Z_MASK, Z50, Z90, M50, M90, SFR50, SFR90 = redshift_complete_mask()


