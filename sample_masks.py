import numpy as np

from import_custom_catalog import CC
from utility_scripts import get_lum, generate_combined_mask, CustomTimer
from density_grid_interpolation_3d import find_logne_for_ratio

from scipy.stats import binned_statistic_2d

global SNR_LIM
SNR_LIM = 5

def bgs_mask():
    """
    This generates a catalog-level mask with 3 criteria:
    -Selection criteria from lss catalog (implied by zwarn = 0 - only BGS galaxies have a ZWARN in this catalog)
    -zwarn = 0
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

    ha_snr_mask = (ha_flux * np.sqrt(ha_ivar)) > snr_lim
    hb_snr_mask = (hb_flux * np.sqrt(hb_ivar)) > snr_lim

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

    ha_snr_mask = (ha_flux * np.sqrt(ha_ivar)) > snr_lim
    hb_snr_mask = (hb_flux * np.sqrt(hb_ivar)) > snr_lim

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

    return np.array(mass_mask)


def bgs_metallicity_cut(snr_lim=SNR_LIM):
    """
    This generates a BGS-level mask that cuts out the low-SNR lines used to calculate metallicity
    :return: boolean array (bgs length)
    """
    oii_3726_flux = np.array(CC.catalog['OII_3726_FLUX'][BGS_MASK])
    oii_3726_err_inv = np.array(np.sqrt(CC.catalog['OII_3726_FLUX_IVAR'][BGS_MASK]))
    oii_3729_flux = np.array(CC.catalog['OII_3729_FLUX'][BGS_MASK])
    oii_3729_err_inv = np.array(np.sqrt(CC.catalog['OII_3729_FLUX_IVAR'][BGS_MASK]))
    oiii_4959_flux = np.array(CC.catalog['OIII_4959_FLUX'][BGS_MASK])
    oiii_4959_err_inv = np.array(np.sqrt(CC.catalog['OIII_4959_FLUX_IVAR'][BGS_MASK]))
    oiii_5007_flux = np.array(CC.catalog['OIII_5007_FLUX'][BGS_MASK])
    oiii_5007_err_inv = np.array(np.sqrt(CC.catalog['OIII_5007_FLUX_IVAR'][BGS_MASK]))
    hbeta_flux = np.array(CC.catalog['HBETA_FLUX'][BGS_MASK])
    hbeta_flux_err_inv = np.array(np.sqrt(CC.catalog['HBETA_FLUX_IVAR'][BGS_MASK]))

    # The 3726, 3729, and Hbeta snr cuts are redundant with other masks we use to define our samples
    # But since this is relatively inexpensive, we repeat them here to make sure the metallicity mask is as complete as possible
    oii_3726_snr = (oii_3726_flux * oii_3726_err_inv) > snr_lim
    oii_3729_snr = (oii_3729_flux * oii_3729_err_inv) > snr_lim
    oiii_4959_snr = (oiii_4959_flux * oiii_4959_err_inv) > snr_lim
    oiii_5007_snr = (oiii_5007_flux * oiii_5007_err_inv) > snr_lim
    hbeta_snr = (hbeta_flux * hbeta_flux_err_inv) > snr_lim

    metallicity_mask = generate_combined_mask(oii_3726_snr, oii_3729_snr, oiii_4959_snr, oiii_5007_snr, hbeta_snr)

    return np.array(metallicity_mask)


def bgs_ne_snr_cut(snr_lim=5, line=0):
    """
    Generates a float array of valid ne values and a BGS-length boolean array for those values.
    Valid ne values are those where the total snr > 5 for all lines
    :param snr_lim: Changes the required snr for electron density. Default 5
    :param line: whether to use the average of oii and sii (0), oii (1), or sii (2). Default 0
    :return: float array of ne values (BGS length), boolean array mask for ne values (BGS length)
    """

    # THIS IS FULLY DEPRECATED NOW THAT WE ARE CALCULATING NE USING METALLICITY
    # DO NOT USE

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
    if line == 0:
        ne = np.array((ne_oii + ne_sii) * 0.5)
    elif line == 1:
        ne = np.array(ne_oii)
    elif line == 2:
        ne = np.array(ne_sii)
    else:
        print("No valid line specified, defaulting to average")

    # ne and ne_mask are both BGS length
    return ne, ne_mask


def bgs_oii_ne_snr_cut(snr_lim=5):
    # THIS IS FULLY DEPRECATED NOW THAT WE ARE CALCULATING NE USING METALLICITY
    # DO NOT USE

    # First calculate SNR for OII and SII in the BGS sample. Masks are BGS length
    #oii_1_snr = CC.catalog['OII_3726_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['OII_3726_FLUX_IVAR'][BGS_MASK]) > snr_lim
    #oii_2_snr = CC.catalog['OII_3729_FLUX'][BGS_MASK] * np.sqrt(CC.catalog['OII_3729_FLUX_IVAR'][BGS_MASK]) > snr_lim
    #oii_snr = oii_1_snr #generate_combined_mask(oii_1_snr, oii_2_snr)  # mask for oii - catalog length
    # We will only be using oii in this case
    combined_snr = np.ones(len(CC.catalog['OII_3726_FLUX'][BGS_MASK]))#oii_snr  # bgs length

    # Import the ne from both OII and SII
    ne_oii = CC.catalog['NE_OII'][BGS_MASK]  # ne values, bgs length

    # The locations with values are the inverse of the masked array mask
    valid_oii_mask = ~ne_oii.mask

    # The values are the data out of these masked arrays
    ne_oii_vals = ne_oii.data

    # This deals with any cases where the ratio is outside the valid range of the analytical equation from Sanders+2016
    positive_ne_oii = ne_oii_vals > 0

    # Require snr > 5 and positive ne values (negative values are outside the range of Sanders+2016 equation)
    print(valid_oii_mask)
    print(combined_snr)
    print(positive_ne_oii)
    ne_mask = valid_oii_mask & positive_ne_oii

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
    # THIS IS FULLY DEPRECATED NOW THAT WE ARE CALCULATING NE USING METALLICITY
    # DO NOT USE

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


def bgs_ne_oii_cut(snr_lim=5, q=7.5):
    # This is the most up-to-date way to get the right snr mask for ne using oii
    # Creates a BGS-length SNR mask
    oii_3726 = CC.catalog['OII_3726_FLUX'][BGS_MASK]
    oii_3726_ivar = CC.catalog['OII_3726_FLUX_IVAR'][BGS_MASK]
    oii_3729 = CC.catalog['OII_3729_FLUX'][BGS_MASK]
    oii_3729_ivar = CC.catalog['OII_3729_FLUX_IVAR'][BGS_MASK]
    oii_3726_snr_mask = (oii_3726 * np.sqrt(oii_3726_ivar)) > snr_lim
    oii_3729_snr_mask = (oii_3729 * np.sqrt(oii_3729_ivar)) > snr_lim
    ne = CC.catalog[f'NE_OII_{q}'][BGS_MASK]
    valid_ne = (~ne.mask) & (~np.isnan(ne.data))
    oii_snr_mask = generate_combined_mask(oii_3726_snr_mask, oii_3729_snr_mask, valid_ne)

    return oii_snr_mask


def bgs_ne_sii_cut(snr_lim=5, q=7.5):
    # This is the most up-to-date way to get the right snr mask for ne using sii
    # Creates a BGS_length SNR mask
    sii_6716 = CC.catalog['SII_6716_FLUX'][BGS_MASK]
    sii_6716_ivar = CC.catalog['SII_6716_FLUX_IVAR'][BGS_MASK]
    sii_6731 = CC.catalog['SII_6731_FLUX'][BGS_MASK]
    sii_6731_ivar = CC.catalog['SII_6731_FLUX_IVAR'][BGS_MASK]
    sii_6716_snr_mask = sii_6716 * np.sqrt(sii_6716_ivar) > snr_lim
    sii_6731_snr_mask = sii_6731 * np.sqrt(sii_6731_ivar) > snr_lim
    ne = CC.catalog[f'NE_SII_{q}'][BGS_MASK]
    valid_ne = (~ne.mask) & (~np.isnan(ne.data))
    oii_snr_mask = generate_combined_mask(sii_6716_snr_mask, sii_6731_snr_mask, valid_ne)

    return oii_snr_mask


def bgs_combined_snr_mask():
    """
    This generates a BGS-length mask that includes all SNR cuts:
    -Halpha/Hbeta lines (>5) (sfr mask)
    -CIGALE successfully fit for a mass
    -OIII 5007 and 4959 for metallicity (and others, redundant)
    -All 4 [OII] and [SII] lines (This could later be changed to just oii, if we want)
    :return: boolean array (bgs length)
    """

    sfr_mask = bgs_hydrogen_snr_cut(snr_lim=SNR_LIM)
    mass_mask = bgs_mass_cut()
    metallicity_mask = bgs_metallicity_cut(snr_lim=SNR_LIM)
    #_, ne_mask = bgs_ne_snr_cut(snr_lim=SNR_LIM)
    # We are changing to only cut on OII snr and ignoring SII altogether.
    ne_mask = bgs_ne_oii_cut(snr_lim=SNR_LIM)
    #sii_ne_mask = bgs_ne_sii_cut(snr_lim=SNR_LIM)
    #ne_mask = generate_combined_mask(oii_ne_mask, sii_ne_mask)
    bgs_complete_snr_mask = generate_combined_mask(sfr_mask, mass_mask, metallicity_mask, ne_mask)

    return bgs_complete_snr_mask


global BGS_MASK  # This is catalog length - master bgs mask
global CAT_SFR_MASK  # This is catalog length
global CAT_MASS_MASK  # This is catalog length
global BGS_SFR_MASK  # This is BGS length
global BGS_MASS_MASK  # This is BGS length
global BGS_SNR_MASK

BGS_MASK = bgs_mask()                                       # This is catalog length - selects everything in the BGS
CAT_SFR_MASK = cat_hydrogen_snr_cut(snr_lim=SNR_LIM)        # This is catalog length - selects everything that passes hydrogen SNR cuts
CAT_MASS_MASK = cat_mass_cut()                              # This is catalog length - selects everything that passes mass cuts
BGS_SFR_MASK = bgs_hydrogen_snr_cut(snr_lim=SNR_LIM)        # This is BGS length - selects everything that passes hydrogen cuts
BGS_MASS_MASK = bgs_mass_cut()                              # This is BGS length - selects everything that passes mass cuts
BGS_SNR_MASK = bgs_combined_snr_mask()                      # This is BGS length - selects everything that passes all SNR cuts

#if __name__ == '__main__':
#    print(f'Total BGS Galaxies that pass SNR: {sum(np.array(BGS_SNR_MASK))}')


def redshift_percentiles():

    combined_snr_mask = BGS_SNR_MASK                # Includes BGS, SFR, MASS, NE cuts

    redshift = CC.catalog['Z'][BGS_MASK]

    redshift = redshift[combined_snr_mask]

    z80 = np.percentile(redshift, 80)
    z90 = np.percentile(redshift, 90)

    z40 = np.percentile(redshift, 40)
    z50 = np.percentile(redshift, 50)

    #if __name__ == '__main__':
    #    print(f'Low-z limit: {z50:.3f}')
    #    print(f'All-z limit: {z90:.3f}')

    return z40, z50, z80, z90


def redshift_complete_mask():
    """
    This generates a BGS-length mask that includes only objects above the 10% minima in the hi-z bin
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

    #if __name__ == '__main__':
    #    print(f"lo-z mass: {mass_lo_10}", f"all-z mass: {mass_hi_10}")
    #    print(f"lo-z sfr: {sfr_lo_10}", f"all-z sfr: {sfr_hi_10}")

    hi_z_bin = (mass > mass_hi_10) & (sfr > sfr_hi_10) & (redshift < z90) & BGS_SNR_MASK
    lo_z_bin = (mass > mass_lo_10) & (sfr > sfr_lo_10) & (redshift < z50) & BGS_SNR_MASK

    medz50 = np.median(redshift[lo_z_bin])
    medz90 = np.median(redshift[hi_z_bin])

    print(f"Median lo-z mass:{np.median(mass[lo_z_bin])}")
    print(f"Median hi-z mass:{np.median(mass[hi_z_bin])}")
    print(f"Median lo-z sfr:{np.median(sfr[lo_z_bin])}")
    print(f"Median hi-z sfr:{np.median(sfr[hi_z_bin])}")

    return lo_z_bin, hi_z_bin, z50, z90, medz50, medz90, mass_lo_10, mass_hi_10, sfr_lo_10, sfr_hi_10


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
global MEDZ50
global MEDZ90

LO_Z_MASK, HI_Z_MASK, Z50, Z90, MEDZ50, MEDZ90, M50, M90, SFR50, SFR90 = redshift_complete_mask()

bgs_count = sum(np.array(BGS_MASK))
snr_count = sum(np.array(BGS_SNR_MASK))
lo_z_count = sum(np.array(LO_Z_MASK))
hi_z_count = sum(np.array(HI_Z_MASK))
all_count = sum(np.array(np.logical_or(np.array(LO_Z_MASK), np.array(HI_Z_MASK))))

#print(snr_count / bgs_count * 10**7)
#print(lo_z_count / bgs_count * 10**7)
#print(hi_z_count / bgs_count * 10**7)
#print(all_count / bgs_count * 10**7)

#print(sum(np.array(np.logical_or(np.array(LO_Z_MASK), np.array(HI_Z_MASK)))))

if __name__ == '__main__':
    print(f'low-z:\ncount:\t\t{lo_z_count}\nmax z:\t\t{Z50:.3f}\nmed z:\t\t{MEDZ50}\nmin mass:\t{M50:.3f}\nmin sfr:\t{SFR50:.3f}')
    print(f'all-z:\ncount:\t\t{hi_z_count}\nmax z:\t\t{Z90:.3f}\nmed z:\t\t{MEDZ90}\nmin mass:\t{M90:.3f}\nmin sfr:\t{SFR90:.3f}')


def spot_checking_ne():
    # This is deprecated after the switch to calculating ne differently. Needs to be fixed or removed.
    metallicity = CC.catalog['METALLICITY_O3N2'][BGS_MASK]
    oii_ratio = CC.catalog['OII_DOUBLET_RATIO'][BGS_MASK]
    sii_ratio = CC.catalog['SII_DOUBLET_RATIO'][BGS_MASK]
    ne_oii = CC.catalog['NE_OII'][BGS_MASK]
    ne_sii = CC.catalog['NE_SII'][BGS_MASK]

    metallicity = metallicity[LO_Z_MASK]
    oii_ratio = oii_ratio[LO_Z_MASK]
    sii_ratio = sii_ratio[LO_Z_MASK]
    ne_oii = ne_oii[LO_Z_MASK]
    ne_sii = ne_sii[LO_Z_MASK]

    #print(sum(np.array(LO_Z_MASK)))

    random_indices = [93, 2835, 548, 3432, 4278]

    for i in random_indices:
        print(oii_ratio[i], metallicity[i])
        print(find_logne_for_ratio(obs_ratio=1/oii_ratio[i], obs_logOH=metallicity[i]), np.log10(ne_oii[i]))
