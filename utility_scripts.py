import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import time


class CustomTimer:
    def __init__(self, dlen, calcstr=''):
        self.t = time.time()
        self.lastFullSecElapsed = int(time.time() - self.t)
        self.dataLength = int(dlen)
        if calcstr != '':
            self.calcstr = ' ' + str(calcstr)
        else:
            self.calcstr = calcstr
    def update_time(self, i):
        elapsed = time.time() - self.t
        fullSecElapsed = int(elapsed)
        if fullSecElapsed > self.lastFullSecElapsed:
            self.lastFullSecElapsed = fullSecElapsed
            percent = 100 * (i + 1) / self.dataLength
            totalTime = elapsed / (percent / 100)
            remaining = totalTime - elapsed
            trString = (f"Calculating{self.calcstr}, " + str(int(percent)) + "% complete. approx "
                        + str(int(remaining) // 60) + "m" + str(int(remaining) % 60) + "s remaining...")
            print('\r' + trString, end='', flush=True)


def check_files(desi_id, specprod = 'fuji', my_dir = '/Documents/school/research/desidata'):
    """
    input: desi_id is string
    :return: None
    """
    homedir = os.path.expanduser('~')
    my_dir = homedir + my_dir

    specprod_dir = f'{my_dir}/public/edr/spectro/redux/{specprod}'
    short_id = str(desi_id)[:-2]

    file = f'{specprod_dir}/healpix/sv1/bright/{short_id}/{desi_id}/spectra-sv1-bright-{desi_id}.fits'

    if not os.path.isfile(file):
        #print(f'downloading {desi_id}...')
        print("downloading sv1 brights...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv1/bright/{short_id}/{desi_id}/')
        print("downloading sv1 darks...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv1/dark/{short_id}/{desi_id}/')
        print("downloading sv3 brights...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv3/bright/{short_id}/{desi_id}/')
        print("downloading sv3 darks...")
        os.system(f'wget -r -nH --no-parent -e robots=off --reject="index.html*" --directory-prefix={my_dir} https://data.desi.lbl.gov/public/edr/spectro/redux/{specprod}/healpix/sv3/dark/{short_id}/{desi_id}/')
        print("downloading sv3 brights...")
    else:
        print("data is already local.")

    print("done!")

def E_z(z):
    Om = 0.3
    Ol = 0.7
    return np.sqrt( Om * ( 1 + z ) ** 3 + Ol ) # omega_k is 0

def D_c(zf):
    h = .7
    stepSize = 0.001
    steps = int(zf/stepSize)
    hInv = np.zeros(steps)
    zRange = np.linspace(0, zf, num=steps)#np.arange(0, zf, stepSize)

    D_h = 3000/h

    for i in range(len(zRange)):
        hInv[i] = 1./E_z(zRange[i])

    #print(len(hInv),len(zRange))
    out = D_h*np.trapz(hInv, x=zRange)

    return out

def D_m(z):
    return D_c(z)

def D_l_man(z):
    return (1 + z) * D_m(z)

def get_lum(f, z):
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    D_l = cosmo.luminosity_distance(z).cgs.value #this puts dL in cm
    f = f * 1E-17 #flux in erg/cm^2 now
    return f*4*np.pi*D_l**2

def generate_combined_mask(*masks):
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

def testfastqa():
    os.system(f"fastqa --targetids ")