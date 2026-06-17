import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq
plt.rcParams.update({
    'text.usetex': True
})

def read_density_table(filename):
    """
    Read MAPPINGS v5.1 pressure table.

    Returns:
        logne   : array
        logq   : array
        logOH  : array
        oii    : array
    """

    logne  = []
    logq  = []
    logOH = []
    oii   = []
    sii = []

    with open(filename, "r") as f:
        for line in f:
            # Skip header / separator lines
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            if line.startswith("----"):
                continue
            if not line[0].isdigit():
                continue

            try:
                # Fixed-width slices from byte description
                ne  = float(line[0:4])     # bytes 1–4
                q   = float(line[5:9])     # bytes 6–9
                OH  = float(line[10:14])   # bytes 11–14
                OII = float(line[55:62])   # bytes 56–62
                SII = float(line[90:97])   # bytes 91-97

                logne.append(ne)
                logq.append(q)
                logOH.append(OH)
                oii.append(OII)
                sii.append(SII)

            except ValueError:
                # Should not happen for valid rows, but skip defensively
                continue

    return (
        np.array(logne),
        np.array(logq),
        np.array(logOH),
        np.array(oii),
        np.array(sii)
    )

def ne_model(R, a, b, c):
    return np.log10((c * R - a * b) / (a - R))

def ne_model_global(X,
                    a0, a1,
                    b0, b1,
                    c0, c1):

    R, Z = X

    a = a0 + a1 * Z
    b = b0 + b1 * Z
    c = c0 + c1 * Z

    return np.log10((c * R - a * b) / (a - R))

def fit_ne_vs_ratio_by_metallicity_simultaneous(
    logne, logq, logOH, ratio,
    logq_fixed):

    # Select rows at fixed ionization parameter
    m = logq == logq_fixed
    if not np.any(m):
        raise ValueError(f"No rows found for logq = {logq_fixed}")

    R = ratio[m]
    Z = logOH[m]
    ne = logne[m]

    Z_vals = np.unique(Z)

    p0_global = (
        0.35, 0.003,  # a0,a1
        5000, -300,  # b0,b1
        1200, -70  # c0,c1
    )

    popt, pcov = curve_fit(
        ne_model_global,
        (R, Z),
        ne,
        p0=p0_global,
        maxfev=100000
    )

    fit_params = {}

    for Zval in np.unique(Z):
        a = popt[0] + popt[1] * Zval
        b = popt[2] + popt[3] * Zval
        c = popt[4] + popt[5] * Zval

        fit_params[Zval] = np.array([a, b, c])

    return Z_vals, fit_params


def fit_ne_vs_ratio_by_metallicity(
    logne, logq, logOH, ratio,
    logq_fixed,
    p0=(0.3771, 2468, 635)
        # initial guesses are tweaked from Sanders+16.
        # If using SII, replace these with (0.4315, 2107, 627.1) and then tweak further
):
    
    #Fit ne(R) at fixed log(q) for each metallicity.

    #Returns:
    #    Z_vals : sorted array of metallicities
    #    fit_params : dict mapping Z -> (a, b, c)

    # Select rows at fixed ionization parameter
    m = logq == logq_fixed
    if not np.any(m):
        raise ValueError(f"No rows found for logq = {logq_fixed}")

    logne_sel = logne[m]
    logOH_sel = logOH[m]
    ratio_sel = ratio[m]

    Z_vals = np.unique(logOH_sel)
    fit_params = {}

    for Z in Z_vals:
        mz = logOH_sel == Z

        R = ratio_sel[mz]
        ne = logne_sel[mz]
        #print(R, ne)
        #plt.plot(ne, R, label=f"Z = {Z}")

        # Sort for numerical stability
        idx = np.argsort(R)
        R = R[idx]
        ne = ne[idx]

        popt, pcov = curve_fit(
            ne_model,
            R,
            ne,
            p0=p0,
            maxfev=100000
        )

        #print(Z)
        #print(popt)
        #print(np.sqrt(np.diag(pcov)))

        fit_params[Z] = popt  # (a, b, c)

    return Z_vals, fit_params

    ### NEW SMOOTHING ADDITION BELOW - this is too smooth, makes everything equally spaced

    # Extract fitted coefficients
    a_vals = np.array([fit_params[Z][0] for Z in Z_vals])
    b_vals = np.array([fit_params[Z][1] for Z in Z_vals])
    c_vals = np.array([fit_params[Z][2] for Z in Z_vals])

    # Optional: exclude failed fits
    good = (
            np.isfinite(a_vals) &
            np.isfinite(b_vals) &
            np.isfinite(c_vals)
    )

    # Fit smooth trends with metallicity
    pa = np.polyfit(Z_vals[good], a_vals[good], 1)
    pb = np.polyfit(Z_vals[good], b_vals[good], 1)
    pc = np.polyfit(Z_vals[good], c_vals[good], 1)

    # Replace fitted parameters with smoothed values
    fit_params_smooth = {}

    for Z in Z_vals:
        a = np.polyval(pa, Z)
        b = np.polyval(pb, Z)
        c = np.polyval(pc, Z)

        fit_params_smooth[Z] = np.array([a, b, c])

    return Z_vals, fit_params_smooth


def plot_fits():
    logne, logq, logOH, oii, sii = read_density_table("apjab16edt2_mrt.txt")

    # Choose which ratio to use
    ratio = oii  # or sii

    # Fit once for a given ionization parameter
    Z_vals, fit_params = fit_ne_vs_ratio_by_metallicity(
        logne, logq, logOH, ratio,
        logq_fixed=7.5
    )
    #for Z in Z_vals:
    #    print(Z, fit_params[Z])
    fs = 18
    smfs = fs-4
    colors = ['b', 'g', 'r', 'c', 'm']
    y = np.linspace(0.38369, 1.4524, 100)
    fig = plt.figure()
    for i, Z in enumerate(Z_vals):
       #print(Z, fit_params[Z])
        x = ne_model(y, *fit_params[Z])
        plt.plot(x, y, label=Z, color=colors[i])
    plt.xlim(0.5,5)
    plt.xlabel(r"$\log{n_e}~[\mathrm{cm}^{-3}]$", fontsize=fs)
    plt.ylabel(r"$F_{\lambda3729}/F_{\lambda3726}$", fontsize=fs)
    plt.legend(title=r'$12 + \log(\mathrm{O}/\mathrm{H})$', fontsize=smfs)
    plt.tight_layout()
    plt.savefig('paper_figures/density_grid_interpolation_fits.png', dpi=DPI)
    plt.show()

def infer_logne_from_fits(
    logOH,
    ratio_obs,
    Z_vals,
    fit_params
):
    """
    Infer log(ne) given metallicity and observed ratio
    using fitted ne(R) relations and linear interpolation in Z.
    """

    # Catch masked or invalid metallicities
    if np.ma.is_masked(logOH) or not np.isfinite(logOH):
        return np.nan

    Z_vals = np.asarray(Z_vals)

    # Check bounds
    if logOH < Z_vals.min() or logOH > Z_vals.max():
        return np.nan

    # Exact metallicity match
    # We are omitting this to avoid discontinuities and make possible future error propagation easier
    #if logOH in fit_params:
    #    a, b, c = fit_params[logOH]
    #    return ne_model(ratio_obs, a, b, c)

    # Find bracketing metallicities
    i_hi = np.searchsorted(Z_vals, logOH)
    Z_lo = Z_vals[i_hi - 1]
    Z_hi = Z_vals[i_hi]

    # Evaluate both fits
    ne_lo = ne_model(ratio_obs, *fit_params[Z_lo])
    ne_hi = ne_model(ratio_obs, *fit_params[Z_hi])

    # Linear interpolation in metallicity
    w = (logOH - Z_lo) / (Z_hi - Z_lo)
    return (1 - w) * ne_lo + w * ne_hi

"""
#testing
logne, logq, logOH, oii, sii = read_density_table("apjab16edt2_mrt.txt")

# Choose which ratio to use
ratio = oii   # or sii

# Fit once for a given ionization parameter
Z_vals, fit_params = fit_ne_vs_ratio_by_metallicity(
    logne, logq, logOH, ratio,
    logq_fixed=7.5
)

# Repeated inference calls
logne_inferred = infer_logne_from_fits(
    logOH=8.427361704136585,
    ratio_obs=1/0.7968041,
    Z_vals=Z_vals,
    fit_params=fit_params
)

print(logne_inferred)

#plot_fits()

quit()
"""


################# OLD 2D INTERPOLATOR ####################


def build_oii_density_interpolator_2d(logne, logq, logOH, oii, sii, logq_fixed):
    """
    Builds a 2D RegularGridInterpolator at fixed log(q):
        (logne, logOH) -> OII ratio
    """

    # Select rows at the desired q
    m = logq == logq_fixed

    if not np.any(m):
        raise ValueError(f"No table entries found for logq = {logq_fixed}")

    logne_sel = logne[m]
    logOH_sel = logOH[m]
    oii_sel   = oii[m]
    sii_sel = sii[m]

    ne_vals = np.unique(logne_sel)
    OH_vals = np.unique(logOH_sel)

    # Create 2D grids
    OII_grid = np.full(
        (len(ne_vals), len(OH_vals)),
        np.nan
    )
    SII_grid = np.full(
        (len(ne_vals), len(OH_vals)),
        np.nan
    )

    # Fill grids
    for ne, OH, r in zip(logne_sel, logOH_sel, oii_sel):
        i = np.where(ne_vals == ne)[0][0]
        j = np.where(OH_vals == OH)[0][0]
        OII_grid[i, j] = r
    for ne, OH, r in zip(logne_sel, logOH_sel, sii_sel):
        i = np.where(ne_vals == ne)[0][0]
        j = np.where(OH_vals == OH)[0][0]
        SII_grid[i, j] = r

    interp_oii = RegularGridInterpolator(
        (ne_vals, OH_vals),
        OII_grid,
        bounds_error=False,
        fill_value=np.nan
    )

    interp_sii = RegularGridInterpolator(
        (ne_vals, OH_vals),
        SII_grid,
        bounds_error=False,
        fill_value=np.nan
    )

    return interp_oii, interp_sii, ne_vals



def infer_logne_2d(logOH, ratio_obs, ratio_interp, ne_bounds):
    """
    Infer log(ne) given log(O/H)+12 and observed ion flux ratio,
    using a 2D interpolator at fixed log(q).
    """

    def f(logne):
        model = ratio_interp((logne, logOH))
        return model - ratio_obs

    nemin, nemax = ne_bounds

    fmin = f(nemin)
    fmax = f(nemax)

    # Check that solution is bracketed
    if np.isnan(fmin) or np.isnan(fmax) or fmin * fmax > 0:
        return np.nan

    return brentq(f, nemin, nemax)


def example_density_interp():
    # Example usage

    # Read in table
    logne, logq, logOH, oii, sii = read_density_table(
        "apjab16edt2_mrt.txt"
    )

    # Build interpolator once for a given logq
    oii_interp, sii_interp, ne_vals = build_oii_density_interpolator_2d(
        logne, logq, logOH, oii, sii,
        logq_fixed=6.5
    )

    ne_bounds = (ne_vals.min(), ne_vals.max())

    # Repeated calls
    logne_inferred = infer_logne_2d(
        logOH=8.93,
        oii_obs=1.29,
        oii_interp=oii_interp,
        ne_bounds=ne_bounds
    )

    print(logne_inferred)


if __name__ == '__main__':
    DPI = 300
    plot_fits()





#############################################################################################
########################### OLD WORK, BEFORE ADDING FITTING #################################
#############################################################################################


def ionization_param(oii, oiii):
    # Morisset+16
    # We need an equation for ionization parameter that is dependent on pressure

    log_u = -2.74 - (1.0 * np.log10(oii/oiii))


def pressure_interpolation_grid():
    # read
    t = ascii.read("apjab16edt1_mrt.txt", format="cds")

    # axes
    logne_vals = np.unique(t['log(ne)'])
    logOH_vals = np.unique(t['logO/H+12'])
    logU_vals = np.unique(t['log(q)'])

    # choose the ratio column you want, e.g. 'OII' (make sure name matches)
    ratio_col = 'OII'   # replace with actual column name in your table

    # build grid: shape must be (len(logne_vals), len(logOH_vals))
    ratio_grid = np.full(
        (len(logne_vals), len(logOH_vals), len(logU_vals)),
        np.nan
    )
    for i, ne in enumerate(logne_vals):
        for j, oh in enumerate(logOH_vals):
            for k, logU in enumerate(logU_vals):
                mask = (
                        (t['log(ne)'] == ne) &
                        (t['logO/H+12'] == oh) &
                        (t['log(q)'] == logU)
                )
                if np.any(mask):
                    ratio_grid[i, j, k] = t[mask][ratio_col][0]

    # make sure no NaNs remain
    if np.isnan(ratio_grid).any():
        raise RuntimeError("Grid has missing values -> not purely regular.")

    interp_ratio = RegularGridInterpolator(
        (logne_vals, logOH_vals, logU_vals),
        ratio_grid,
        bounds_error=False,
        fill_value=None
    )

    return logne_vals, logOH_vals, logU_vals, ratio_grid, interp_ratio


def density_interpolation_grid():
    # read
    t = ascii.read("apjab16edt2_mrt.txt", format="cds")

    # axes
    logne_vals = np.unique(t['log(ne)'])
    logOH_vals = np.unique(t['logO/H+12'])
    logU_vals = np.unique(t['log(q)'])

    # choose the ratio column you want, e.g. 'OII' (make sure name matches)
    ratio_col = 'OII'   # replace with actual column name in your table

    # build grid: shape must be (len(logne_vals), len(logOH_vals))
    ratio_grid = np.full(
        (len(logne_vals), len(logOH_vals), len(logU_vals)),
        np.nan
    )
    for i, ne in enumerate(logne_vals):
        for j, oh in enumerate(logOH_vals):
            for k, logU in enumerate(logU_vals):
                mask = (
                        (t['log(ne)'] == ne) &
                        (t['logO/H+12'] == oh) &
                        (t['log(q)'] == logU)
                )
                if np.any(mask):
                    ratio_grid[i, j, k] = t[mask][ratio_col][0]

    # make sure no NaNs remain
    if np.isnan(ratio_grid).any():
        raise RuntimeError("Grid has missing values -> not purely regular.")

    interp_ratio = RegularGridInterpolator(
        (logne_vals, logOH_vals, logU_vals),
        ratio_grid,
        bounds_error=False,
        fill_value=None
    )

    return logne_vals, logOH_vals, logU_vals, ratio_grid, interp_ratio


# inversion function: find logne for obs_ratio and obs_logOH
def find_logne_for_ratio(obs_ratio, obs_logOH, logU, ne_min=None, ne_max=None):

    if ne_min is None: ne_min = logne_vals.min()
    if ne_max is None: ne_max = logne_vals.max()
    f = lambda ne: float(
        interp_ratio((ne, obs_logOH, logU)) - obs_ratio
    )
    # Optional: check sign change on bracket
    fa, fb = f(ne_min), f(ne_max)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("Requested metallicity outside grid range or extrapolating.")
    if fa == 0:
        return ne_min
    if fb == 0:
        return ne_max
    if fa*fb > 0:
        # no sign change: maybe monotonicity fails or obs_ratio outside grid range
        # return NaN or best-fit using minimization:
        res = root_scalar(lambda x: f(x), x0=(ne_min+ne_max)/2, method='secant')
        if res.converged:
            return res.root
        else:
            return np.nan

    sol = root_scalar(
        f,
        bracket=[logne_vals.min(), logne_vals.max()],
        method='brentq'
    )

    return sol.root


def plot_grids(logU_plot=6.5):

    kU = np.argmin(np.abs(logU_vals - logU_plot))

    # --- Plot 1: ratio vs log(ne), one curve per metallicity
    plt.figure(figsize=(7,5))

    for j, oh in enumerate(logOH_vals):
        # Extract the ratio as a function of logne for this metallicity
        ratios = ratio_grid[:, j, kU]
        plt.plot(logne_vals, ratios, label=f'{oh:.2f}')

    a = 0.3771
    b = 2468
    c = 638.4
    R = np.linspace(0.3839, 1.4558, 100)
    ne = np.log10((c * R - a * b) / (a - R))
    plt.plot(ne, R, label='Sanders+16', color='k')

    plt.ylabel('[O II] 3729/3726 line ratio')
    plt.xlabel(r'$log(n_e/cm^{3})$')
    plt.title(f'Density-sensitive [O II] ratio (log U = {logU_plot:.2f})')
    plt.legend(title='log(O/H) + 12', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'paper_figures/density_grid_{logU_plot:.2f}.png')
    plt.show()

    # --- Make fine grids
    ratio_range = np.linspace(np.nanmin(ratio_grid), np.nanmax(ratio_grid), 200)
    logOH_range = np.linspace(logOH_vals.min(), logOH_vals.max(), 200)
    logU_range = np.linspace(logU_vals.min(), logU_vals.max(), 200)

    # Allocate array for logne at each (ratio, metallicity)
    # shape: (len(ratio_range), len(logOH_range)) so Y=ratio, X=metallicity
    logne_lookup = np.full((len(ratio_range), len(logOH_range)), np.nan)

    # Invert ratio(logne, logOH) → logne(ratio, logOH)
    for j, oh in enumerate(logOH_range):
        for i, r in enumerate(ratio_range):
            f = lambda ne: interp_ratio((ne, oh, logU_plot)) - r
            try:
                logne_lookup[i, j] = brentq(f, logne_vals.min(), logne_vals.max())
            except ValueError:
                continue  # ratio outside range for this metallicity

    # --- Plot 2: metallicity on x, ratio on y
    plt.figure(figsize=(7,5))
    X, Y = np.meshgrid(logOH_range, ratio_range)

    # Contours of constant logne
    cs = plt.contour(X, Y, logne_lookup,
                     levels=np.arange(logne_vals.min(), logne_vals.max()+0.5, 0.5),
                     colors='white', linewidths=1)
    plt.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # Color background
    im = plt.imshow(logne_lookup, extent=[logOH_range.min(), logOH_range.max(), ratio_range.min(), ratio_range.max()],
                    origin='lower', aspect='auto', cmap='plasma')
    plt.colorbar(im, label='$log(n_e/cm^3)$')

    plt.xlabel('log(O/H) + 12')
    plt.ylabel('[O II] 3729/3726 line ratio')
    plt.title(f'Contours of constant electron density (log U = {logU_plot:.2f})')
    plt.tight_layout()
    plt.savefig(f'paper_figures/density_grid_interpolation_{logU_plot:.2f}.png', dpi=300)
    plt.show()


def plot_all_curves():

    linestyles = [(0, (5, 10)), '--', (0, (5, 1)), (0, (1, 10)), ':', (0, (1, 1)), (0, (3, 10, 1, 10)), '-.', (0, (3, 1, 1, 1))]
    colors = ['b', 'g', 'r', 'c', 'm']

    fig, ax = plt.subplots(figsize=(12, 8))

    # --- main grid
    for i, logU_plot in enumerate(np.arange(6.5, 8.75, 0.25)):
        kU = np.argmin(np.abs(logU_vals - logU_plot))

        for j, oh in enumerate(logOH_vals):
            # Extract the ratio as a function of logne for this metallicity
            ratios = ratio_grid[:, j, kU]
            plt.plot(logne_vals, ratios, label=f'{oh:.2f}, {logU_plot:.2f}', linestyle=linestyles[i], color=colors[j], linewidth=1)

    a = 0.3771
    b = 2468
    c = 638.4
    R = np.linspace(0.3839, 1.4558, 100)
    ne = np.log10((c * R - a * b) / (a - R))
    plt.plot(ne, R, label='Sanders+16', color='k')

    plt.ylabel('[O II] 3729/3726 line ratio', fontsize=20)
    plt.xlabel(r'$log(n_e/cm^{3})$', fontsize=20)
    plt.title(f'Metallicity and IP-sensitive [O II] ratio calibration', fontsize=20)
    # --- proxy handles for log(U) (linestyle only)
    style_handles = [
        Line2D([0], [0], color='k', lw=2, linestyle=ls)
        for ls in linestyles
    ]
    style_labels = [f'{u:.2f}' for u in logU_vals[:len(linestyles)]]

    leg1 = ax.legend(
        style_handles,
        style_labels,
        title='log(U)',
        bbox_to_anchor=(1.03, .7),
        loc='upper left',
        fontsize=14,
    )
    ax.add_artist(leg1)  # keep this legend when adding the next

    # --- proxy handles for metallicity (color only)
    color_handles = [
        Line2D([0], [0], color=c, lw=2)
        for c in colors
    ]
    color_labels = [f'{oh:.2f}' for oh in logOH_vals[:len(colors)]]

    leg2 = ax.legend(
        color_handles,
        color_labels,
        title='12 + log(O/H)',
        bbox_to_anchor=(1.03, .98),  # vertical offset → whitespace
        loc='upper left',
        fontsize=14,
    )

    #plt.legend(title='log(O/H) + 12, log(U)', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(right=0.83)
    plt.savefig(f'paper_figures/all_density_grid.png')
    plt.show()
    return 0

logne_vals, logOH_vals, logU_vals, ratio_grid, interp_ratio = density_interpolation_grid()


if __name__ == '__main__':
    #for logU in np.arange(6.5, 8.75, 0.25):
    #    plot_grids(logU_plot=logU)
    plot_all_curves()