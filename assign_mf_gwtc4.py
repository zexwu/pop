import argparse

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.stats import truncnorm

from imf import chabrier03_imf, gwtc3_mf

ang_eq = u.dimensionless_angles()
# (km/s)/kpc -> mas/yr
k_pm = (1 * u.km / u.s / u.kpc).to(u.mas / u.yr, ang_eq).value
kappa = (4 * const.G / const.c**2 / u.kpc * u.M_sun).to(u.mas**2, ang_eq).value
days_per_year = u.year.to(u.day)

# mass boundaries for object types
wd_min, wd_max = 1, 8
rem_min, rem_max = 8, 100

wd_mean, wd_std = 0.60, 0.16  # White Dwarfs
ns_mean, ns_std = 1.35, 0.04  # Neutron Stars

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="Input file with kinematics.")
parser.add_argument("--kick", type=float, help="Mean BH natal kick velocity in km/s.")
parser.add_argument("--seed", type=int, help="Random seed")
args = parser.parse_args()
np.random.seed(args.seed)
# fmt: on

# load lens-source kinematic data
dat = np.loadtxt(args.file, skiprows=2)
dl, ds, lens_pos, pm_y, pm_z, vl_y, vl_z, vs_y, vs_z = dat.T
num_samples = len(ds)
dl[dl == ds] -= 1e-5

# sample lens mass
logm = np.random.uniform(-2, 2, num_samples)
mass = 10**logm
mf = chabrier03_imf(mass)
stype = np.zeros(num_samples, dtype=int)

is_wd = (mass >= wd_min) & (mass < wd_max)
is_rem = (mass >= rem_min) & (mass < rem_max)

stype[is_wd] = 1
stype[is_rem] = 2

# fmt: off
mass[is_wd] = truncnorm.rvs(-3.5, 3.5, loc=wd_mean, scale=wd_std, size=num_samples, random_state=1)[is_wd]
# fmt: on

# sample BH mass from GWTC-3 mf
logm_bh_grid = np.linspace(np.log10(1), np.log10(100), 1001)
cdf = np.cumsum(gwtc3_mf(10**logm_bh_grid))
cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
rd = np.random.rand(num_samples)
mass[is_rem] = 10 ** np.interp(rd, cdf, logm_bh_grid)[is_rem]

# random directions
cth = np.random.uniform(-1, 1, num_samples)
sth = np.sqrt(1 - cth**2)
phi = np.random.uniform(0, 2 * np.pi, num_samples)
kick_dir_x = sth * np.cos(phi)
kick_dir_y = sth * np.sin(phi)


# apply NS kicks
sigma_ns = 100 # km/s
v_ns_components = np.random.normal(0.0, sigma_ns, (num_samples, 3))
v_ns_speed = np.sqrt(np.sum(v_ns_components**2, axis=1))
pm_y[is_rem] += (v_ns_speed * kick_dir_x / dl * k_pm)[is_rem]
pm_z[is_rem] += (v_ns_speed * kick_dir_y / dl * k_pm)[is_rem]
vl_y[is_rem] += (v_ns_speed * kick_dir_x)[is_rem]
vl_z[is_rem] += (v_ns_speed * kick_dir_y)[is_rem]

# observables
pi_rel = 1 / dl - 1 / ds
thetaE = (kappa * mass * pi_rel) ** 0.5
mu_rel = (pm_y**2 + pm_z**2) ** 0.5

wt = mu_rel * thetaE * mf

piE = pi_rel / thetaE
tE = thetaE / mu_rel * days_per_year

# output
# fmt: off
output = np.c_[dl, ds, lens_pos, pm_y, pm_z, vl_y, vl_z, vs_y, vs_z, mass, thetaE, piE, tE, stype, wt]
header = "dl, ds, lens_pos, pm_y, pm_z, vl_y, vl_z, vs_y, vs_z, mass, thetaE, piE, tE, stype, wt".split(",")
header = [f"{h:>10s}" for h in header]
fmt = ["%10.3e" for _ in header]
fmt[0] = "%10.4e"; fmt[1] = "%10.4e"; fmt[2] = "%10d"; fmt[-2] = "%10d"
np.savetxt(
    args.file.replace(".res", f".kick{int(args.kick)}.gwtc3"),
    output,
    header=" ".join(header),
    comments="",
    fmt=fmt,
)
