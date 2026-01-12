"""
Monte Carlo simulation for (lens, source) pairs
"""

import argparse
import io
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

# Gaia velocity table
GAIA_TABLE_DATA = """-6  -5.936  -0.143  2.525  2.016  -250.2  -6.0  106.4   85.0
-4  -6.024  -0.232  2.653  2.336  -252.3  -9.7  111.1   97.8
-2  -5.873   0.252  3.069  2.410  -240.3  10.3  125.5   98.6
 0  -5.741  -0.127  3.029  2.661  -223.1  -4.9  117.7  103.4
 2  -5.969   0.045  3.134  2.721  -224.7   1.7  118.0  102.4
 4  -5.999  -0.191  3.040  2.449  -221.2  -7.0  112.1   90.3
 6  -6.123   0.045  2.787  2.212  -220.5   1.6  100.3   79.7"""


def rho_bulge(d: np.ndarray, l: float, b: float, dgc: float) -> np.ndarray:
    """Bulge density distribution using revised COBE model (Dwek et al. 1995)."""
    x = dgc - d * np.cos(b) * np.cos(l)
    y = d * np.cos(b) * np.sin(l)
    z = d * np.sin(b)

    # Bar angle
    eta = (20.0 / 180.0) * np.pi
    xp = x * np.cos(eta) + y * np.sin(eta)
    yp = -x * np.sin(eta) + y * np.cos(eta)
    zp = z

    # Scale parameters (scaled from 8.5 kpc to current dgc)
    x0 = 1.58 * dgc / 8.5
    y0 = 0.62 * dgc / 8.5
    z0 = 0.43 * dgc / 8.5

    # Density profile
    rs2 = np.sqrt(((xp / x0) ** 2 + (yp / y0) ** 2) ** 2 + (zp / z0) ** 4)

    # From Batista et al. (2011) normalized
    rhob = 1.23 * np.exp(-0.5 * rs2)
    return rhob


def rho_disk(d: np.ndarray, l: float, b: float, dgc: float) -> np.ndarray:
    """Disk density distribution from Bennett et al. (2014)."""
    x = dgc - d * np.cos(b) * np.cos(l)
    y = d * np.cos(b) * np.sin(l)
    z = d * np.sin(b)
    r = np.sqrt(x**2 + y**2)

    # Bennett 2014 parameters
    norm = 0.93  # M_sun/pc^3
    hr_plus = 2530.0  # pc
    hr_minus = 1320.0  # pc
    ac = np.sqrt(r**2 + (abs(z) / 0.079) ** 2) * 1000
    bc = 0.5
    ac2 = ac**2
    bc2 = bc**2

    rhod4 = norm * (
        np.exp(-np.sqrt(bc2 + ac2 / hr_plus**2)) - np.exp(-np.sqrt(bc2 + ac2 / hr_minus**2))
    )
    return rhod4


def calculate_density_factor_tab(dgc: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-calculate density factors for a grid of radii.
    Returns (r_grid, density_factors) for interpolation.
    """
    # Disk parameters from Bennett 2014
    norm = 0.93  # M_sun/pc^3
    hr_plus = 2530.0  # pc
    hr_minus = 1320.0  # pc
    bc = 0.5
    bc2 = bc**2

    # Create radius grid from 0 to 15 kpc (should cover all reasonable lens distances)
    r_grid = np.linspace(0, 8, 81)  # 0.1 kpc steps
    r_pc = r_grid * 1000.0

    # Pre-calculate for solar radius
    dgc_pc = dgc * 1000.0
    nz_max = 10000
    z_values = np.arange(1, nz_max + 1)

    # Calculate solar radius normalization
    ac2_sun = dgc_pc**2 + (z_values / 0.079) ** 2
    rho_sun = norm * (
        np.exp(-np.sqrt(bc2 + ac2_sun / hr_plus**2)) - np.exp(-np.sqrt(bc2 + ac2_sun / hr_minus**2))
    )
    sigma_sun = np.sum(2.0 * rho_sun)
    sheight_sun = np.sum(2.0 * rho_sun * z_values) / sigma_sun
    fac_sun = sigma_sun * sheight_sun

    # Calculate for each radius in grid
    density_factors = np.zeros_like(r_grid)

    for i, r in enumerate(r_pc):
        ac2 = r**2 + (z_values / 0.079) ** 2
        rho = norm * (
            np.exp(-np.sqrt(bc2 + ac2 / hr_plus**2)) - np.exp(-np.sqrt(bc2 + ac2 / hr_minus**2))
        )
        sigma = np.sum(2.0 * rho)
        sheight = np.sum(2.0 * rho * z_values) / sigma
        fac = sigma * sheight
        density_factors[i] = fac / fac_sun

    # Clamp to reasonable range
    density_factors = np.clip(density_factors, 0.1, 10.0)

    return r_grid, density_factors


def getv(ra, dec, t0):
    """
    Calculate Earth's velocity
    """
    ecc = 0.0167
    vernal = 2719.55
    offset = 75
    peri = vernal - offset

    spring = np.array([1.0, 0.0, 0.0])
    summer = np.array([0.0, 0.9174, 0.3971])

    phi = (1 - offset / 365.25) * 2 * np.pi
    psi = get_psi(phi, ecc)

    costh = (np.cos(psi) - ecc) / (1 - ecc * np.cos(psi))
    sinth = -np.sqrt(1 - costh**2)

    xpos = spring * costh + summer * sinth
    ypos = -spring * sinth + summer * costh

    north = np.array([0, 0, 1])
    rad = np.array(
        [
            np.cos(ra * np.pi / 180) * np.cos(dec * np.pi / 180),
            np.sin(ra * np.pi / 180) * np.cos(dec * np.pi / 180),
            np.sin(dec * np.pi / 180),
        ]
    )

    east = np.cross(north, rad)
    east = east / np.linalg.norm(east)
    north = np.cross(rad, east)

    # Calculate Earth's positions
    phi_t2 = (t0 + 1 - peri) / 365.25 * 2 * np.pi
    psi_t2 = get_psi(phi_t2, ecc)
    sun_t2 = xpos * (np.cos(psi_t2) - ecc) + ypos * np.sin(psi_t2) * np.sqrt(1 - ecc**2)

    phi_t1 = (t0 - 1 - peri) / 365.25 * 2 * np.pi
    psi_t1 = get_psi(phi_t1, ecc)
    sun_t1 = xpos * (np.cos(psi_t1) - ecc) + ypos * np.sin(psi_t1) * np.sqrt(1 - ecc**2)

    # Calculate velocity components
    qn2 = np.dot(sun_t2, north)
    qe2 = np.dot(sun_t2, east)
    qn1 = np.dot(sun_t1, north)
    qe1 = np.dot(sun_t1, east)

    vn0 = ((qn2 - qn1) / 2) * 365.25 * 4.74
    ve0 = ((qe2 - qe1) / 2) * 365.25 * 4.74

    return vn0, ve0


def get_psi(phi, ecc):
    """
    Solve Kepler's equation: psi = phi + ecc * sin(psi)
    """
    psi = (phi + np.pi) % (2 * np.pi) - np.pi
    for _ in range(4):
        fun = psi - ecc * np.sin(psi)
        dif = phi - fun
        der = 1 - ecc * np.cos(psi)
        psi = psi + dif / der
    return psi


def sim_len_source(
    nsample: int,
    lens_pos: str,
    l_deg: float,
    b_deg: float,
    ds_min: float = 0.0,
    ds_max: float = 12.0,
    dl_min: float = 0.0,
    dl_max: float = 12.0,
    dgc: float = 8.0,
    t0: float = 9500.0,
    seed: int = 0,
    output: str = None,
    comments: str = "",
) -> np.ndarray:
    """Main Monte Carlo simulation for microlensing events"""

    print("=============================================================================")
    print(f"Observation field: l={l_deg:.6f}°, b={b_deg:.6f}°")
    print(f"Nsample: {nsample}")
    print(f"Running {lens_pos} simulation...")

    np.random.seed(seed)
    # Load Gaia velocity table
    gaia_data = np.loadtxt(io.StringIO(GAIA_TABLE_DATA))
    l_vals, mu_l, mu_b, sig_mu_l, sig_mu_b, v_l, v_b, sig_vy_vals, sig_vb_vals = gaia_data.T

    r_grid, density_factors_tab = calculate_density_factor_tab(dgc)

    # Setup observation field
    l_rad = l_deg * np.pi / 180.0
    b_rad = b_deg * np.pi / 180.0

    # Setup velocity components
    vrot = 220.0
    pecy = 12.0
    pecz = 7.0

    # Sampling lens and source distances baded on density profiles
    ds_grid = np.linspace(ds_min, ds_max, 2001)
    dl_grid = np.linspace(dl_min, dl_max, 2001)

    if lens_pos == "bulge":
        dl_pdf = rho_bulge(dl_grid, l_rad, b_rad, dgc) * dl_grid**2
    elif lens_pos == "disk":
        dl_pdf = rho_disk(dl_grid, l_rad, b_rad, dgc) * dl_grid**2
    elif lens_pos == "bulge+disk":
        dl_pdf = (
            rho_bulge(dl_grid, l_rad, b_rad, dgc) + rho_disk(dl_grid, l_rad, b_rad, dgc)
        ) * dl_grid**2

    dl_cdf = np.cumsum(dl_pdf)
    dl_cdf /= dl_cdf[-1]

    ds_pdf = rho_bulge(ds_grid, l_rad, b_rad, dgc) * np.interp(ds_grid, dl_grid, dl_cdf)
    ds_cdf = np.cumsum(ds_pdf)
    ds_cdf /= ds_cdf[-1]

    # Conditional sampling
    ds = np.interp(np.random.rand(nsample), ds_cdf, ds_grid)
    max_dl_cdf = np.interp(ds, dl_grid, dl_cdf) - 1e-3
    dl = np.interp(np.random.rand(nsample) * max_dl_cdf, dl_cdf, dl_grid)

    # lens position 0->bulge; 1->disk
    pos_lens = np.zeros(nsample, dtype=int)
    if lens_pos == "disk":
        pos_lens += 1
    if lens_pos == "bulge+disk":
        rhod = rho_disk(dl, l_rad, b_rad, dgc)
        rhob = rho_bulge(dl, l_rad, b_rad, dgc)
        disk_frac = rhod / (rhod + rhob)
        pos_lens = 1 * (np.random.rand(nsample) < disk_frac)

    disk = pos_lens == 1
    bulge = pos_lens == 0

    # pre-generated random numbers for velocity components
    rds = np.random.randn(nsample, 2)
    rdl = np.random.randn(nsample, 2)

    vs_y = np.zeros(nsample)
    vs_z = np.zeros(nsample)
    vl_y = np.zeros(nsample)
    vl_z = np.zeros(nsample)

    # bulge velocity
    vy_mean_bulge = np.interp(l_deg, l_vals, v_l) + vrot + pecy
    vz_mean_bulge = np.interp(l_deg, l_vals, v_b) + pecz
    sig_vy_bulge = np.interp(l_deg, l_vals, sig_vy_vals)
    sig_vz_bulge = np.interp(l_deg, l_vals, sig_vb_vals)
    vs_y = vy_mean_bulge + rds[:, 0] * sig_vy_bulge
    vs_z = vz_mean_bulge + rds[:, 1] * sig_vz_bulge
    vl_y[bulge] = vy_mean_bulge + rdl[bulge, 0] * sig_vy_bulge
    vl_z[bulge] = vz_mean_bulge + rdl[bulge, 1] * sig_vz_bulge

    # disk velocity
    sig_vx_disk = 34.0
    sig_vy_disk = 28.0
    sig_vz_disk = 18.0
    sigtot1 = sig_vx_disk**2 + sig_vy_disk**2 + sig_vz_disk**2
    sigtot2 = sigtot1 / (2.0 * vrot)

    # lens position in Galactic coordinates
    lx = dgc - dl * np.cos(b_rad) * np.cos(l_rad)
    ly = dl * np.cos(b_rad) * np.sin(l_rad)
    lr = np.sqrt(lx**2 + ly**2)

    # surface density factor
    densityfactor = np.interp(lr[disk], r_grid, density_factors_tab)

    # Adjust lens velocity mean and dispersions
    vy_mean_disk = vrot - sigtot2 * densityfactor
    vz_mean_disk = 0.0
    sig_vy_disk *= np.sqrt(densityfactor)
    sig_vz_disk *= np.sqrt(densityfactor)

    # sampling velocities
    vl_y[disk] = vy_mean_disk + rdl[disk, 0] * sig_vy_disk
    vl_z[disk] = vz_mean_disk + rdl[disk, 1] * sig_vz_disk

    # observer velocity
    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    ra, dec = c.icrs.ra.deg, c.icrs.dec.deg  # returns (RA, Dec) in degrees
    ven, vee = getv(ra, dec, t0)
    ven = -ven
    vee = -vee

    # Convert to Galactic frame
    gangle = 60.2 * np.pi / 180.0  # Galactic angle in radians
    vez = ven * np.cos(gangle) - vee * np.sin(gangle)
    vey = ven * np.sin(gangle) + vee * np.cos(gangle)

    # rotation + peculiar motion + Earth velocity
    vobs_y = vrot + vey + pecy
    vobs_z = vez + pecz

    # transverse velocity
    vy = vl_y - (vs_y * (dl / ds) + vobs_y * (ds - dl) / ds)
    vz = vl_z - (vs_z * (dl / ds) + vobs_z * (ds - dl) / ds)

    # proper motion
    fac = ((u.km / u.s) / u.kpc).to(1 / u.yr) * (u.rad).to(u.mas)
    pm_y = vy / dl * fac
    pm_z = vz / dl * fac

    # output array
    events = np.c_[dl, ds, pos_lens, pm_y, pm_z, vl_y, vl_z, vs_y, vs_z]
    # output format
    header = "dl, ds, lens_pos, pm_y, pm_z, vl_y, vl_z, vs_y, vs_z".split(", ")
    fmt = len(header) * ["%10.3e"]
    fmt[2] = "%10d"; fmt[0] = "%10.4e"; fmt[1] = "%10.4e"
    fmt = " ".join(fmt)
    header = " ".join([f"{h:>10s}" for h in header])
    print("=============================================================================")
    print(f"Saving {len(events)} events to {output}")
    np.savetxt(output, events, fmt=fmt, header=header, comments=comments)
    print("Simulation completed")
    print("=============================================================================")

    return events


def main():
    """Main function with argument parsing."""
    # fmt: off
    parser = argparse.ArgumentParser(description="lens & source simulation")
    parser.add_argument("--lens-pos", choices=["bulge", "disk", "bulge+disk"], required=True, help="lens-position")
    parser.add_argument("--l-deg", type=float, default=-0.14, help="Galactic longitude (degrees)")
    parser.add_argument("--b-deg", type=float, default=-1.62, help="Galactic latitude (degrees)")
    parser.add_argument("--nsample", type=int, default=int(1e6), help="Nsample")
    parser.add_argument("--output", type=str, default=None, help="output (default: {len-pos}.res)")
    parser.add_argument("--ds-min", type=float, default=0.0, help="Minimum source distance (kpc)")
    parser.add_argument("--ds-max", type=float, default=12.0, help="Maximum source distance (kpc)")
    parser.add_argument("--dl-min", type=float, default=0.0, help="Minimum lens distance (kpc)")
    parser.add_argument("--dl-max", type=float, default=12.0, help="Maximum lens distance (kpc)")
    parser.add_argument("--dgc", type=float, default=8.0, help="Distance to Galactic center (kpc)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--t0", type=float, default=9500, help="Observation time (JD-2450000)")
    # fmt: on

    args = parser.parse_args()
    cmd = "# " + " ".join(sys.argv) + "\n"

    # Set output filename
    if args.output is None:
        args.output = f"{args.lens_pos}.res"

    # Run simulation
    sim_len_source(**vars(args), comments=cmd)


if __name__ == "__main__":
    main()
