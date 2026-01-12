import numpy as np
from numpy.typing import NDArray

ArrayLike = NDArray[np.float64]


def mroz17_imf(mass: ArrayLike) -> ArrayLike:
    """
    Calculates the Initial Mass Function (IMF) from Mroz et al. (2017).
    Represents dN/d(log M)
    """
    mass = np.asarray(mass, dtype=float)
    conditions = [
        mass < 0.08,
        (mass >= 0.08) & (mass < 0.5),
        mass >= 0.5,
    ]
    norm2 = 0.08**-0.8 / 0.08**-1.3
    norm3 = norm2 * (0.5**-1.3 / 0.5**-2.0)
    choices = [
        mass**-0.8,
        norm2 * mass**-1.3,
        norm3 * mass**-2.0,
    ]
    imf = np.select(conditions, choices, default=0.0)
    return imf * mass


def chabrier03_imf(mass: ArrayLike) -> ArrayLike:
    """
    Calculates the Chabrier (2003) Initial Mass Function (IMF).
    Represents dN/d(log M)
    """
    mass = np.asarray(mass, dtype=float)
    log_mass = np.log10(mass)
    a_ln = 0.158
    log_mc = np.log10(0.079)
    sigma_log_m = 0.69
    a_pl = 4.43e-2
    alpha_pl = -1.3

    conditions = [mass < 1.0, mass >= 1.0]
    choices = [
        a_ln * np.exp(-((log_mass - log_mc) ** 2) / (2.0 * sigma_log_m**2)),
        a_pl * mass**alpha_pl,
    ]
    return np.select(conditions, choices, default=0.0)


def chabrier03_pdmf(mass: ArrayLike) -> ArrayLike:
    """
    Calculates the Chabrier (2003) Present Day Mass Function (PDMF).
    Represents dN/d(log M).
    """
    mass = np.asarray(mass, dtype=float)
    log_mass = np.log10(mass)
    conditions = [
        mass < 1.0,
        (mass >= 1.0) & (mass < 10**0.54),
        (mass >= 10**0.54) & (mass < 10**1.26),
        (mass >= 10**1.26) & (mass < 10**1.8),
    ]
    choices = [
        0.158 * np.exp(-((log_mass - np.log10(0.079)) ** 2) / (2.0 * 0.69**2)),
        4.4e-2 * mass**-4.37,
        1.5e-2 * mass**-3.53,
        2.5e-4 * mass**-2.11,
    ]
    return np.select(conditions, choices, default=0.0)


def gwtc3_mf(mass: ArrayLike) -> ArrayLike:
    """
    Calculates the the GWTC-3 BH Mass Funciton, adopted from https://arxiv.org/abs/2111.03634
    Represents dN/d(log M).
    """
    mass = np.asarray(mass, dtype=float)
    delta_m = 4.62
    m_max = 87.73
    m_min = 5.06
    alpha = -3.66
    mu_m = 31.59
    sigma_m = 5.51
    lambda_peak = 0.034

    power_law = mass**alpha
    norm = (alpha + 1) / (m_max ** (alpha + 1) - m_min ** (alpha + 1))
    power_law *= norm

    gaussian = np.exp(-0.5 * (mass - mu_m) ** 2 / sigma_m**2)
    gaussian /= np.sqrt(2 * np.pi) * sigma_m

    smoothing = np.zeros_like(mass)
    mask = (mass >= m_min) & (mass < m_min + delta_m)
    m_masked = mass[mask]
    exponent = delta_m / (m_masked - m_min) + delta_m / (m_masked - m_min - delta_m)

    smoothing[mask] = 1 / (1 + np.exp(exponent))
    smoothing[mass >= m_min + delta_m] = 1.0

    mixed_model = (1 - lambda_peak) * power_law + lambda_peak * gaussian
    return mixed_model * smoothing * mass


if __name__ == "__main__":
    logm1 = np.logspace(-2, 2, 500)
    logm2 = np.logspace(0.5, 2.5, 500)

    imf_mroz = mroz17_imf(logm1)
    imf_chabrier = chabrier03_imf(logm1)
    pdmf_chabrier = chabrier03_pdmf(logm1)
    spec_gwtc3 = gwtc3_mf(logm2)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, tight_layout=True)
    ax.plot(logm1, imf_mroz, label="Mroz et al. (2017) IMF")
    ax.plot(logm1, imf_chabrier, label="Chabrier (2003) IMF", ls="--")
    ax.plot(logm1, pdmf_chabrier, label="Chabrier (2003) PDMF", ls=":")
    ax.plot(logm2, spec_gwtc3, label="GWTC-3 MF", color="k")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Mass ($M_\odot$)")
    ax.set_ylabel(r"Relative Number per log Mass Interval")
    ax.set_title("Astrophysical Mass Functions")
    ax.set_ylim(1e-4, 5)
    ax.legend()
    plt.tight_layout()
    plt.show()
