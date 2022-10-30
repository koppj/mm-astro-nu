"A simulation of the combined detection of supernova neutrinos at DUNE and at Hyper-K assuming incoherent scattering"
import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate, optimize
from moment import *

# Emission data from simulation (taken from 0912.0260).
# Col 0: time in s, col 1:luminosity in 1e51 ergs,
# col 2: average energy in MeV, col 3: alpha parameter
nu_e = np.loadtxt('sn-data/Sf/neutrino_signal_nu_e', usecols=(0,1,2,5)).transpose()
nubar_e = np.loadtxt('sn-data/Sf/neutrino_signal_nubar_e', usecols=(0,1,2,5)).transpose()
nu_x = np.loadtxt('sn-data/Sf/neutrino_signal_nu_x', usecols=(0,1,2,5)).transpose()

theta12 = 0.5836381018669038    #mixing angles
theta13 = 0.14957471689591403
theta23 = 0.8552113334772213

MeV = 1e6

chi_thrs = 4.60517    #Chi-square threshold for 90% confidence for 2 dof

U = PMNS(theta12, theta13, theta23)    #PMNS matrix

muc = 1e-14*mu_b    #scaling of magnetic moment
B = B_gal

t = nu_e[0]

thrs = 3    #Detection energy threshold in MeV
d = 10*kpc    #Distance from the SN
T = d/c    #Time for neutrinos to reach the Earth

# Cross-sections for Hyper-K assuming linear scaling with the energy
sigma_nue = 18.6*0.511*1e-49    #Cross section in m**2/MeV
sigma_nubar_e = 7.8*0.511*1e-49
sigma_nux = 3*0.511*1e-49
sigma_nubar_x = 2.6*0.511*1e-49

sigma_hk = [sigma_nue, sigma_nux, sigma_nux]
sigma_bar_hk = [sigma_nubar_e, sigma_nubar_x, sigma_nubar_x]

# Cross-sections for DUNE from hep-ph/0307244.
# CC: charged current, NC: neutral current, ES: elastic scattering
sigma_e_CC = np.loadtxt('./cross_sections/nu_e_CC.csv').transpose()
sigma_bare_CC = np.loadtxt('./cross_sections/nubar_CC.csv').transpose()
sigma_e_ES = np.loadtxt('./cross_sections/nu_e_ES.csv').transpose()
sigma_bare_ES = np.loadtxt('./cross_sections/nu_bar_e_ES.csv').transpose()
sigma_x_NC = np.loadtxt('./cross_sections/nu_NC.csv').transpose()
sigma_barx_NC = np.loadtxt('./cross_sections/nu_bar_NC.csv').transpose()
sigma_x_ES = np.loadtxt('./cross_sections/nu_x_ES.csv').transpose()
sigma_barx_ES = np.loadtxt('./cross_sections/nu_bar_x_ES.csv').transpose()

sigma_dn = [[sigma_e_CC, sigma_e_ES], [sigma_x_NC, sigma_x_ES]]
sigma_bar_dn = [[sigma_bare_CC, sigma_bare_ES], [sigma_barx_NC, sigma_barx_ES]]

def sigma_extract (sigma, x):
    "Returns the cross section for neutrino energy x in m^2"
    if x < sigma[0][0] or x > sigma[0][-1]:
        return 0
    else:
        return np.interp(x, sigma[0], sigma[1])*1e-47

n = 374e9*mol*10/18    #Number of target electrons in a 374 kt water Cerenkov detector
n_ar = 40e9*mol/39.95    #number of Argon atoms in 40kt fiducial mass
n_e = 40e9*mol*18/39.95    #number of electrons

n_el = 1e6    #Electron density in ISM in 1/m^3

binning = np.linspace(-5e-3, 2e-2, 6)

def lim_inf (En, mN):
    "Inferior integration limit for the cross section"
    return ((2*En**2*(me/MeV)-(En+(me/MeV))*mN**2)/((me/MeV)*(2*En+(me/MeV)))
            - np.sqrt((En**2*(4*En**2*(me/MeV)**2-4*(me/MeV)*(En+(me/MeV))*mN**2+mN**4))
                      /((me/MeV)**2*(2*En+(me/MeV))**2)))/2

def lim_sup (En, mN):
    "Superior integration limit for the cross section"
    return ((2*En**2*(me/MeV)-(En+(me/MeV))*mN**2)/((me/MeV)*(2*En+(me/MeV)))
            + np.sqrt((En**2*(4*En**2*(me/MeV)**2-4*(me/MeV)*(En+(me/MeV))*mN**2+mN**4))
                      /((me/MeV)**2*(2*En+(me/MeV))**2)))/2

def sigma2 (En, mu, mN):
    "Analytically integrated cross section"
    return (mu**2*al/(8*En**2*(me/MeV)**2)
            * (2*(me/MeV)*(-2*(me/MeV)*(4*En+(me/MeV))+mN**2)
               *np.sqrt(En**2*(4*En**2*(me/MeV)**2-4*(me/MeV)*(En+(me/MeV))*mN**2+mN**4)/((me/MeV)**2*(2*En+(me/MeV))**2))
               - (8*En**2*(me/MeV)**2-2*(me/MeV)*(2*En+(me/MeV))*mN**2+mN**4)
               * (np.log(lim_inf(En, mN)) - np.log(lim_sup(En, mN)))))/(hbar*c**3*epsilon0)

def spec (x, mean, alph, mu, mN):
    "Energy spectrum of SN neutrinos with magnetic moment mu in MeV"
    if (lim_inf(x*MeV, mN*MeV) > 0 and lim_sup(x*MeV, mN*MeV) > 0 and mu != 0):
        return (x**alph * np.exp(-(alph+1)*x/mean)
                * ((1+alph)/mean)**(1+alph)/special.gamma(1+alph)
                * np.exp(-sigma2(x*MeV, mu, mN*MeV)*d*n_el))
    else:
        return x**alph * np.exp(-(alph+1)*x/mean) * ((1+alph)/mean)**(1+alph)/special.gamma(1+alph)

def frac_hk(mean, alph, thrsh, mu, mN):
    "Convoluted flux above the detection energy threshold with the energy"
    en_thrs = mN + mN**2*MeV/(2*me)
    if mean == 0:
        return 0
    # elif mu == 0:
    #     return mean * special.gammaincc(2+alph, thrs*(1+alph)/mean)
    elif thrs >= en_thrs:
        return integrate.quad(lambda x: spec(x, mean, alph, mu, mN) * x, thrsh, np.infty, args=(), limit=100)[0]
    else:
        return (integrate.quad(lambda x: spec(x, mean, alph, 0, mN) * x, thrsh, en_thrs + 1e-10, args=(), limit=100)[0]
                + integrate.quad(lambda x: spec(x, mean, alph, mu, mN) * x, en_thrs + 1e-10, np.infty, args=(), limit=100)[0])

def frac_dn(mean, alph, thrsh, mu, mN, sigma):
    "Convoluted flux above the detection energy threshold with the cross-section"
    en_thrs = mN + mN**2*MeV/(2*me)
    if mean == 0:
        return 0
    elif thrs >= en_thrs:
        return integrate.quad(lambda x: spec(x, mean, alph, mu, mN) * sigma_extract(sigma, x),
                              thrsh, np.infty, args=(), limit=100)[0]
    else:
        return (integrate.quad(lambda x: spec(x, mean, alph, 0, mN) * sigma_extract(sigma, x),
                               thrsh, en_thrs+ 1e-10, args=(), limit=100)[0]
                + integrate.quad(lambda x: spec(x, mean, alph, mu, mN) * sigma_extract(sigma, x),
                                 en_thrs + 1e-10, np.infty, args=(), limit=100)[0])

idx_in = np.where(t>=-5e-3)[0][0]
idx_fn = np.where(t<=2e-2)[0][-1]

factor = 1e51 * erg/(4*np.pi*d**2*1e6*eV)    #Conversion factor from erg to Mev times 1/area

# Number of emitted neutrinos (Luminosity/(Mean Energy * Total Surface))
N_e = [nu_e[1][i] * factor/nu_e[2][i] if nu_e[2][i] != 0 else 0 for i in range(idx_in, idx_fn+1)]
N_ebar = [nubar_e[1][i] * factor/nubar_e[2][i] if nubar_e[2][i] != 0 else 0 for i in range(idx_in, idx_fn+1)]
N_x = [nu_x[1][i] * factor/nu_x[2][i]  if nu_x[2][i] != 0 else 0 for i in range(idx_in, idx_fn+1)]

def chi_square (mu, mN, mh, hk=True, dn=True):
    """Computes chi-square of dataset with magnetic moment mu assuming mu=0

    Parameters:
        mu (float):  The neutrino magnetic moment, in ev/T
        mN (float):  The right-handed neutrino mass, in MeV
        mh (str):    The mass hierarchy ("NH" for normal hierarchy, "IH" for inverted hierarchy)

    Optional Keyword arguments:
        hk (bool):  Include Hyper-K in the analysis (default: True)
        dn (bool):  Include DUNE in the analysis (default: True)

    Returns:
        chi (float):  The chi-square value
    """

    if mh == "NH":
        # Neutrino masses
        m = [0, np.sqrt(7.59e-17), np.sqrt(2.32e-15)]

        # Flavor-mass state correspondence at high densities
        i_e = 3
        i_mu = 1
        i_tau = 2
        i_be = 1
        i_bmu = 2
        i_btau = 3
    elif mh == "IH":
        m = [np.sqrt(2.32e-15), np.sqrt(2.32e-15+7.59e-17), 0]

        i_e = 2
        i_mu = 1
        i_tau = 3
        i_be = 3
        i_bmu = 2
        i_btau = 1
    else:
        raise ValueError(f'\'{mh}\' is not a valid value for mh; supported values are \'NH\', \'IH\'')

    # Hyper-K
    if hk:
        # Fluxes of initial nu_e, nu_e bar and nu_x arriving at Earth w/o magnetic conversion
        frac_e_hk = [frac_hk(nu_e[2][i], nu_e[3][i], thrs, 0, mN) for i in range(idx_in, idx_fn+1)]
        frac_bare_hk = [frac_hk(nubar_e[2][i], nubar_e[3][i], thrs, 0, mN) for i in range(idx_in, idx_fn+1)]
        frac_x_hk = [frac_hk(nu_x[2][i], nu_x[3][i], thrs, 0, mN) for i in range(idx_in, idx_fn+1)]

        "With magnetic conversion"
        frac_emu_hk = [frac_hk(nu_e[2][i], nu_e[3][i], thrs, U[1,i_e-1]*mu, mN) for i in range(idx_in, idx_fn+1)]
        frac_baremu_hk = [frac_hk(nubar_e[2][i], nubar_e[3][i], thrs, U[1,i_be-1]*mu, mN) for i in range(idx_in, idx_fn+1)]
        frac_xmu_hk = [[frac_hk(nu_x[2][i], nu_x[3][i], thrs, U[1,j]*mu, mN) for i in range(idx_in, idx_fn+1)] for j in range(3)]
        "The index j indicates the mass state of nu_x"

        # Number of neutrinos interacting with the detector for times t
        dec_hk = [[((U[j,i_mu-1]**2+U[j,i_tau-1]**2) * frac_x_hk[i] * N_x[i]
                 + U[j,i_e-1]**2 * frac_e_hk[i] * N_e[i])
                * sigma_hk[j] * n * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]

        dec_bar_hk = [[(U[j,i_be-1]**2 * frac_bare_hk[i] * N_ebar[i]
                     + (U[j,i_bmu-1]**2 + U[j,i_btau-1]**2) * frac_x_hk[i] * N_x[i])
                    * sigma_bar_hk[j] * n * (t[i+1+idx_in]-t[i+idx_in])
                    for i in range(idx_fn+1-idx_in)] for j in range(3)]

        # With magnetic conversion
        dec_mu_hk = [[((U[j,i_mu-1]**2 * frac_xmu_hk[i_mu-1][i]
                     + U[j,i_tau-1]**2 * frac_xmu_hk[i_tau-1][i]) * N_x[i]
                    + U[j,i_e-1]**2 * frac_emu_hk[i] * N_e[i])
                   * sigma_hk[j] * n * (t[i+1+idx_in]-t[i+idx_in])
                   for i in range(idx_fn+1-idx_in)] for j in range(3)]

        dec_bar_mu_hk = [[(U[j,i_be-1]**2 * frac_baremu_hk[i] * N_ebar[i]
                        + (U[j,i_bmu-1]**2 * frac_xmu_hk[i_bmu-1][i]
                           + U[j,i_btau-1]**2 * frac_xmu_hk[i_btau-1][i]) * N_x[i])
                       * sigma_bar_hk[j] * n * (t[i+1+idx_in]-t[i+idx_in])
                       for i in range(idx_fn+1-idx_in)] for j in range(3)]

        # Bins
        bins_hk = np.array([[sum(dec_hk[j][np.where(t>=binning[i])[0][0]-idx_in
                                     :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                         for j in range(3)])
        bins_hk = np.array([np.append(bins_hk[i], bins_hk[i][-1]) for i in range(3)])

        bins_bar_hk = np.array([[sum(dec_bar_hk[j][np.where(t>=binning[i])[0][0]-idx_in
                                             :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                             for j in range(3)])
        bins_bar_hk = np.array([np.append(bins_bar_hk[i], bins_bar_hk[i][-1]) for i in range(3)])

        bins_mu_hk = np.array([[sum(dec_mu_hk[j][np.where(t>=binning[i])[0][0]-idx_in
                                           :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                            for j in range(3)])
        bins_mu_hk = np.array([np.append(bins_mu_hk[i], bins_mu_hk[i][-1]) for i in range(3)])

        bins_bar_mu_hk = np.array([[sum(dec_bar_mu_hk[j][np.where(t>=binning[i])[0][0]-idx_in
                                                   :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                for j in range(3)])
        bins_bar_mu_hk = np.array([np.append(bins_bar_mu_hk[i], bins_bar_mu_hk[i][-1]) for i in range(3)])

        # Histogram for Hyper-K
        # fig, ax = plt.subplots()
        # fig.set_size_inches(6, 5)
    
        # ax.plot(binning*1e3, np.sum(bins_hk + bins_bar_hk, axis=0),
        #         ds='steps-post', label=f'$\\mu_\\nu = 0$')
        # ax.plot(binning*1e3, np.sum(bins_mu_hk + bins_bar_mu_hk, axis=0),
        #         ds='steps-post', label=f'$\\mu_\\nu = {mu/mu_b} \\mu_B$', ls='--')
        # ax.set_xlim(-5, 20)
        # ax.set_ylim(0, 50)
        # ax.set_ylabel('Events per bin', fontsize=15)
        # ax.set_xlabel('t (ms)', fontsize=15)
        # ax.tick_params(axis='both', which='both', labelsize=12)
        # ax.legend(fontsize=12)

    else:
        bins_hk = np.zeros((3,6))
        bins_bar_hk = np.zeros((3,6))
        bins_mu_hk = np.zeros((3,6))
        bins_bar_mu_hk = np.zeros((3,6))

    # DUNE

    if dn:
        # Fluxes of initial nu_e, nu_e bar and nu_x arriving at Earth convoluted with cross-sections w/o magnetic conversion
        # The index l represents the interaction type (0:NC or CC, 1:ES)
        # The index j represents the flavor (0:nu_e, 1:nu_x) in the cross section
        frac_e_dn = [[[frac_dn(nu_e[2][i], nu_e[3][i], thrs, 0, mN, sigma_dn[j][l])
                       for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)]
        frac_bare_dn = [[[frac_dn(nubar_e[2][i], nubar_e[3][i], thrs, 0, mN, sigma_bar_dn[j][l])
                          for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)]
        frac_x_dn = [[[frac_dn(nu_x[2][i], nu_x[3][i], thrs, 0, mN, sigma_dn[j][l])
                       for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)]
        frac_barx_dn = [[[frac_dn(nu_x[2][i], nu_x[3][i], thrs, 0, mN, sigma_bar_dn[j][l])
                          for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)]

        # With magnetic conversion
        frac_emu_dn = [[[frac_dn(nu_e[2][i], nu_e[3][i], thrs, U[1,i_e-1]*mu, mN, sigma_dn[j][l]) for i in range(idx_in, idx_fn+1)]
                     for l in range(2)] for j in range(2)]
        frac_baremu_dn = [[[frac_dn(nubar_e[2][i], nubar_e[3][i], thrs, U[1,i_be-1]*mu, mN, sigma_bar_dn[j][l])
                            for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)]
        frac_xmu_dn = [[[[frac_dn(nu_x[2][i], nu_x[3][i], thrs, U[1,j]*mu, mN, sigma_dn[k][l])
                          for i in range(idx_in, idx_fn+1)] for j in range(3)] for l in range(2)] for k in range(2)]
        frac_barxmu_dn = [[[[frac_dn(nu_x[2][i], nu_x[3][i], thrs, U[1,j]*mu, mN, sigma_bar_dn[k][l]) for i in range(idx_in, idx_fn+1)]
                         for j in range(3)] for l in range(2)] for k in range(2)]

        # Number of neutrinos interacting with the detector for times t
        dec_dn = [[((U[j,i_mu-1]**2+U[j,i_tau-1]**2) * (frac_x_dn[min(j,1)][0][i] * n_ar + frac_x_dn[min(j,1)][1][i] * n_e) * N_x[i]
                 + U[j,i_e-1]**2 * (frac_e_dn[min(j,1)][0][i] * n_ar + frac_e_dn[min(j,1)][1][i] * n_e) * N_e[i])
                * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]

        dec_bar_dn = [[(U[j,i_be-1]**2 * (frac_bare_dn[min(j,1)][0][i] * n_ar + frac_bare_dn[min(j,1)][1][i] * n_e) * N_ebar[i]
                     + (U[j,i_bmu-1]**2 + U[j,i_btau-1]**2)
                     * (frac_barx_dn[min(j,1)][0][i] * n_ar + frac_barx_dn[min(j,1)][1][i] * n_e) * N_x[i])
                    * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]

        # With magnetic conversion
        dec_mu_dn = [[((U[j,i_mu-1]**2 * (frac_xmu_dn[min(j,1)][0][i_mu-1][i] * n_ar + frac_xmu_dn[min(j,1)][1][i_mu-1][i] * n_e)
                     + U[j,i_tau-1]**2 * (frac_xmu_dn[min(j,1)][0][i_tau-1][i] * n_ar + frac_xmu_dn[min(j,1)][1][i_tau-1][i] * n_e)) * N_x[i]
                    + U[j,i_e-1]**2 * (frac_emu_dn[min(j,1)][0][i] * n_ar + frac_emu_dn[min(j,1)][1][i] * n_e) * N_e[i])
                   * (t[i+1+idx_in]-t[i+idx_in])
                   for i in range(idx_fn+1-idx_in)] for j in range(3)]

        dec_bar_mu_dn = [[(U[j,i_be-1]**2 * (frac_baremu_dn[min(j,1)][0][i] * n_ar + frac_baremu_dn[min(j,1)][1][i] * n_e) * N_ebar[i]
                        + (U[j,i_bmu-1]**2 * (frac_barxmu_dn[min(j,1)][0][i_bmu-1][i] * n_ar + frac_barxmu_dn[min(j,1)][1][i_bmu-1][i] * n_e)
                           + U[j,i_btau-1]**2 * (frac_barxmu_dn[min(j,1)][0][i_btau-1][i] * n_ar
                                                 + frac_barxmu_dn[min(j,1)][1][i_btau-1][i] * n_e)) * N_x[i])
                       * (t[i+1+idx_in]-t[i+idx_in])
                       for i in range(idx_fn+1-idx_in)] for j in range(3)]

        # Bins
        bins_dn = np.array([[sum(dec_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                     :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                         for j in range(3)])
        bins_dn = np.array([np.append(bins_dn[i], bins_dn[i][-1]) for i in range(3)])

        bins_bar_dn = np.array([[sum(dec_bar_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                             :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                             for j in range(3)])
        bins_bar_dn = np.array([np.append(bins_bar_dn[i], bins_bar_dn[i][-1]) for i in range(3)])

        bins_mu_dn = np.array([[sum(dec_mu_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                           :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                            for j in range(3)])
        bins_mu_dn = np.array([np.append(bins_mu_dn[i], bins_mu_dn[i][-1]) for i in range(3)])

        bins_bar_mu_dn = np.array([[sum(dec_bar_mu_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                                   :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                for j in range(3)])
        bins_bar_mu_dn = np.array([np.append(bins_bar_mu_dn[i], bins_bar_mu_dn[i][-1]) for i in range(3)])

        # Histogram for DUNE
        # fig2, ax2 = plt.subplots()
        # fig2.set_size_inches(6, 5)

        # ax2.plot(binning*1e3, np.sum(bins_dn + bins_bar_dn, axis=0),
        #         ds='steps-post', label=f'$\\mu_\\nu = 0$')
        # ax2.plot(binning*1e3, np.sum(bins_mu_dn + bins_bar_mu_dn, axis=0),
        #         ds='steps-post', label=f'$\\mu_\\nu = {mu/mu_b} \\mu_B$', ls='--')
        # ax2.set_xlim(-5, 20)
        # ax2.set_ylim(0, 50)
        # ax2.set_ylabel('Events per bin', fontsize=15)
        # ax2.set_xlabel('t (ms)', fontsize=15)
        # ax2.tick_params(axis='both', which='both', labelsize=12)
        # ax2.legend(fontsize=12)

    else:
        bins_dn = np.zeros((3,6))
        bins_bar_dn = np.zeros((3,6))
        bins_mu_dn = np.zeros((3,6))
        bins_bar_mu_dn = np.zeros((3,6))

    # Combined histogram
    # fig3, ax3 = plt.subplots()
    # fig3.set_size_inches(6, 5)

    # ax3.plot(binning*1e3, np.sum(bins_dn + bins_bar_dn + bins_hk + bins_bar_hk, axis=0),
    #         ds='steps-post', label=f'$\\mu_\\nu = 0$')
    # ax3.plot(binning*1e3, np.sum(bins_mu_dn + bins_bar_mu_dn + bins_mu_hk + bins_bar_mu_hk, axis=0),
    #         ds='steps-post', label=f'$\\mu_\\nu = {mu/mu_b} \\mu_B$', ls='--')
    # ax3.set_xlim(-5, 20)
    # ax3.set_ylim(0, 50)
    # ax3.set_ylabel('Events per bin', fontsize=15)
    # ax3.set_xlabel('t (ms)', fontsize=15)
    # ax3.tick_params(axis='both', which='both', labelsize=12)
    # ax3.legend(fontsize=12)

    def chi_temp (a):
        c = a**2/0.1**2
        if hk:
            c += np.sum((np.sum(bins_hk + bins_bar_hk, axis=0)[:-1] - (1+a)*np.sum(bins_mu_hk + bins_bar_mu_hk, axis=0)[:-1])**2
                        /((1+a)*np.sum(bins_mu_hk + bins_bar_mu_hk, axis=0)[:-1]))
        if dn:
            c += np.sum((np.sum(bins_dn + bins_bar_dn, axis=0)[:-1] - (1+a)*np.sum(bins_mu_dn + bins_bar_mu_dn, axis=0)[:-1])**2
                        /((1+a)*np.sum(bins_mu_dn + bins_bar_mu_dn, axis=0)[:-1]))
        return c

    chi = optimize.minimize(chi_temp, 0).fun

    return chi
