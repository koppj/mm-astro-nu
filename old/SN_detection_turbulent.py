"A simulation of the detection of supernova neutrinos assuming coherent conversion on a turbulent magnetic field"
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import special, integrate, optimize, interpolate
from moment import *

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rc('text', usetex=True)

"Emission data from simulation. Col 0: time in s, col 1:luminosity in 1e51 ergs,"
"col 2: average energy in MeV, col 3: alpha parameter"
nu_e = np.loadtxt('sn-data/Sf/neutrino_signal_nu_e', usecols=(0,1,2,5)).transpose()
nubar_e = np.loadtxt('sn-data/Sf/neutrino_signal_nubar_e', usecols=(0,1,2,5)).transpose()
nu_x = np.loadtxt('sn-data/Sf/neutrino_signal_nu_x', usecols=(0,1,2,5)).transpose()

"Extracted cross sections in 10^43*cm^2"
sigma_e_CC = np.loadtxt('./cross_sections/nu_e_CC.csv').transpose()
sigma_bare_CC = np.loadtxt('./cross_sections/nubar_CC.csv').transpose()
sigma_e_ES = np.loadtxt('./cross_sections/nu_e_ES.csv').transpose()
sigma_bare_ES = np.loadtxt('./cross_sections/nu_bar_e_ES.csv').transpose()
#sigma_e_O = np.loadtxt('./cross_sections/nu_e_O.csv').transpose()
sigma_e_O = np.array([[0,0],[1e20,0]]).T # FIXME
sigma_e_O[1] *= 1e43
#sigma_bare_O = np.loadtxt('./cross_sections/nubar_e_O.csv').transpose()
sigma_bare_O = np.array([[0,0],[1e20,0]]).T # FIXME
sigma_bare_O[1] *= 1e43
#sigma_IBD = np.loadtxt('./cross_sections/IBD.csv').transpose()
sigma_IBD = np.array([[0,0],[1e20,0]]).T # FIXME
sigma_IBD[1] *= 1e43
sigma_x_NC = np.loadtxt('./cross_sections/nu_NC.csv').transpose()
sigma_barx_NC = np.loadtxt('./cross_sections/nu_bar_NC.csv').transpose()
sigma_x_ES = np.loadtxt('./cross_sections/nu_x_ES.csv').transpose()
sigma_barx_ES = np.loadtxt('./cross_sections/nu_bar_x_ES.csv').transpose()

sigma_HK = [[sigma_e_ES, sigma_e_O], [sigma_x_ES, [[0],[0]]]]
sigma_bar_HK = [[sigma_bare_ES, sigma_bare_O, sigma_IBD],
                [sigma_barx_ES, [[0],[0]], [[0],[0]]]]

sigma_DN = [[sigma_e_CC, sigma_e_ES], [sigma_x_NC, sigma_x_ES]]

sigma_bar_DN = [[sigma_bare_CC, sigma_bare_ES], [sigma_barx_NC, sigma_barx_ES]]

def sigma_extract (sigma, x):
    "Returns the cross section for neutrino energy x in m^2"
    if x < sigma[0][0] or x > sigma[0][-1]:
        return 0
    else:
        return np.interp(x, sigma[0], sigma[1])*1e-47

theta12 = 0.5836381018669038    #mixing angle
theta13 = 0.14957471689591403
theta23 = 0.8552113334772213

MeV = 1e6

# chi_thrs = 4.60517    #Chi-square threshold for 90% confidence for 2 dof
chi_thrs = 2.71    #Chi-square threshold for 90% confidence for 2 dof

U = PMNS(theta12, theta13, theta23)    #PMNS matrix

muc = 1e-13*mu_b    #scaling of magnetic moment
Bcons = B_gal

mus = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])

t = nu_e[0]

thrs = 3    #Detection threshold in MeV
d = 10*kpc    #Distance from the SN

n_e_HK = 374e9*mol*10/18    #Number of target electrons/protons in HK
n_O_HK = 374e9*mol/18    #Number of target oxygen atoms in HK
n_ar_DN = 40e9*mol/39.95    #number of target Argon atoms in DUNE
n_e_DN = 40e9*mol*18/39.95    #number of target electrons in DUNE

file_muN = open('nu_mu_limit_NO.txt', 'a')
file_muI = open('nu_mu_limit_IO.txt', 'a')
file_nusN = open('nus_limit_NO.txt', 'a')
file_nusI = open('nus_limit_IO.txt', 'a')

Nl = 1    # Number of sampled turbulence scales
Nb = 1    # Number of turbulent magnetic field strengths

lrange = np.logspace(-2, -1, Nl)
Brange = np.linspace(2, 5, Nb)

for i in range(Nl):
    for j in range(Nb):     
        "Generate the turbulent magnetic field"
        N = 1000
        lout = lrange[i]*kpc    #outer scale of turbulence
        kmin = 2*np.pi/(1e-1*lout)
        kmax = N * 2 * np.pi/d + kmin
        R = np.linspace(kmin, kmax, int(N/2)+1)
        Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(N/2)]
                 else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * np.pi)
                 for i in range(int(N/2)+1)]
        Bk1 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(N/2)-1-i]) for i in range(N)])
        Bx = np.real(np.fft.ifft(Bk1))
        Bx = Bx*np.sqrt(N)*Brange[j]*1e-10/(np.sqrt(np.sum(np.abs(Bx)**2)))
        Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(N/2)]
                 else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * 2 * np.pi)
                 for i in range(int(N/2)+1)]
        Bk2 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(N/2)-1-i]) for i in range(N)])
        By = np.real(np.fft.ifft(Bk2))
        By = By*np.sqrt(N)*Brange[j]*1e-10/(np.sqrt(np.sum(np.abs(By)**2)))
        Banx = interpolate.interp1d(np.linspace(0, d/kpc, N), Bx)
        Bany = interpolate.interp1d(np.linspace(0, d/kpc, N), By)
        
        rho0 = qp.basis(2, 0) * qp.basis(2, 0).dag()    #the initial density matrix
        
        "The operators to track the diagonal elements of the density matrix"
        op1 = qp.basis(2, 0) * qp.basis(2, 0).dag()
        op2 = qp.basis(2, 1) * qp.basis(2, 1).dag()
        
        H_dx = 0.5 * (-1j) * qp.Qobj([[ 0, 1],
                                      [ -1, 0]])*kpc/(hbar*c)
        
        H_dy = 0.5 * (-1j) * qp.Qobj([[ 0, 1j],
                                      [ 1j, 0]])*kpc/(hbar*c)
        
        dist = np.linspace(0, d/kpc, 100)
        
        options = qp.Options(nsteps=1E6)
        
        def B_varx (t, args):
            "Generate variable B-field"
            if t <= d/kpc:
                return Bcons + Banx(t)
            else:
                return 0
        
        def B_vary (t, args):
            "Generate variable B-field"
            if t <= d/kpc:
                return Bany(t)
            else:
                return 0
        
        def P_surv (mu):
            "Compute the survival probability"
            H_v = [[mu*H_dx, B_varx], [mu*H_dy, B_vary]]
            result = qp.mesolve(H_v, rho0, dist, e_ops=[op1, op2])
            return result.expect[0][-1]
        
        def spec (x, mean, alph):
            "Energy spectrum of SN neutrinos in MeV"
            return (x**alph * np.exp(-(alph+1)*x/mean)
                    * ((1+alph)/mean)**(1+alph)/special.gamma(1+alph))
        
        def frac (mean, alph, sigma):
            "Integrate the flux times the cross section"
            if mean == 0:
                return 0
            else:
                return integrate.quad(lambda x: spec(x, mean, alph) * sigma_extract(sigma, x),
                                      0, 100, args=(), limit=100)[0]
        
        mu_max = 4e-13
        Nmu = 20
        mus = np.linspace(0, mu_max, Nmu) * mu_b
        Ps = np.array([P_surv(mus[i]) for i in range(Nmu)])
        # mus = np.linspace(0, np.pi, 200)
        # Ps = np.array([1-np.sin(mus[i])**2 for i in range(200)])
        P_app = interpolate.interp1d(mus, Ps, 'cubic')
        
        binning     = np.linspace(-5e-3, 2e-2, 6)
        bin_centers = 0.5 * (binning[:-1] + binning[1:])
        
        idx_in = np.where(t>=-5e-3)[0][0]
        idx_fn = np.where(t<=2e-2)[0][-1]
        
        "Number of emitted neutrinos (Luminosity/(Mean Energy * Total Surface))"
        factor = 1e51 * erg/(4*np.pi*d**2*1e6*eV)    #Conversion factor from erg to Mev times 1/area
        
        N_e = [nu_e[1][i] * factor/nu_e[2][i] if nu_e[2][i] != 0 else 0 for i in range(idx_in, idx_fn+1)]
        N_ebar = [nubar_e[1][i] * factor/nubar_e[2][i] if nubar_e[2][i] != 0 else 0 for i in range(idx_in, idx_fn+1)]
        N_x = [nu_x[1][i] * factor/nu_x[2][i]  if nu_x[2][i] != 0 else 0 for i in range(idx_in, idx_fn+1)]
        
        def chi_square (mu, mh, dn=True, hk=True, return_rates=False):
            "Computes chi-square of dataset with magnetic moment mu assuming mu=0"
            
            if mh == "NH":
                "Flavor-mass state correspondence at high densities"
                i_e = 3
                i_mu = 1
                i_tau = 2
                i_be = 1
                i_bmu = 2
                i_btau = 3
            elif mh == "IH":
                i_e = 2
                i_mu = 1
                i_tau = 3
                i_be = 3
                i_bmu = 2
                i_btau = 1
            else:
                raise ValueError(f'\'{mh}\' is not a valid value for mh; supported values are \'NH\', \'IH\'')
            if dn:
                frac_e_dn = np.array([[[frac(nu_e[2][i], nu_e[3][i], sigma_DN[j][l]) for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)])
                frac_bare_dn = np.array([[[frac(nubar_e[2][i], nubar_e[3][i], sigma_bar_DN[j][l]) for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)])
                frac_x_dn = np.array([[[frac(nu_x[2][i], nu_x[3][i], sigma_DN[j][l]) for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)])
                frac_barx_dn = np.array([[[frac(nu_x[2][i], nu_x[3][i], sigma_bar_DN[j][l]) for i in range(idx_in, idx_fn+1)]
                              for l in range(2)] for j in range(2)])
                frac_emu_dn = frac_e_dn * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_e-1, i_e-1])
                frac_baremu_dn = frac_bare_dn * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_be-1, i_be-1])
                frac_mumu_dn = frac_x_dn * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_mu-1, i_mu-1])
                frac_barmumu_dn = frac_barx_dn * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_bmu-1, i_bmu-1])
                frac_taumu_dn = frac_x_dn * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_tau-1, i_tau-1])
                frac_bartaumu_dn = frac_barx_dn * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_btau-1, i_btau-1])
                
                "Number of neutrinos interacting with the detector for times t"
                dec_dn = [[((U[j,i_mu-1]**2+U[j,i_tau-1]**2) * (frac_x_dn[min(j,1)][0][i] * n_ar_DN + frac_x_dn[min(j,1)][1][i] * n_e_DN) * N_x[i]
                          + U[j,i_e-1]**2 * (frac_e_dn[min(j,1)][0][i] * n_ar_DN + frac_e_dn[min(j,1)][1][i] * n_e_DN) * N_e[i])
                        * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]
            
                dec_bar_dn = [[(U[j,i_be-1]**2 * (frac_bare_dn[min(j,1)][0][i] * n_ar_DN + frac_bare_dn[min(j,1)][1][i] * n_e_DN) * N_ebar[i]
                              + (U[j,i_bmu-1]**2 + U[j,i_btau-1]**2) * (frac_barx_dn[min(j,1)][0][i] * n_ar_DN + frac_barx_dn[min(j,1)][1][i] * n_e_DN) * N_x[i])
                            * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]
                
                "With magnetic conversion"
                dec_mu_dn = [[((U[j,i_mu-1]**2 * (frac_mumu_dn[min(j,1)][0][i] * n_ar_DN + frac_mumu_dn[min(j,1)][1][i] * n_e_DN)
                              + U[j,i_tau-1]**2 * (frac_taumu_dn[min(j,1)][0][i] * n_ar_DN + frac_taumu_dn[min(j,1)][1][i] * n_e_DN)) * N_x[i]
                            + U[j,i_e-1]**2 * (frac_emu_dn[min(j,1)][0][i] * n_ar_DN + frac_emu_dn[min(j,1)][1][i] * n_e_DN) * N_e[i])
                            * (t[i+1+idx_in]-t[i+idx_in])
                            for i in range(idx_fn+1-idx_in)] for j in range(3)]
                
                dec_bar_mu_dn = [[(U[j,i_be-1]**2 * (frac_baremu_dn[min(j,1)][0][i] * n_ar_DN + frac_baremu_dn[min(j,1)][1][i] * n_e_DN) * N_ebar[i]
                                + (U[j,i_bmu-1]**2 * (frac_barmumu_dn[min(j,1)][0][i] * n_ar_DN + frac_barmumu_dn[min(j,1)][1][i] * n_e_DN)
                                    + U[j,i_btau-1]**2 * (frac_bartaumu_dn[min(j,1)][0][i] * n_ar_DN
                                                          + frac_bartaumu_dn[min(j,1)][1][i] * n_e_DN)) * N_x[i])
                                * (t[i+1+idx_in]-t[i+idx_in])
                                for i in range(idx_fn+1-idx_in)] for j in range(3)]
                
                "Bins"
                bins_dn = np.array([[sum(dec_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                              :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                  for j in range(3)])
                
                bins_bar_dn = np.array([[sum(dec_bar_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                                      :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                      for j in range(3)])
                
                bins_mu_dn = np.array([[sum(dec_mu_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                                    :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                    for j in range(3)])
                
                bins_bar_mu_dn = np.array([[sum(dec_bar_mu_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                                            :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                        for j in range(3)])
                
               
                # print(bins_bar_dn)
                # print(bins_bar_mu_dn)
            else:
                bins_dn = np.zeros((3,6))
                bins_bar_dn = np.zeros((3,6))
                bins_mu_dn = np.zeros((3,6))
                bins_bar_mu_dn = np.zeros((3,6))
                
            if hk:
                frac_e_hk = np.array([[[frac(nu_e[2][i], nu_e[3][i], sigma_HK[j][l]) for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)])
                frac_bare_hk = np.array([[[frac(nubar_e[2][i], nubar_e[3][i], sigma_bar_HK[j][l]) for i in range(idx_in, idx_fn+1)] for l in range(3)] for j in range(2)])
                frac_x_hk = np.array([[[frac(nu_x[2][i], nu_x[3][i], sigma_HK[j][l]) for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)])
                frac_barx_hk = np.array([[[frac(nu_x[2][i], nu_x[3][i], sigma_bar_HK[j][l]) for i in range(idx_in, idx_fn+1)]
                              for l in range(3)] for j in range(2)])
                frac_emu_hk = frac_e_hk * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_e-1, i_e-1])
                frac_baremu_hk = frac_bare_hk * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_be-1, i_be-1])
                frac_mumu_hk = frac_x_hk * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_mu-1, i_mu-1])
                frac_barmumu_hk = frac_barx_hk * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_bmu-1, i_bmu-1])
                frac_taumu_hk = frac_x_hk * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_tau-1, i_tau-1])
                frac_bartaumu_hk = frac_barx_hk * P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_btau-1, i_btau-1])
                
                "Number of neutrinos interacting with the detector for times t"
                dec = [[((U[j,i_mu-1]**2+U[j,i_tau-1]**2) * (frac_x_hk[min(j,1)][0][i] * n_e_HK + frac_x_hk[min(j,1)][1][i] * n_O_HK) * N_x[i]
                          + U[j,i_e-1]**2 * (frac_e_hk[min(j,1)][0][i] * n_e_HK + frac_e_hk[min(j,1)][1][i] * n_O_HK) * N_e[i])
                        * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]
            
                dec_bar = [[(U[j,i_be-1]**2 * ((frac_bare_hk[min(j,1)][0][i] + frac_bare_hk[min(j,1)][2][i]) * n_e_HK
                                               + frac_bare_hk[min(j,1)][1][i] * n_O_HK) * N_ebar[i]
                              + (U[j,i_bmu-1]**2 + U[j,i_btau-1]**2) * ((frac_barx_hk[min(j,1)][0][i] + frac_barx_hk[min(j,1)][2][i]) * n_e_HK
                                                                        + frac_barx_hk[min(j,1)][1][i] * n_O_HK) * N_x[i])
                            * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]
                
                "With magnetic conversion"
                dec_mu = [[((U[j,i_mu-1]**2 * (frac_mumu_hk[min(j,1)][0][i] * n_e_HK + frac_mumu_hk[min(j,1)][1][i] * n_O_HK)
                              + U[j,i_tau-1]**2 * (frac_taumu_hk[min(j,1)][0][i] * n_e_HK + frac_taumu_hk[min(j,1)][1][i] * n_O_HK)) * N_x[i]
                            + U[j,i_e-1]**2 * (frac_emu_hk[min(j,1)][0][i] * n_e_HK + frac_emu_hk[min(j,1)][1][i] * n_O_HK) * N_e[i])
                            * (t[i+1+idx_in]-t[i+idx_in])
                            for i in range(idx_fn+1-idx_in)] for j in range(3)]
                
                dec_bar_mu = [[(U[j,i_be-1]**2 * ((frac_baremu_hk[min(j,1)][0][i] + frac_baremu_hk[min(j,1)][2][i]) * n_e_HK
                                                  + frac_baremu_hk[min(j,1)][1][i] * n_O_HK) * N_ebar[i]
                                + (U[j,i_bmu-1]**2 * ((frac_barmumu_hk[min(j,1)][0][i] + frac_barmumu_hk[min(j,1)][2][i]) * n_e_HK
                                                      + frac_barmumu_hk[min(j,1)][1][i] * n_O_HK)
                                    + U[j,i_btau-1]**2 * ((frac_bartaumu_hk[min(j,1)][0][i] + frac_bartaumu_hk[min(j,1)][2][i]) * n_e_HK
                                                          + frac_bartaumu_hk[min(j,1)][1][i] * n_O_HK)) * N_x[i])
                                * (t[i+1+idx_in]-t[i+idx_in])
                                for i in range(idx_fn+1-idx_in)] for j in range(3)]
                
                "Bins"
                bins_hk = np.array([[sum(dec[j][np.where(t>=binning[i])[0][0]-idx_in
                                              :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                  for j in range(3)])
                
                bins_bar_hk = np.array([[sum(dec_bar[j][np.where(t>=binning[i])[0][0]-idx_in
                                                      :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                      for j in range(3)])
                
                bins_mu_hk = np.array([[sum(dec_mu[j][np.where(t>=binning[i])[0][0]-idx_in
                                                    :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                    for j in range(3)])
                
                bins_bar_mu_hk = np.array([[sum(dec_bar_mu[j][np.where(t>=binning[i])[0][0]-idx_in
                                                            :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                        for j in range(3)])
                
                # fig, ax = plt.subplots()
                # fig.set_size_inches(3, 3)
            
                # ax.plot(binning*1e3, np.sum(bins_hk + bins_bar_hk, axis=0),
                #         ds='steps-post', label='$\\mu_\\nu = 0$', color='xkcd:blue')
                # ax.plot(binning*1e3, np.sum(bins_mu_hk + bins_bar_mu_hk, axis=0),
                #         ds='steps-post', label='$\\mu_\\nu = 10^{-13} \\mu_B$', ls='--',
                #         color='xkcd:red')
                
                # ax.plot(binning*1e3, np.sum(bins_hk, axis=0),
                #         ds='steps-post', label=f'$\\mu_\\nu = 0$', color='xkcd:blue')
                # ax.plot(binning*1e3, np.sum(bins_bar_hk, axis=0),
                #         ds='steps-post', label=f'$\\mu_\\nu = 0$', color='xkcd:red')
                # ax.set_xlim(-5, 20)
                # ax.set_ylim(0, 400)
                # ax.set_ylabel('Events per bin', fontsize=12)
                # ax.set_xlabel('t (ms)', fontsize=12)
                # ax.yaxis.set_ticks(np.arange(0, 401, 50))
                # ax.xaxis.set_ticks(np.arange(-5, 21, 5))
                # ax.tick_params(axis='both', which='both', labelsize=12)
                # ax.legend(fontsize=10)
                # plt.tight_layout()
                # plt.savefig('HK_1e-13mu_b_NO.pdf')
                
                # print(bins_bar)
                # print(bins_bar_mu)
                
            else:
                bins_hk = np.zeros((3,6))
                bins_bar_hk = np.zeros((3,6))
                bins_mu_hk = np.zeros((3,6))
                bins_bar_mu_hk = np.zeros((3,6))
                    
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

            if return_rates:
                rates = {}
                rates['DUNE','nu',   'nomu'] = bins_dn
                rates['DUNE','nubar','nomu'] = bins_bar_dn
                rates['DUNE','nu',   'mu']   = bins_mu_dn
                rates['DUNE','nubar','mu']   = bins_bar_mu_dn
                rates['HK','nu',   'nomu']   = bins_hk
                rates['HK','nubar','nomu']   = bins_bar_hk
                rates['HK','nu',   'mu']     = bins_mu_hk
                rates['HK','nubar','mu']     = bins_bar_mu_hk
                return chi, rates
            else:
                return chi

#        file_muN.write(str(optimize.root(lambda mu: chi_square(mu*1e-13*mu_b*[[0,0,0],[0,1,0],[0,0,0]], 'NH') - chi_thrs, 1).x[0]) + " \n")
#        file_muI.write(str(optimize.root(lambda mu: chi_square(mu*1e-13*mu_b*[[0,0,0],[0,1,0],[0,0,0]], 'IH') - chi_thrs, 1).x[0]) + " \n")
#        file_nusN.write(str(optimize.root(lambda mu: chi_square(mu*1e-13*mu_b*[[1,0,0],[0,1,0],[0,0,1]], 'NH') - chi_thrs, 0.25).x[0]) + " \n")
#        file_nusI.write(str(optimize.root(lambda mu: chi_square(mu*1e-13*mu_b*[[1,0,0],[0,1,0],[0,0,1]], 'IH') - chi_thrs, 0.25).x[0]) + " \n")

file_muN.close()
file_muI.close()
file_nusN.close()
file_nusI.close()

# plot event rates
def repeat_last(x):
    return x[list(range(len(x))) + [-1]]
mu = 1.
chi_NH, rates_NH = chi_square(mu*1e-13*mu_b*np.array([[0,0,0],[0,1,0],[0,0,0]]), 'NH', return_rates=True)
chi_IH, rates_IH = chi_square(mu*1e-13*mu_b*np.array([[0,0,0],[0,1,0],[0,0,0]]), 'IH', return_rates=True)
#chi_NH, rates_NH = chi_square(mu*1e-13*mu_b*np.array([[1,0,0],[0,1,0],[0,0,1]]), 'NH', return_rates=True)
#chi_IH, rates_IH = chi_square(mu*1e-13*mu_b*np.array([[1,0,0],[0,1,0],[0,0,1]]), 'IH', return_rates=True)
exp_labels = { 'DUNE': 'DUNE', 'HK': 'HyperK' }
for exp in ['DUNE','HK']:
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)

    hist_nomu_NH = np.sum(rates_NH[exp,'nu','nomu'] + rates_NH[exp,'nubar','nomu'], axis=0)
    hist_mu_NH   = np.sum(rates_NH[exp,'nu','mu']   + rates_NH[exp,'nubar','mu'], axis=0)
    hist_nomu_IH = np.sum(rates_IH[exp,'nu','nomu'] + rates_IH[exp,'nubar','nomu'], axis=0)
    hist_mu_IH   = np.sum(rates_IH[exp,'nu','mu']   + rates_IH[exp,'nubar','mu'], axis=0)
    plot_nomu_IH = ax.hist(bin_centers*1e3, bins=binning*1e3, weights=hist_nomu_IH,
            histtype='step', color='#cc0000', ls='--')[2][0]
    plot_mu_IH   = ax.hist(bin_centers*1e3, bins=binning*1e3, weights=hist_mu_IH,
            histtype='stepfilled', color='#ffaaaa', ec='#cc0000')[2][0]
    plot_err = ax.fill_between(binning*1e3, repeat_last(0.9*hist_nomu_IH),
                                            repeat_last(1.1*hist_nomu_IH),
                               step='post', color='#00000033', ec=None, zorder=2.5)
    ax.fill_between(binning*1e3, repeat_last(0.9*hist_mu_IH),
                                 repeat_last(1.1*hist_mu_IH),
                    step='post', color='#00000033', ec=None, zorder=2.5)

    plot_nomu_NH = ax.hist(bin_centers*1e3, bins=binning*1e3, weights=hist_nomu_NH,
                           histtype='step', color='#0000cc', ls='--')[2][0]
    plot_mu_NH   = ax.hist(bin_centers*1e3, bins=binning*1e3, weights=hist_mu_NH,
                           histtype='stepfilled', color='#aaaaff', ec='#0000cc')[2][0]
    ax.fill_between(binning*1e3, repeat_last(0.9*hist_nomu_NH),
                                 repeat_last(1.1*hist_nomu_NH),
                    step='post', color='#00000033', ec=None, zorder=2.5)
    ax.fill_between(binning*1e3, repeat_last(0.9*hist_mu_NH),
                                 repeat_last(1.1*hist_mu_NH),
                    step='post', color='#00000033', ec=None, zorder=2.5)

    legend_title_proxy = matplotlib.patches.Rectangle((0,0), 0, 0, color='w')
    leg = ax.legend([legend_title_proxy, plot_nomu_NH, plot_mu_NH,
                     legend_title_proxy, plot_nomu_IH, plot_mu_IH, plot_err],
                    ['\\bf normal ordering', '$\\mu_\\nu = 0$', '$\\mu_\\nu = 10^{-13} \\mu_B$',
                    '\\bf inverted ordering','$\\mu_\\nu = 0$', '$\\mu_\\nu = 10^{-13} \\mu_B$',
                    '10\% flux error'],
                    fontsize=10)
    for item, label in zip(leg.legendHandles, leg.texts): # move legend titles to the left
        item.set_x(10)
        label.set_position((10,0))
        if re.match('.*ordering', label._text):
#            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-20,0))
        if re.match('10.*', label._text):
            item.set_y(-5)
            label.set_position((10,-5))
    leg._legend_box.set_width(100.)
    leg._legend_box.set_height(101.)

    ax.set_xlim(-5, 20)
    ax.set_ylim(0, 40)
    ax.set_ylabel('events / 5\,ms', fontsize=12)
    ax.set_xlabel('time after core bounce [ms]', fontsize=12)
    ax.yaxis.set_ticks(np.arange(0, 36, 5))
    ax.xaxis.set_ticks(np.arange(-5, 21, 5))
    ax.tick_params(axis='both', which='both', labelsize=12)
    plt.tight_layout()
    plt.annotate(exp_labels[exp], (0.96,0.05), xycoords='axes fraction',
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='square', fc='#ffffffdd', ec='black'))
    plt.annotate('$8.8 M_\odot$, 10\,kpc',
                 (0.03,0.97), xycoords='axes fraction', ha='left', va='top')
    plt.savefig('{:s}_1e-13mu_b_NO.pdf'.format(exp))
 
# Nmu = 50
# murange = np.linspace(0, mu_max, Nmu) * mu_b
# chis_mu_NO = np.array([chi_square(murange[i]*np.array([[0,0,0],[0,1,0],[0,0,0]]), 'NH') for i in range(Nmu)])
# chis_mu_IO = np.array([chi_square(murange[i]*np.array([[0,0,0],[0,1,0],[0,0,0]]), 'IH') for i in range(Nmu)])
# chis_nus_NO = np.array([chi_square(murange[i]*np.array([[1,0,0],[0,1,0],[0,0,1]]), 'NH') for i in range(Nmu)])
# chis_nus_IO = np.array([chi_square(murange[i]*np.array([[1,0,0],[0,1,0],[0,0,1]]), 'IH') for i in range(Nmu)])

# fig, ax = plt.subplots()
# ax.set_ylim(0, 10)
# ax.set_xlim(0, mu_max)
# ax.plot(murange/mu_b, chis_mu_NO)

# np.savetxt("chi_mu_NO.txt", np.array([murange, chis_mu_NO]))
# np.savetxt("chi_mu_IO.txt",np.array([murange, chis_mu_IO]))
# np.savetxt("chi_nus_NO.txt",np.array([murange, chis_nus_NO]))
# np.savetxt("chi_nus_IO.txt",np.array([murange, chis_nus_IO]))

# xs = np.linspace(0, mu_max*mu_b, 50)
# chis1 = np.array([chi_square(xs[i]*np.array([[1,0,0],[0,1,0],[0,0,1]]), "NH") for i in range(50)])
# chis2 = np.array([chi_square(xs[i]*np.array([[1,0,0],[0,1,0],[0,0,1]]), "IH") for i in range(50)])
# fig, ax = plt.subplots()
# ax.set_xlim(0, mu_max)
# ax.set_ylim(0, 10)
# ax.hlines(3.84, 0, mu_max, ls='--', color='gray')
# ax.hlines(6.63, 0, mu_max, ls='--', color='gray')
# ax.set_ylabel(r'$\chi^2$')
# ax.set_xlabel(r'$\mu_\nu (\mu_B)$')
# s = (r'Combined analysis of DUNE and Hyper-K setting $\\mu_e = \\mu_\\mu = \\mu_\\tau = \\mu_\\nu$')
# ax.text(-0.25e-13, -2, s)
# ax.plot(xs/mu_b, chis1, label='NH')
# ax.plot(xs/mu_b, chis2, label='IH')
# ax.legend()
# plt.tight_layout()
# plt.savefig('combined_nus.pdf')
