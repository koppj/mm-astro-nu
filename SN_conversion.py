"Compute the evolution of neutrinos inside a supernova with magnetic moment"
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import qutip as qp
from scipy import special, integrate, optimize
from moment import *


"Emission data from simulation. Col 0: time in s, col 1:luminosity in 1e51 ergs,"
"col 2: average energy in MeV, col 3: alpha parameter"
nu_e = np.loadtxt('sn-data/Sf/neutrino_signal_nu_e', usecols=(0,1,2,5)).transpose()
nubar_e = np.loadtxt('sn-data/Sf/neutrino_signal_nubar_e', usecols=(0,1,2,5)).transpose()
nu_x = np.loadtxt('sn-data/Sf/neutrino_signal_nu_x', usecols=(0,1,2,5)).transpose()

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rc('text', usetex=True)

#mixing angles
theta12 = 0.5836381018669038
theta13 = 0.14957471689591403
theta23 = 0.8552113334772213

#magnetic moment matrix
mu = np.array([[0, 1e-13*mu_b, 1e-13*mu_b],
              [-1e-13*mu_b, 0, 0*1e-13*mu_b],
              [-1e-13*mu_b, -0*1e-13*mu_b, 0]])

Ye = 0.5
Ve = 7.6    #Matter potential coefficient in eV

B0 = 5e6    #Magnetic field in T
R = 696340    #Radius of SN in km
r0 = 0.0024 * R

def V(x, args):
    "Matter potential in eV for x in km"
    return Ve * (x+1)**(-2.4)

def B(x, args):
    "Magnetic field profile"
    return B0 * (r0/(x+1))**3

MeV = 1e6

psi0 = qp.Qobj(np.dot(PMNS(theta12,theta13,theta23), np.array([1,0,0,0,0,0])))
# psi0 = qp.Qobj(np.array([1,0,0,0,0,0]))

E = 1e6    #neutrino energy in eV

# H0 = qp.Qobj(np.diag([0, 7.59e-5/(2*E), 2.32e-3/(2*E), 0, 7.59e-5/(2*E), 2.32e-3/(2*E)]))/(hbar*c)
H0 = qp.Qobj(np.diag([2.32e-3/(2*E), (2.32e-3+7.59e-5)/(2*E), 0, 2.32e-3/(2*E), (2.32e-3+7.59e-5)/(2*E), 0]))/(hbar*c)
Hm = qp.Qobj(np.linalg.multi_dot([PMNS(theta12,theta13,theta23), np.diag([(3*Ye-1)/2, (Ye-1)/2, (Ye-1)/2, 0, 0, 0]),
                                  np.transpose(PMNS(theta12,theta13,theta23))]))/(hbar*c)
# Hm = qp.Qobj(np.linalg.multi_dot([PMNS(theta12,theta13,theta23), np.diag([(3*Ye-1)/2, (Ye-1)/2, (Ye-1)/2,
#                                                                           -(3*Ye-1)/2, -(Ye-1)/2, -(Ye-1)/2]),
#                                         np.transpose(PMNS(theta12,theta13,theta23))]))/(hbar*c)
Hu = [sum([mu[i,j] * Hx(i, j) for i in range(3) for j in range(3)]), B]
Hb = sum([mu[i,j] * Hx(i, j) for i in range(3) for j in range(3)])

ts = np.logspace(3, 5, 101)
# ts = [1e4, 1e6]
# ts = np.linspace(1000, 1100, 1001)

result = qp.sesolve([H0, [Hm, V], [Hb, B]], psi0, ts, [rho11, rho22, rho33, rho44, rho55, rho66])

fig, ax = plt.subplots()

ax.plot(ts, result.expect[0], label=r'$\nu_L^1$')
ax.plot(ts, result.expect[1], label=r'$\nu_L^2$')
ax.plot(ts, result.expect[2], label=r'$\nu_L^3$')
ax.plot(ts, result.expect[3], label=r'$N_R^1$')
ax.plot(ts, result.expect[4], label=r'$N_R^2$')
ax.plot(ts, result.expect[5], label=r'$N_R^3$')

ax.set_title(r'Initial $\nu_e$, $B_0 = 10^{10}$ G, $\mu_\nu = 5 \times 10^{-13} \mu_B$, IO, Dirac, $E_\nu=1$ MeV')
ax.set_ylabel('Component')
ax.set_xlabel('L (in km)')
ax.set_xscale('log')
ax.set_ylim(0, 1)
ax.set_xlim(1e3, 1e5)
ax.legend()

plt.tight_layout()
plt.savefig("nu_e_conversion_iO")