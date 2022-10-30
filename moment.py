"""Module containing definitions for applications involving neutrino oscillations and magnetic moment"""
import qutip as qp
import numpy as np

hbar = 6.582E-16    #h-bar in eV*s
mu_b = 5.788E-5    #Bohr magneton in ev/T
B_gal = 4.2E-10    #Mean galactic B-field in T
c = 2.998E8    #Speed of light in m/s
kpc = 30.9E18    #Kiloparsec in meters
eV = 1.602E-19    #eV in J
erg = 1e-7    #erg in J
me = 5.11E5    #electron mass in eV
al = 0.007297352    #Fine structure constant
epsilon0 = 5.526349406e7    #Vacuum permittivity in e^2/(eV m)
mol = 6.022e23

"The operators to determine the diagonal elements of the density matrix"
rho11 = qp.basis(6, 0) * qp.basis(6, 0).dag()
rho22 = qp.basis(6, 1) * qp.basis(6, 1).dag()
rho33 = qp.basis(6, 2) * qp.basis(6, 2).dag()
rho44 = qp.basis(6, 3) * qp.basis(6, 3).dag()
rho55 = qp.basis(6, 4) * qp.basis(6, 4).dag()
rho66 = qp.basis(6, 5) * qp.basis(6, 5).dag()

def PMNS (theta12, theta13, theta23):
    "Computes the PMNS matrix for given mixing angles"
    rot23 = np.array([[1, 0, 0, 0, 0, 0], [0, np.cos(theta23), np.sin(theta23), 0, 0, 0],
                      [0, -np.sin(theta23), np.cos(theta23), 0, 0, 0], [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, np.cos(theta23), np.sin(theta23)], [0, 0, 0, 0, -np.sin(theta23), np.cos(theta23)]])
    
    rot13 = np.array([[np.cos(theta13), 0, np.sin(theta13), 0, 0, 0], [0, 1, 0, 0, 0, 0],
                      [-np.sin(theta13), 0, np.cos(theta13), 0, 0, 0], [0, 0, 0, np.cos(theta13), 0, np.sin(theta13)],
                      [0, 0, 0, 0, 1, 0], [0, 0, 0, -np.sin(theta13), 0, np.cos(theta13)]])
    
    rot12 = np.array([[np.cos(theta12), np.sin(theta12), 0, 0, 0, 0], [-np.sin(theta12), np.cos(theta12), 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0], [0, 0, 0, np.cos(theta12), np.sin(theta12), 0],
                      [0, 0, 0, -np.sin(theta12), np.cos(theta12), 0], [0, 0, 0, 0, 0, 1]])
    
    return np.linalg.multi_dot([rot23, rot13, rot12])

def Hx(nu1, nu2):
    """Generates magnetic moment hamiltonian between flavors nu1 and nu2 for B-field
       in x direction"""
    return 1/(hbar*c) * 0.5 * 1j * (qp.basis(6, nu1) * qp.basis(6, nu2+3).dag()
                                       - qp.basis(6, nu2+3) * qp.basis(6, nu1).dag())

def Hy(nu1, nu2):
    """Generates magnetic moment hamiltonian between flavors nu1 and nu2 for B-field
       in y direction"""
    return 1/(hbar*c) * 0.5 * (qp.basis(6, nu1) * qp.basis(6, nu2+3).dag()
                               + qp.basis(6, nu2+3) * qp.basis(6, nu1).dag())
