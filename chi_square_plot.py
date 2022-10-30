"A simulation of the detection of supernova neutrinos assuming coherent conversion on a turbulent magnetic field"
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from moment import *

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size':16})
matplotlib.rc('text', usetex=True)

chis_mu_NO = np.loadtxt("chi_mu_NO.txt")
chis_mu_IO = np.loadtxt("chi_mu_IO.txt")
chis_nus_NO = np.loadtxt("chi_nus_NO.txt")
chis_nus_IO = np.loadtxt("chi_nus_IO.txt")

fig, ax = plt.subplots(figsize=(((1+np.sqrt(5))/2 * 4), 4))

ax.hlines(2.71, 0, 2e-13, color='k', ls='--', lw=1)
ax.hlines(3.84, 0, 2e-13, color='k', ls='--', lw=1)
ax.hlines(6.63, 0, 2e-13, color='k', ls='--', lw=1)
ax.plot(chis_mu_NO[0]/mu_b, chis_mu_NO[1], label=r'$\nu_\mu$ only', color='b')
ax.plot(chis_nus_NO[0]/mu_b, chis_nus_NO[1], label=r'All $\nu$''\'s', color='r')
ax.set_ylim(0, 10)
ax.set_xlim(chis_nus_NO[0,0]/mu_b, chis_nus_NO[0,-1]/mu_b)
ax.set_ylabel(r'$\chi^2$')
ax.set_xlabel(r'$\mu_\nu (\mu_B)$')
ax.text(0.01e-13, 2.8, r'$90 \%$ C.L.', size=10)
ax.text(0.01e-13, 3.9, r'$95 \%$ C.L.', size=10)
ax.text(0.01e-13, 6.7, r'$99 \%$ C.L.', size=10)
ax.legend(frameon=False, title='NO', loc=9, fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.savefig("chi-square_NO.pdf")