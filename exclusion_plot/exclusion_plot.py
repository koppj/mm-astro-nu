import scipy, os, subprocess, shutil # scipy for integrating, the rest are for calling programs outside of python, changing directories ect
import numpy as np                   # used for dealing with arrays
import matplotlib.pyplot as plt      # plotting package
from Header import *                 # this contains all my ploting options for matplotlib
import pandas as pd                  # reading in all data using pandas to make life easy
import pickle                        # for importing and saving the interpolated muon width grid
# set the default printing option so I don't
# have to count the number of digits all the time
# Unit conversion and physical constants

import direct_det_neutrinos as dd
u = dd.my_units()

default_colours = ['black',u'#E24A33', u'#348ABD', u'#988ED5', u'#777777', u'#FBC15E', u'#8EBA42', u'#FFB5B8', u'#CB04A5']
lighter_default_colours =['#666666','#ed9284','#80badc','#c1bbe5', '#adadad', '#fcd99e', '#bbd68c', '#ffd2d4', '#fb4cda']

stellar_cooling_data = pd.read_csv('./data/stellar_cooling.csv', delimiter=',', names=['mN','d'])

# Constraints from 1707.08573 for nu_mu
charm_nomad_numu = pd.read_csv('data/MM_numu_constraints.csv', delimiter=',', names=['mN','d'])
charm_nomad_numu['mN'] *= u.GeV/u.MeV
IC_numu = pd.read_csv('data/IC_DC_1707.08573.csv', delimiter=',', names=['mN','d'])
IC_numu['mN'] *= u.GeV/u.MeV
miniboone_numu = pd.read_csv('data/miniboone.csv', delimiter=',', names=['mN','d'])
miniboone_numu['d'] *= 1/u.GeV / u.muB

miniboone_numu_roi = pd.read_csv('data/miniboone-roi.csv', delimiter=',', names=['mN','d'])
miniboone_numu_roi['d'] *= 1/u.GeV / u.muB

miniboone_lsnd_numu_roi = pd.read_csv('data/miniboone-lsnd-roi.csv', delimiter=',', names=['mN','d'])
miniboone_lsnd_numu_roi['d'] *= 1/u.GeV / u.muB

# Vedrans nuclear recoil exclusion curve
NR_CLs90 = pd.read_csv('data/Xenon1T/nuclear_recoil_bound_numu.dat', delimiter='\t', names=['mN','d'])

mN_vals = np.logspace(-2, np.log10(3), 50)
d_vals = np.logspace(-11, -7, 50)
XXp, YYp = np.meshgrid(mN_vals, d_vals)
ZZp = np.load('./data/Xenon1T_final_contours.npy')

mN_vals = np.logspace(-2, np.log10(3), 30)
d_vals = np.logspace(-11, -6, 30)
XXp_lim, YYp_lim = np.meshgrid(mN_vals, d_vals)
ZZp_lim = np.load('./data/Xenon1T_final_contours_v2.npy')

#electron recoil contour
ER_CLs90 = np.array([5.69017100e-11, 5.69017100e-11, 5.67095373e-11, 5.64592207e-11, 5.69240043e-11,
        5.67275883e-11, 5.68532948e-11, 5.70610913e-11, 5.73718028e-11,
        5.80001086e-11, 5.91329145e-11, 6.11065424e-11, 6.44584909e-11,
        6.58068912e-11, 6.71854273e-11, 6.96055345e-11, 6.92002897e-11,
        7.88970766e-11, 9.78780184e-11, 1.45579077e-10, 1.50340341e-10,
        2.16527347e-10, 2.24030861e-10, 2.48632172e-10, 3.22052406e-10,
        4.79005324e-10, 5.78047904e-10, 7.12449578e-10, 1.05966338e-09,
        1.16380492e-09, 1.57609256e-09, 2.34420460e-09, 3.48665764e-09,
        5.18588759e-09, 5.33943525e-09, 5.65592672e-09, 6.43191121e-09,
        7.71324084e-09, 9.07766761e-09, 1.14723050e-08, 1.70633570e-08,
        2.33561738e-08, 2.53792201e-08, 3.77478364e-08, 4.34632083e-08,
        5.61443241e-08, 8.35063787e-08, 9.52908404e-08, 1.24203388e-07,
        1.84734171e-07, 2.74764759e-07, 2.81614329e-07, 4.08671944e-07,
        6.07839076e-07, 9.04070731e-07, 1.34467151e-06, 2.00000000e-06, 1e8])

mN_xenon1T = np.array([0, 1.00000000e-02, 1.21735705e-02, 1.48195818e-02, 1.80407224e-02,
        2.19620005e-02, 2.67355961e-02, 3.25467663e-02, 3.96210354e-02,
        4.82329466e-02, 5.87167175e-02, 7.14792098e-02, 8.70157199e-02,
        9.17917066e-02, 1.05929200e-01, 1.28953658e-01, 1.56982644e-01,
        1.91103928e-01, 2.10127377e-01, 2.31268090e-01, 2.32641714e-01,
        2.80044750e-01, 2.83208030e-01, 3.44765291e-01, 3.79105022e-01,
        4.10482304e-01, 4.19702457e-01, 4.57854953e-01, 5.03394686e-01,
        5.10927744e-01, 5.82429678e-01, 6.13561911e-01, 6.20192858e-01,
        6.21861045e-01, 6.21981490e-01, 7.57173550e-01, 9.21750557e-01,
        1.03288739e+00, 1.12209954e+00, 1.21818997e+00, 1.32264626e+00,
        1.36599578e+00, 1.40751343e+00, 1.60576178e+00, 1.66290459e+00,
        1.88225513e+00, 2.00592313e+00, 2.02434862e+00, 2.30114730e+00,
        2.43414676e+00, 2.46241972e+00, 2.46435506e+00, 2.97113834e+00,
        2.99583182e+00, 2.99921630e+00, 2.99983293e+00, 2.99993197e+00, 2.99993197e+00])

# current best contour
borexino_CL90 = np.array([3.47944489e-11,3.47944489e-11, 3.47950219e-11, 3.47971900e-11, 3.47989717e-11,
       3.47457262e-11, 3.47482309e-11, 3.47494324e-11, 3.47495250e-11,
       3.47505609e-11, 3.47564272e-11, 3.47569931e-11, 3.47166860e-11,
       3.47230635e-11, 3.47293785e-11, 3.46932907e-11, 3.46596281e-11,
       3.46756784e-11, 3.46138528e-11, 3.46031532e-11, 3.45718126e-11,
       3.45619625e-11, 3.45587902e-11, 3.46039794e-11, 3.46931677e-11,
       3.48761514e-11, 3.51757848e-11, 3.56084338e-11, 3.61239013e-11,
       3.66172101e-11, 3.65926433e-11, 3.51352309e-11, 3.24382551e-11,
       3.22203623e-11, 3.87592897e-11, 1.75711615e-10, 3.36231251e-10,
       2.23170261e-10, 1.88471123e-10, 7.62235727e-10, 8.09327374e-10,
       8.31003741e-10, 8.60040382e-10, 8.99729117e-10, 9.55556535e-10,
       1.03732532e-09, 1.16437826e-09, 1.38027565e-09, 1.80332545e-09,
       2.88082779e-09, 7.94609163e-09, 8.75839636e-08, 1e8])

mN_borexino = np.insert(np.logspace(-2,np.log10(10), 60)[0:52] * u.MeV, 0, 0)

#supernova neutrinos
sn_min = np.loadtxt('./data/contour_min_27_08.txt')
sn_avg = np.loadtxt('./data/contour_combined_26_08.txt')
sn_max = np.loadtxt('./data/contour_max_31_08.txt')

sn_1kpc_min = np.loadtxt('./data/contours_1kpc_min_1_09.txt')
sn_1kpc_avg = np.loadtxt('./data/contours_1kpc_31_08.txt')
sn_1kpc_max = np.loadtxt('./data/contours_1kpc_max_1_09.txt')

fig, ax = plt.subplots(figsize=np.array([(1+np.sqrt(5))/2,1])*5)

ax.grid(True, lw=0.25, ls='--', alpha=0.4)
ax.set_xlim(1E-18, 1E+3)
ax.set_ylim(1E-14, 1E7)

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_xlabel(r'Right-handed neutrino mass $M_N$ [MeV]')
ax.set_ylabel(r'Neutrino magnetic moment [$\mu_B$]')

# Borexino limit
ax.plot(mN_borexino/u.MeV, 2.*borexino_CL90, lw = 0.75, c = default_colours[0])

# Xenon 1T NR limit
ax.plot(NR_CLs90['mN'],2.*NR_CLs90['d'], c = default_colours[0])

ax.plot(mN_xenon1T,ER_CLs90, c = default_colours[0], ls='--')

# ultimate_bf = 115.85803425043466
# ax.contour(XXp_lim, 2.*YYp_lim, ZZp_lim, levels = ultimate_bf + np.array([scipy.stats.chi2.ppf(0.9,2)]), 
#            colors = default_colours[0], linestyles='--', zorder = 3)

# Terrestrial experiments
ax.fill_between(charm_nomad_numu['mN'][1:],charm_nomad_numu['d'][1:],1e8, color = '#e3e3e3', lw=0.)
ax.fill_between(miniboone_numu['mN'],miniboone_numu['d'],1e8, color = '#e3e3e3', lw=0.)

ax.fill_between(IC_numu['mN'][60:88],IC_numu['d'][60:88], 1e8, color = '#e3e3e3', ls='--', zorder = 0)
# ax.plot(miniboone_numu_roi['mN'],miniboone_numu_roi['d'], color = default_colours[6], ls='-')

#Stellar cooling constraints
ax.plot(stellar_cooling_data['mN'],stellar_cooling_data['d'], c = default_colours[3] )
ax.fill_between(stellar_cooling_data['mN'],stellar_cooling_data['d'],1e7, color = lighter_default_colours[3], lw = 0.1 )

#SN magnetic moment constraints at 10 kpc
ax.plot(sn_avg[0], sn_avg[1], c = 'xkcd:blue')
ax.fill_between(sn_avg[0], sn_min[1], sn_max[1], color = 'xkcd:blue', alpha=0.25)

#SN magnetic moment constraints at 10 kpc
ax.plot(sn_1kpc_avg[0], sn_1kpc_avg[1], c = 'xkcd:orange')
ax.fill_between(sn_1kpc_avg[0], sn_1kpc_min[1], sn_1kpc_max[1], color = 'xkcd:orange', alpha=0.25)

#Annotations

ax.text(0.75, 0.52, r'Stellar', transform=ax.transAxes, fontsize=12, c = default_colours[3])
ax.text(0.75, 0.485, r'cooling', transform=ax.transAxes, fontsize=12, c = default_colours[3])

ax.text(0.65, 0.19, r'\textsc{Borexino}', transform=ax.transAxes, fontsize=10)

ax.text(0.12, 0.85, 'Hyper-K+DUNE (10 kpc, $B=1$-$10 \\mu$G)', transform=ax.transAxes, fontsize=10, c='xkcd:blue')

ax.text(0.21, 0.3, 'Hyper-K+DUNE (1 kpc, $B=1$-$10 \\mu$G)', transform=ax.transAxes, fontsize=10, c='xkcd:orange', rotation=51)

ax.text(0.77, 0.39, r'\textsc{Xenon1T}', transform=ax.transAxes, fontsize=10, c = default_colours[0])
ax.text(0.75, 0.36, r'electron recoil', transform=ax.transAxes, fontsize=10, c = default_colours[0])
ax.text(0.775, 0.335, r'(incl. tritium)', transform=ax.transAxes, fontsize=5, c = default_colours[0])

plt.arrow(0.858, 0.35, 0.004, -0.015,head_width = 0.01, transform=ax.transAxes, zorder = 5, color = default_colours[0])

ax.text(0.65, 0.33, r'\textsc{Xenon1T}', transform=ax.transAxes, fontsize=10)
ax.text(0.635, 0.30, r'nuclear recoil', transform=ax.transAxes, fontsize=10)

ax.text(0.87, 0.2, r'\textsc{MiniBooNE}', transform=ax.transAxes,
        fontsize=8, c = default_colours[4], rotation = 0)
ax.text(0.91, 0.17, r'\textsc{IceCube}', transform=ax.transAxes,
        fontsize=8, c = default_colours[4], rotation = 0)
plt.arrow(0.98, 0.19, 0.003, 0.015,head_width = 0.01, transform=ax.transAxes, zorder = 5, color = default_colours[4])
ax.text(0.82, 0.24, r'\textsc{Charm-II}', transform=ax.transAxes,
        fontsize=8, c = default_colours[4], rotation = 0)
ax.text(0.94, 0.275, r'\textsc{Nomad}', transform=ax.transAxes, 
        fontsize=8, c = default_colours[4], rotation = -17)
ax.text(0.005, 0.01, r'All limits at 90\% CL', transform=ax.transAxes, fontsize=8, c = default_colours[0])

plt.tight_layout()
plt.savefig('exclusion_plot.pdf')