"A program to numerically determine the flavor ratios of neutrinos with a magnetic moment"
import ternary
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy import integrate
from moment import *

"Neutrino mixing parameters"
theta12 = 0.5836381018669038    #mixing angle
theta13 = 0.14957471689591403
theta23 = 0.8552113334772213
m21 = 7.59e-5    #squared mass difference in eV**2
m31 = 2.32e-3

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rc('text', usetex=True)

"Ice-Cube limits on flavor ratios (taken from 1507.03991)"
contour1 = np.concatenate((np.flip(np.loadtxt('contour.csv'), axis=0), np.loadtxt('contour2.csv')[1:]), axis=0)
contour2 = np.loadtxt('contour3.csv')
contour3 = np.loadtxt('contour4.csv')

"Neutrino masses in normal and inverted hierarchies, resp."
m_nh = [0, np.sqrt(m21), np.sqrt(m31)]
m_ih = [np.sqrt(m31), np.sqrt(m31+m21), 0]

N = 101    #Number of plot points for magnetic moment

B = B_gal    #B-field
d = 10*kpc    #Distance travelled in galactic B-field
T = d/c    #Time for neutrinos to reach the Earth

U = PMNS(theta12, theta13, theta23)    #PMNS matrix

mu = [np.pi*2*hbar/(2*T*B_gal*mu_b*U[1,i]) for i in range(3)]    #maximal magnetic moment to be considered
m_idx = 1    #left-handed mass with which the sterile mass coincides
mus = np.linspace(0, np.abs(mu[m_idx]), N) * mu_b

GeV = 1e9
TeV = 1e12
PeV = 1e15

"Parameters for the initial flux of astrophysical neutrinos (taken from 1507.03991)"
phi_0 = 6.7
gamma = 2.5
E_min = 25 * TeV
E_max = 2.8 * PeV

h = np.sqrt(3/4)    #height of ternary plot

def spec (x, mu, mN, mnu):
    "Energy spectrum of SN neutrinos with magnetic moment mu in eV after conversion"
    return (phi_0 * (x/(100*TeV))**(-gamma)
            * ((mN**2-mnu**2)**2 + 4 * x**2 * mu**2 * B**2
               * np.cos(np.sqrt(((mN**2-mnu**2))**2+4*x**2*mu**2*B**2)*T/(4*hbar*x))**2)
            /((mN**2-mnu**2)**2 + 4 * x**2 * mu**2 * B**2))

def frac(mu, mN, mnu):
    "Integrated flux"
    return (integrate.quad(lambda x: spec(x, mu, mN, mnu), E_min, E_max, args=())[0]
            *((-gamma+1) * (100*TeV)**(-gamma))/(phi_0 * (E_max**(-gamma+1) - E_min**(-gamma+1)))
            if (mu != 0 and np.abs(mN**2-mnu**2) < np.abs(1e-3*4*E_max*mu*B))
            else 1)

def ratios(f_in, mu, mN, m):
    "Returns the final flavor ratios"
    f_temp = [sum([f_in[i] * U[i, j]**2 for i in range(3)]) * frac(U[1, j]*mu, mN, m[j]) for j in range(3)]
    f = np.array([sum([f_temp[j] * U[k, j]**2 for j in range(3)]) for k in range(3)])
    return f/np.sum(f)

def three_to_two (f):
    "Converts coordinates from ternary plot into cartesian ones (three coordinates to two)"
    return [[(f[i][1] + 2*f[i][0])/2, f[i][1]*h] for i in range(len(f))]

fs = [[0, 1, 0], [1, 2, 0]]    #Ratios at the source

points_b = np.array([[ratios(fs[i], mus[j], m_nh[m_idx], m_nh) for j in range(N)] for i in range(2)])
points_v = np.array([ratios(fs[i], 0, m_nh[m_idx], m_nh) for i in range(2)])

"Color maps"
blues = plt.get_cmap('Blues')
greens = plt.get_cmap('Greens')
grey = plt.get_cmap('Greys')

cmbl = ListedColormap(blues(np.linspace(1.0, 0.25, 256)))
cmgr = ListedColormap(greens(np.linspace(1.0, 0.25, 256)))

fig, tax = ternary.figure()

fig.set_size_inches(4,2.5)

ax = tax.get_axes()

# tax.set_title("Flavor Ratios")
tax.gridlines(multiple=0.1, color='grey')
tax.boundary()
tax.set_background_color('white')
tax.bottom_axis_label("$\\nu_e$", offset=0.03, fontsize=12)
tax.right_axis_label("$\\nu_{\\mu}$", offset=0.1, fontsize=12)
tax.left_axis_label("$\\nu_{\\tau}$", offset=0.16, fontsize=12)
tax.plot(contour1, color=grey(0.8))
tax.plot(contour2, color=grey(0.8))
tax.plot(contour3, color=grey(0.8))
tax.scatter([points_v[0]], marker='s', color=cmgr(0), label='$(0:1:0)$')
tax.scatter([points_v[1]], marker='o', color=cmbl(0), label='$(1:2:0)$')
# for i in range(N-1):
#     tax.plot(points_b[0][i:i+2], color=cmgr(1-i/(2*N-2)), linewidth=2.0)
#     tax.plot(points_b[1][i:i+2], color=cmbl(1-i/(2*N-2)), linewidth=2.0)
    # tax.scatter([points_b[0][i]], color=cmgr(1-i/(2*N-2)))
    # tax.scatter([points_b[1][i]], color=cmbl(1-i/(2*N-2)))
line_seg_1 = LineCollection([three_to_two(points_b[0][i:i+2]) for i in range(N-1)],
                            cmap=cmgr, linewidths=2)
line_seg_2 = LineCollection([three_to_two(points_b[1][i:i+2]) for i in range(N-1)],
                            cmap=cmbl, linewidths=2)
line_seg_1.set_array(mus/mu_b)
line_seg_2.set_array(mus/mu_b)
ax.add_collection(line_seg_1)
ax.add_collection(line_seg_2)
ax.axis('off')
ax.set_aspect(1)
ax.text(0.54, 0.52, "68\%", rotation=20)
ax.text(0.50, 0.59, "90\%", rotation=20)
# ax.set_ylim(0, 1)
# ax.set_xlim(0, 1)
tax.ticks(axis='lbr', linewidth=0.5, multiple=0.2, tick_formats="%.1f", offset=0.025)

cbar = fig.colorbar(line_seg_1)
cbar.set_label('Neutrino magnetic moment [$\\mu_B$]', size=12)
fig.colorbar(line_seg_2)

# s = ('Flavor ratios for transition magnetic moment. We set all $\\mu_{xy}=$' f'{mu[1, 0]/mu_b}'
#       f'$\\mu_B$, $E_\\nu =$ 1PeV and consider a magnetic field of strength {B_gal*1e10} $\\mu G$.')
# tax.get_axes().text( 0.5, -0.1, s, horizontalalignment='center', verticalalignment='center',
#                     transform=tax.get_axes().transAxes)

tax.legend(frameon=False, loc=(0.6,0.8))
plt.tight_layout()
