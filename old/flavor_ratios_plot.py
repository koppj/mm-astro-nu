"A program to numerically determine the flavor ratios of neutrinos with a magnetic moment"
import ternary
import matplotlib
import alphashape
import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from scipy import integrate, interpolate
from moment import *

"Neutrino mixing parameters"
theta12 = 0.5836381018669038    #mixing angle
theta13 = 0.14957471689591403
theta23 = 0.8552113334772213
m21 = 7.59e-5    #squared mass difference in eV**2
m31 = 2.32e-3

GeV = 1e9
TeV = 1e12
PeV = 1e15

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
matplotlib.rc('text', usetex=True)

"IceCube limits on flavor ratios (from 1507.03991)"
contour1 = np.concatenate((np.flip(np.loadtxt('contour.csv'), axis=0), np.loadtxt('contour2.csv')[1:]), axis=0)
contour2 = np.loadtxt('contour3.csv')
contour3 = np.loadtxt('contour4.csv')

"IceCube projected limits (from 2008.04323)"
contour8 = np.roll(np.loadtxt('8.csv'), 2, axis=1)
contour15 = np.roll(np.loadtxt('15.csv'), 2, axis=1)
contour1510 = np.roll(np.loadtxt('15+10.csv'), 2, axis=1)

"Neutrino masses in normal and inverted hierarchies, resp."
m_nh = [0, np.sqrt(m21), np.sqrt(m31)]
m_ih = [np.sqrt(m31), np.sqrt(m31+m21), 0]

N = 101    #Number of plot points for magnetic moment

B = B_gal    #B-field
d = 1e6*kpc    #Distance travelled in galactic B-field
T = d/c    #Time for neutrinos to reach the Earth

U = PMNS(theta12, theta13, theta23)    #PMNS matrix

N = 20

mu_max = 4e-15

"Parameters for the initial flux of astrophysical neutrinos (taken from 1507.03991)"
phi_0 = 6.7
gamma = 2.5
E_min = 25 * TeV
E_max = 2.8 * PeV
sig = 7.81473    #chi-squared threshold for 3 dofs at 95% confidence level

h = np.sqrt(3/4)    #height of ternary plot

def ratios(f_in, m, Um):
    "Returns the final flavor ratios"
    f_temp = [sum([f_in[i] * Um[i, j]**2 for i in range(3)]) for j in range(3)]
    f = np.array([np.sum([f_temp[j] * Um[k, j]**2 for j in range(3)]) for k in range(3)])
    return f/np.sum(f)

def three_to_two (f):
    "Converts coordinates from ternary plot into cartesian ones (three coordinates to two)"
    return [[(f[i][1] + 2*f[i][0])/2, f[i][1]*h] for i in range(len(f))]

fs = [[0, 1, 0], [1, 2, 0]]    #Ratios at the source

# mus = np.linspace(0, max(np.abs(mu)), 2*N) * mu_b
# mus = np.linspace(0, 2e-14, 2*N) * mu_b

points_1v = np.loadtxt("f_010.txt").reshape((2*N, N**3, 3))
points_2v = np.loadtxt("f_120.txt").reshape((2*N, N**3, 3))

point_0 = np.array([0.3134207311070162, 0.3477778271260689, 0.33880144176691485])
points_10 = ratios(fs[0], m_nh, U)
points_20 = ratios(fs[1], m_nh, U)

alpha_shape = alphashape.alphashape(three_to_two(points_1v[0]), 0.)
alpha_shape2 = alphashape.alphashape(three_to_two(points_2v[0]), 0.)

"Color maps"
blues = plt.get_cmap('Blues')
greens = plt.get_cmap('Greens')
grey = plt.get_cmap('Greys')

cmbl = ListedColormap(blues(np.linspace(1.0, 0.25, 256)))
cmgr = ListedColormap(greens(np.linspace(1.0, 0.25, 256)))

fig, tax = ternary.figure()

fig.set_size_inches(7,5)

ax = tax.get_axes()

tax.gridlines(multiple=0.1, color='grey')
tax.boundary()
tax.set_background_color('white')
tax.bottom_axis_label("$\\nu_e$", offset=0.03, fontsize=16)
tax.right_axis_label("$\\nu_{\\mu}$", offset=0.1, fontsize=16)
tax.left_axis_label("$\\nu_{\\tau}$", offset=0.16, fontsize=16)

plt1, = ax.plot(np.transpose(three_to_two(contour1))[0], np.transpose(three_to_two(contour1))[1],
                color=grey(0.8), label='IceCube 2 yr')
tax.plot(contour2, color=grey(0.8))
tax.plot(contour3, color=grey(0.8))
plt8, = ax.plot(np.transpose(three_to_two(contour8+points_10-point_0))[0],
                np.transpose(three_to_two(contour8+points_10-point_0))[1], label='IceCube 8 yr $(68 \%)$')
plt15, = ax.plot(np.transpose(three_to_two(contour15+points_10-point_0))[0],
                 np.transpose(three_to_two(contour15+points_10-point_0))[1], label='IceCube 15 yr $(68 \%)$')
plt25, = ax.plot(np.transpose(three_to_two(contour1510+points_10-point_0))[0],
                 np.transpose(three_to_two(contour1510+points_10-point_0))[1],
                 label='IceCube 15 yr + Gen2 10yr $(68 \%)$')
plt_sc = ax.scatter(np.transpose(three_to_two(points_1v[0]))[0], 
                    np.transpose(three_to_two(points_1v[0]))[1], marker='.', color=cmbl(0),
            label='$(0:1:0)$', s=1)
tax.scatter([points_10], c='b', zorder=2)
for i in range(1, N):
    tax.scatter(points_1v[i], marker='.', color=cmbl(i/(2*N)), s=1)
ax.add_patch(PolygonPatch(alpha_shape, alpha=1, fill=False))
ax.axis('off')
ax.set_aspect(1)
ax.text(0.54, 0.52, "68\%", rotation=20, size=16)
ax.text(0.50, 0.59, "90\%", rotation=20, size=16)
tax.ticks(axis='lbr', linewidth=0.5, multiple=0.2, tick_formats="%.1f", offset=0.025, fontsize=16)

norm = Normalize(0.0, mu_max)

cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmbl), ax=ax)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Neutrino magnetic moment [$\\mu_B$]', size=16)

leg = ax.legend(handles=[plt1, plt8, plt15, plt25], loc=(0.15, 0.1), frameon=False)
ax.legend(handles=[plt_sc], frameon=False, loc=(-0.02, 0.8), markerscale=4,
            title=r"{\centering Initial composition\\$(\nu_e:\nu_\mu:\nu_\tau)$ \par}", title_fontsize=16,
            fontsize=16)
tax._redraw_labels()
ax.add_artist(leg)
plt.tight_layout()
plt.savefig("flavor_ratios_averaging_n.pdf")
plt.close()


# fig, tax = ternary.figure()

# fig.set_size_inches(7,5)

# ax = tax.get_axes()

# tax.gridlines(multiple=0.1, color='grey')
# tax.boundary()
# tax.set_background_color('white')
# tax.bottom_axis_label("$\\nu_e$", offset=0.03, fontsize=16)
# tax.right_axis_label("$\\nu_{\\mu}$", offset=0.1, fontsize=16)
# tax.left_axis_label("$\\nu_{\\tau}$", offset=0.16, fontsize=16)
# plt1, = ax.plot(np.transpose(three_to_two(contour1))[0], np.transpose(three_to_two(contour1))[1],
#                 color=grey(0.8), label='IceCube 2 yr')
# tax.plot(contour2, color=grey(0.8))
# tax.plot(contour3, color=grey(0.8))
# plt8, = ax.plot(np.transpose(three_to_two(contour8+points_20-point_0))[0],
#                 np.transpose(three_to_two(contour8+points_20-point_0))[1], label='IceCube 8 yr $(68 \%)$')
# plt15, = ax.plot(np.transpose(three_to_two(contour15+points_20-point_0))[0],
#                   np.transpose(three_to_two(contour15+points_20-point_0))[1], label='IceCube 15 yr $(68 \%)$')
# plt25, = ax.plot(np.transpose(three_to_two(contour1510+points_20-point_0))[0],
#                   np.transpose(three_to_two(contour1510+points_20-point_0))[1],
#                   label='IceCube 15 yr + Gen2 10yr $(68 \%)$')
# plt_sc = ax.scatter(np.transpose(three_to_two(points_2v[0]))[0], 
#                     np.transpose(three_to_two(points_2v[0]))[1], marker='.', color=cmgr(0),
#             label='$(1:2:0)$', s=1)
# tax.scatter([points_20], c='g', zorder=2)
# for i in range(1, 2*N):
#     tax.scatter(points_2v[i], marker='.', color=cmgr(i/(2*N)), s=1)
# ax.add_patch(PolygonPatch(alpha_shape2, alpha=1, fill=False))
# ax.axis('off')
# ax.set_aspect(1)
# ax.text(0.54, 0.52, "68\%", rotation=20, size=16)
# ax.text(0.50, 0.59, "90\%", rotation=20, size=16)
# tax.ticks(axis='lbr', linewidth=0.5, multiple=0.2, tick_formats="%.1f", offset=0.025, fontsize=16)

# norm = Normalize(0.0, mu_max)

# cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmgr), ax=ax)
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Neutrino magnetic moment [$\\mu_B$]', size=16)

# leg = ax.legend(handles=[plt1, plt8, plt15, plt25], loc=(0.15, 0.1), frameon=False)
# ax.legend(handles=[plt_sc], frameon=False, loc=(-0.02, 0.8), markerscale=4,
#             title=r"{\centering Initial composition\\$(\nu_e:\nu_\mu:\nu_\tau)$ \par}", title_fontsize=16,
#             fontsize=16)
# tax._redraw_labels()
# ax.add_artist(leg)
# plt.tight_layout()

# plt.savefig("flavor_ratios_averaging_n2.pdf")
# plt.close()
