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
d = 1e6*kpc    #Distance travelled in galactic B-field
T = d/c    #Time for neutrinos to reach the Earth

U = PMNS(theta12, theta13, theta23)    #PMNS matrix

mu = [np.pi*2*hbar/(2*T*B_gal*mu_b*U[1,i]) for i in range(3)]    #maximal magnetic moment to be considered
# m_idx = 0    #left-handed mass with which the sterile mass coincides
Mmu = np.array([[0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]])    #magnetic moment matrix in flavor basis

GeV = 1e9
TeV = 1e12
PeV = 1e15

"Generate the turbulent intracluster magnetic field"
din = 1e4*kpc
Nc = 100
lout = 1e3*kpc
kmin = 2*np.pi/lout
kmax = Nc * 2 * np.pi/din + kmin
R = np.linspace(kmin, kmax, int(Nc/2)+1)
Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(Nc/2)]
         else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * np.pi)
         for i in range(int(Nc/2)+1)]
Bk1 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(Nc/2)-1-i]) for i in range(Nc)])
Bx = np.real(np.fft.ifft(Bk1))
Bx = Bx*np.sqrt(Nc)*1e-10/(np.sqrt(np.sum(np.abs(Bx)**2)))
Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(Nc/2)]
         else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * np.pi)
         for i in range(int(Nc/2)+1)]
Bk2 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(Nc/2)-1-i]) for i in range(Nc)])
By = np.real(np.fft.ifft(Bk2))
By = By*np.sqrt(Nc)*1e-10/(np.sqrt(np.sum(np.abs(By)**2)))
Banxc1 = interpolate.interp1d(np.linspace(0, din/kpc, int(Nc/2)), Bx[:int(Nc/2)])
Banyc1 = interpolate.interp1d(np.linspace(0, din/kpc, int(Nc/2)), By[:int(Nc/2)])
Banxc2 = interpolate.interp1d(np.linspace(0, din/kpc, int(Nc/2)), Bx[int(Nc/2):])
Banyc2 = interpolate.interp1d(np.linspace(0, din/kpc, int(Nc/2)), By[int(Nc/2):])


"Generate the turbulent intergalactic magnetic field"
Nb = 1000
lout = 1e4*kpc
kmin = 2*np.pi/(5e3*kpc)
kmax = Nb * 2 * np.pi/d + kmin
R = np.linspace(kmin, kmax, int(Nb/2)+1)
Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(N/2)]
         else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * np.pi)
         for i in range(int(Nb/2)+1)]
Bk1 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(Nb/2)-1-i]) for i in range(Nb)])
Bx = np.real(np.fft.ifft(Bk1))
Bx = Bx*np.sqrt(Nb)*5e-13/(np.sqrt(np.sum(np.abs(Bx)**2)))
Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(Nb/2)]
         else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * np.pi)
         for i in range(int(Nb/2)+1)]
Bk2 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(Nb/2)-1-i]) for i in range(Nb)])
By = np.real(np.fft.ifft(Bk2))
By = By*np.sqrt(Nb)*5e-13/(np.sqrt(np.sum(np.abs(By)**2)))
Banx = interpolate.interp1d(np.linspace(0, d/kpc, Nb), Bx)
Bany = interpolate.interp1d(np.linspace(0, d/kpc, Nb), By)

rho0 = qp.basis(2, 0) * qp.basis(2, 0).dag()    #the initial density matrix

"The operators to track the diagonal elements of the density matrix"
op1 = qp.basis(2, 0) * qp.basis(2, 0).dag()
op2 = qp.basis(2, 1) * qp.basis(2, 1).dag()

H_dx = 0.5 * (-1j) * qp.Qobj([[ 0, 1],
                              [ -1, 0]])*kpc/(hbar*c)

H_dy = 0.5 * (-1j) * qp.Qobj([[ 0, 1j],
                              [ 1j, 0]])*kpc/(hbar*c)

dist = np.linspace(0, d/kpc, 1000)

options = qp.Options(nsteps=1E6)

def B_varx (t, args):
    "Generate variable B-field"
    if t <= din/kpc:
        return Banxc1(t)
    elif t<= (d-din)/kpc:
        return Banx(t)
    elif t<= d/kpc:
        return Banxc2(t-d/kpc+din/kpc)
    else:
        return 0

def B_vary (t, args):
    "Generate variable B-field"
    if t <= din/kpc:
        return Banyc1(t)
    elif t<= (d-din)/kpc:
        return Bany(t)
    elif t<= d/kpc:
        return Banyc2(t-d/kpc+din/kpc)
    else:
        return 0

def P_surv (mu):
    "Compute the survival probability"
    H_v = [[mu*H_dx, B_varx], [mu*H_dy, B_vary]]
    result = qp.mesolve(H_v, rho0, dist, e_ops=[op1, op2])
    return result.expect[0][-1]

N = 10

mu_max = 2e-15
mus = np.linspace(0, mu_max, 2*N) * mu_b
Ps = np.array([P_surv(mus[i]) for i in range(2*N)])
P_app = interpolate.interp1d(mus, Ps, 'cubic')

"Parameters for the initial flux of astrophysical neutrinos (taken from 1507.03991)"
phi_0 = 6.7
gamma = 2.5
E_min = 25 * TeV
E_max = 2.8 * PeV
sig = 7.81473    #chi-squared threshold for 3 dofs at 95% confidence level

h = np.sqrt(3/4)    #height of ternary plot

def ratios(f_in, mmu, m, Um):
    "Returns the final flavor ratios"
    mu = np.linalg.multi_dot([np.transpose(Um[:3, :3]), mmu, Um[:3, :3]])
    # f_temp = [sum([f_in[i] * Um[i, j]**2 for i in range(3)]) * P_surv(mu[j, j]) for j in range(3)]
    f_temp = [sum([f_in[i] * Um[i, j]**2 for i in range(3)]) * P_app(mu[j, j]) for j in range(3)]
    f = np.array([sum([f_temp[j] * Um[k, j]**2 for j in range(3)]) for k in range(3)])
    return f/np.sum(f)

def three_to_two (f):
    "Converts coordinates from ternary plot into cartesian ones (three coordinates to two)"
    return [[(f[i][1] + 2*f[i][0])/2, f[i][1]*h] for i in range(len(f))]

fs = [[0, 1, 0], [1, 2, 0]]    #Ratios at the source

# mus = np.linspace(0, max(np.abs(mu)), 2*N) * mu_b
# mus = np.linspace(0, 2e-14, 2*N) * mu_b

points_1v = np.empty((2*N, N**3, 3))
points_2v = np.empty((2*N, N**3, 3))

for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(2*N):
                # Mmu = np.array([[np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)],
                #                 [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)],
                #                 [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]])
                th13 = 8.61 + (2*i/N - 1)*np.sqrt(sig)*0.13
                th12 = 33.82 + (2*j/N - 1)*np.sqrt(sig*(1-(2*i/N-1)**2))*0.78
                th23 = 48.3 + (2*k/N - 1)*np.sqrt(sig*(1-(2*i/N-1)**2)*(1-(2*j/N-1)**2))*1.9
                U = PMNS(np.radians(th12), np.radians(th13), np.radians(th23))
                points_1v[l, i*N**2 + j*N + k] = ratios(fs[0], mus[l]*Mmu, m_nh, U)
                # points_2v[l, i*N**2 + j*N + k] = ratios(fs[1], mus[l]*Mmu, m_nh, U)

alpha_shape = alphashape.alphashape(three_to_two(points_1v[0]), 0.)
# alpha_shape = alphashape.alphashape(three_to_two(points_2v[0]), 0.)

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
tax.plot(contour1, color=grey(0.8))
tax.plot(contour2, color=grey(0.8))
tax.plot(contour3, color=grey(0.8))
tax.scatter(points_1v[0], marker='.', color=cmbl(0),
            label=r'$(1:2:0)$', s=0.5)
# tax.scatter(points_2v[0], marker='.', color=cmgr(0),
#             label='$(0:1:0)$', s=0.5)
for i in range(1, 2*N):
    tax.scatter(points_1v[i], marker='.', color=cmbl(i/(2*N)), s=1)
    # tax.scatter(points_2v[i], marker='.', color=cmgr(i/(2*N)), s=1)
# line_seg_1 = LineCollection([three_to_two(points_b[0][i:i+2]) for i in range(N-1)],
#                             cmap=cmgr, linewidths=2)
# line_seg_2 = LineCollection([three_to_two(points_b[1][i:i+2]) for i in range(N-1)],
#                             cmap=cmbl, linewidths=2)
# line_seg_1.set_array(mus/mu_b)
# line_seg_2.set_array(mus/mu_b)
# ax.add_collection(line_seg_1)
# ax.add_collection(line_seg_2)
ax.add_patch(PolygonPatch(alpha_shape, alpha=1, fill=False))
ax.axis('off')
ax.set_aspect(1)
ax.text(0.54, 0.52, "68\%", rotation=20, size=16)
ax.text(0.50, 0.59, "90\%", rotation=20, size=16)
tax.ticks(axis='lbr', linewidth=0.5, multiple=0.2, tick_formats="%.1f", offset=0.025, fontsize=16)

# norm = Normalize(0.0, abs(mu[m_idx]))
# norm = Normalize(0.0, max(np.abs(mu)))
norm = Normalize(0.0, mu_max)

cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmbl), ax=ax)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Neutrino magnetic moment [$\\mu_B$]', size=16)
# fig.colorbar(line_seg_2)

# s = ('Flavor ratios for transition magnetic moment. We set all $\\mu_{xy}=$' f'{mu[1, 0]/mu_b}'
#       f'$\\mu_B$, $E_\\nu =$ 1PeV and consider a magnetic field of strength {B_gal*1e10} $\\mu G$.')
# tax.get_axes().text( 0.5, -0.1, s, horizontalalignment='center', verticalalignment='center',
#                     transform=tax.get_axes().transAxes)

tax.legend(frameon=False, loc=(-0.02, 0.8), markerscale=4,
           title=r"{\centering Initial composition\\$(\nu_e:\nu_\mu:\nu_\tau)$ \par}", title_fontsize=16,
           fontsize=16)
tax._redraw_labels()
plt.tight_layout()
plt.savefig("flavor_ratios_varying_mu.pdf")