"""Module containing definitions for applications involving neutrino oscillations and magnetic moment"""
import qutip as qp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy.random as rnd
from scipy import special, integrate, optimize, interpolate
import pathos

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

# The operators to determine the diagonal elements of the density matrix
rho11 = qp.basis(6, 0) * qp.basis(6, 0).dag()
rho22 = qp.basis(6, 1) * qp.basis(6, 1).dag()
rho33 = qp.basis(6, 2) * qp.basis(6, 2).dag()
rho44 = qp.basis(6, 3) * qp.basis(6, 3).dag()
rho55 = qp.basis(6, 4) * qp.basis(6, 4).dag()
rho66 = qp.basis(6, 5) * qp.basis(6, 5).dag()

# Neutrino mixing parameters - NuFit 5.1, normal ordering
theta12 = 33.44 * np.pi/180.
theta13 =  8.57 * np.pi/180.
theta23 = 49.2  * np.pi/180.
m21 = 7.42e-5
m31 = 2.515e-3
m_nh = [0, np.sqrt(m21), np.sqrt(m31)]
m_ih = [np.sqrt(m31), np.sqrt(m31+m21), 0]

# error on neutrino mixing parameters
d_theta12 = 0.76 * np.pi/180.
d_theta13 = 0.13 * np.pi/180.
d_theta23 = 1.2  * np.pi/180.
d_m21     = 0.21e-5
d_m31     = 0.028e-3

# unit conversion
MeV = 1e6
GeV = 1e9
TeV = 1e12
PeV = 1e15

# flavor structures of flavor-universal and muon-only magnetic moments
M_all_flavors = np.array([[1,0,0],[0,1,0],[0,0,1]])
M_mu_only     = np.array([[0,0,0],[0,1,0],[0,0,0]])

n_e_HK = 374e9*mol*10/18    #Number of target electrons/protons in HK
n_O_HK = 374e9*mol/18    #Number of target oxygen atoms in HK
n_ar_DN = 40e9*mol/39.95    #number of target Argon atoms in DUNE
n_e_DN = 40e9*mol*18/39.95    #number of target electrons in DUNE


#---------------------------------------------------------------------------

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

#---------------------------------------------------------------------------

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

#--------------------------------------------------------------------

class neutrino_propagator:
    """Routines for generating the Galactic magnetic field structure
       and propagating neutrinos through it."""
    
    def __init__(self, d=10*kpc, theta_los=0., phi_los=np.pi, a_B_coh=0., Bturb=2.,
                 outer_scale=0.01, mu_range=np.linspace(0, 4e-13, 20)*mu_b):
        """Initialize the object and generate initial B field map.
           
           Parameters:
               d:            distance to the neutrino source
               phi_los:      orientation of the line of sight along the Galactic plane.
                             0 corresponds to a l.o.s. pointing away from the Galactic Center
               theta_los:    orientation of the line of sight relative to the Galactic plane
               a_B_coh:      nuisance parameter describing the shift in the strength
                             of thw homogeneous (large-scale) magnetic field relative
                             to the fiducial model from https://arxiv.org/abs/0704.0458
               Bturb:        field strength of turbulent magnetic field [muG]
               outer_scale:  outer scale of B-field turbulence in kpc
               mu_range:     the range of magnetic moment values
                             over which the survival probability is tabulated"""
        
        self.d         = d
        self.theta_los = theta_los
        self.phi_los   = phi_los

        # load cross-section data [in 1e-43 cm^2]
        self.sigma            = {}
        self.sigma['e_CC']    = np.loadtxt('./cross_sections/nu_e_CC.csv').transpose()
        self.sigma['bare_CC'] = np.loadtxt('./cross_sections/nubar_CC.csv').transpose()
        self.sigma['e_ES']    = np.loadtxt('./cross_sections/nu_e_ES.csv').transpose()
        self.sigma['bare_ES'] = np.loadtxt('./cross_sections/nu_bar_e_ES.csv').transpose()
        self.sigma['e_O']     = np.loadtxt('./cross_sections/nu_e_O.csv').transpose()
        self.sigma['e_O'][1] *= 1e43
        self.sigma['bare_O']  = np.loadtxt('./cross_sections/nubar_e_O.csv').transpose()
        self.sigma['bare_O'][1] *= 1e43
        self.sigma['IBD']     = np.loadtxt('./cross_sections/IBD.csv').transpose()
        self.sigma['IBD'][1] *= 1e43
        self.sigma['x_NC']    = np.loadtxt('./cross_sections/nu_NC.csv').transpose()
        self.sigma['barx_NC'] = np.loadtxt('./cross_sections/nu_bar_NC.csv').transpose()
        self.sigma['x_ES']    = np.loadtxt('./cross_sections/nu_x_ES.csv').transpose()
        self.sigma['barx_ES'] = np.loadtxt('./cross_sections/nu_bar_x_ES.csv').transpose()

        self.sigma['HK']     = [[self.sigma['e_ES'], self.sigma['e_O']],
                                [self.sigma['x_ES'], [[0],[0]]]]
        self.sigma['bar_HK'] = [[self.sigma['bare_ES'], self.sigma['bare_O'], self.sigma['IBD']],
                                [self.sigma['barx_ES'], [[0],[0]],            [[0],[0]]]]
        self.sigma['DN']     = [[self.sigma['e_CC'], self.sigma['e_ES']],
                                [self.sigma['x_NC'], self.sigma['x_ES']]]
        self.sigma['bar_DN'] = [[self.sigma['bare_CC'], self.sigma['bare_ES']],
                                [self.sigma['barx_NC'], self.sigma['barx_ES']]]

        # generate default B field map
        self.generate_B_field_gal(d=d, theta_los=theta_los, phi_los=phi_los,
                                  a_B_coh=a_B_coh, Bturb=Bturb, outer_scale=outer_scale,
                                  mu_range=mu_range)
    
    #-----------------------------------------------------------------------

    def sigma_extract(self, sigma, x):
        """Returns the cross section for neutrino energy x in m^2"""
        if x < sigma[0][0] or x > sigma[0][-1]:
            return 0
        else:
            return np.interp(x, sigma[0], sigma[1])*1e-47

    #-----------------------------------------------------------------------

    def generate_B_field_gal(self, d=10*kpc, theta_los=0., phi_los=np.pi, a_B_coh=0., Bturb=2.,
                             outer_scale=0.01, mu_range=np.linspace(0, 4e-13, 20)*mu_b,
                             random_B_coh=False, plot=False, cpus=1):
        """Generate a new Galactic B field map with the given parameters
           (and randomized turbulence).
           
           Parameters:
               d:            distance to the neutrino source
               phi_los:      orientation of the line of sight along the Galactic plane.
                             0 corresponds to a l.o.s. pointing away from the Galactic Center
               theta_los:    orientation of the line of sight relative to the Galactic plane
               a_B_coh:      nuisance parameter describing the shift in the strength
                             of thw homogeneous (large-scale) magnetic field relative
                             to the fiducial model from https://arxiv.org/abs/0704.0458
               Bturb:        field strength of turbulent magnetic field [muG]
               outer_scale:  outer scale of B-field turbulence in kpc
               mu_range:     the range of magnetic moment values
                             over which the survival probability is tabulated
               random_B_coh: if True, pick random B fields for each of the regions
                             in the coherent B field model. Values are chosen from
                             a Gaussian of width 1 muG
               plot:         if True, create plot of coherent B-field map
               cpus:         number of CPUs to use in tabulating oscillation
                             probabilities"""
        
        self.d         = d
        self.theta_los = theta_los
        self.phi_los   = phi_los
        N              = 1000              # number of sampling points along line of sight
        d_table        = np.linspace(0, d/kpc, N) # dicretized line-of-sight coordinates

        lout = outer_scale*kpc             # outer scale of turbulence
        kmin = 2*np.pi/(1e-1*lout)
        kmax = N * 2 * np.pi/d + kmin
        R = np.linspace(kmin, kmax, int(N/2)+1)
        
        Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(N/2)]
                 else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * np.pi)
                 for i in range(int(N/2)+1)]
        Bk1 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(N/2)-1-i]) for i in range(N)])
        Bx = np.real(np.fft.ifft(Bk1))
        Bx = Bx*np.sqrt(N)*Bturb*1e-10/(np.sqrt(np.sum(np.abs(Bx)**2))) # factor 1e-10: conversion from muG to Tesla
        
        Btemp = [np.random.normal(0, R[i]**(-11/6)) if i in [0, int(N/2)]
                 else np.random.normal(0, R[i]**(-11/6))*np.exp(1j * np.random.uniform(0, 1) * 2 * np.pi)
                 for i in range(int(N/2)+1)]  # FIXME what is the origin of the factor of 2? (JK)
        Bk2 = np.array([Btemp[i] if i <= N/2 else np.conjugate(Btemp[int(N/2)-1-i]) for i in range(N)])
        By = np.real(np.fft.ifft(Bk2))
        By = By*np.sqrt(N)*Bturb*1e-10/(np.sqrt(np.sum(np.abs(By)**2)))
        self.Banx = interpolate.interp1d(d_table, Bx)
        self.Bany = interpolate.interp1d(d_table, By)
        
        # coherent magnetic field
        pitch_angle    = 11.5*np.pi/180. # pitch angle (https://arxiv.org/abs/0704.0458)
        pitch          = np.tan(pitch_angle)
        phi0_table     = np.array([360, 300, 270, 225, 180, 140, 105, 40]) * np.pi/180.
                                         # azimuthal angles of arm boundaries relative
                                         # to our x axis; read from fig. 4 in
                                         # https://arxiv.org/abs/0704.0458
        B_coh_table    = (1 + a_B_coh) * 1e-10 * np.array([
                                   1.0,  # molecular ring between r=3 kpc and r=5 kpc
                                   1.5, -1.0, -0.5, -0.05, -1.0, -0.5, -0.3, -0.7 ])  # arms
        r_B_table      = np.array([3.7,  # approx. radius (in kpc) at which B field was read from the plot
                                   7.0,  7.7,  7.6,  12.2, 16.7, 17.3,  7.0,  6.5 ])
        X_Earth        = np.array([8.5,0,0]) # our location in the Milky Way
        X_los_table    = d_table[None,:] * np.array([np.cos(theta_los) * np.cos(phi_los), # l.o.s. (x,y,z) coordinates
                                                     np.cos(theta_los) * np.sin(phi_los),
                                                    -np.sin(theta_los)])[:,None] + X_Earth[:,None]
        r_los_table    = np.sqrt(X_los_table[0]**2 + X_los_table[1]**2)
                                         # radial coordinate in Galactic plane
        phi_los_table  = np.arctan2(X_los_table[1], X_los_table[0])
                                         # azimuthal coordinate in Galactic plane
        Bcoh_map       = np.zeros((3,N)) # coherent B field in the galactic plane;
                                         #   the first two entries along axis 0
                                         #   correspond to the two axes of that plane;
                                         #   axis 1 corresponds to the l.o.s. coordinate 
                                         #   projected onto the Galactic plane
                    
        # random coherent B fields?
        if random_B_coh:
            B_coh_table = (1 + a_B_coh) * 1e-10 * rnd.normal(scale=1., size=len(B_coh_table))
        
        # B-field of molecular ring
        ii             = ((3<r_los_table) & (r_los_table<5))
        Bcoh_map[:2,ii] = B_coh_table[0] * r_B_table[0] / r_los_table[ii] \
                            * np.array([-np.sin(phi_los_table[ii]), np.cos(phi_los_table[ii])])
        
        # B-field in spiral arm region: loop until we've left the galaxy or the l.o.s.
        k = 0
        while True:
            for phi0, B, r_B in zip(phi0_table, B_coh_table[1:], r_B_table[1:]):
                # find l.o.s. points outside current spiral (but still within the galaxy)
                r_spiral_table = 5*np.exp(pitch*(phi_los_table - phi0 + k*2*np.pi))
                ii = (  (r_los_table >  5) & (r_los_table < 20) & (r_los_table >= r_spiral_table) )
                Bcoh_map[:2,ii] = B * r_B / r_los_table[ii] \
                                    * np.array([-np.sin(phi_los_table[ii] - pitch_angle),
                                                 np.cos(phi_los_table[ii] - pitch_angle)])
            k = k + 1
            if np.count_nonzero(r_los_table[ii]) == 0:
                break
                
        # set B field to zero outside the Galactic plane (i.e. more than 1 kpc from the midplane)
        Bcoh_map[:, np.abs(X_los_table[2]) > 1] = 0.
        
        # transform from B field components in the Galactic plane to components
        # perpendicular to the line of sight.
        # The rotation matrix V rotates the line of sight onto the z-axis
        V = np.array([ [-np.cos(phi_los)**2 * np.sin(theta_los) + np.sin(phi_los)**2,
                         np.cos(phi_los) * np.sin(phi_los) * (-np.sin(theta_los) - 1),
                        -np.cos(phi_los) * np.cos(theta_los) ],
                       [ np.cos(phi_los) * np.sin(phi_los) * (-np.sin(theta_los) - 1),
                         np.cos(phi_los)**2 - np.sin(theta_los) * np.sin(phi_los)**2,
                        -np.sin(phi_los) * np.cos(theta_los) ],
                       [ np.cos(phi_los) * np.cos(theta_los),
                         np.sin(phi_los) * np.cos(theta_los),
                        -np.sin(theta_los) ] ])
        Bcoh_map = np.dot(V, Bcoh_map)
                
        # interpolate coherent B field along line of sight
        self.Bcoh_x = interpolate.interp1d(d_table, Bcoh_map[0])
        self.Bcoh_y = interpolate.interp1d(d_table, Bcoh_map[1])
        
        # plots of coherent B-field component
        if plot:
            fig = plt.figure(figsize=(14,6))
            ax1 = plt.subplot(121)

            # load background image of Milky Way
            mw_img = plt.imread('data/milky-way.jpg')
            d_img  = 41.73 # kpc
            ax1.imshow(mw_img, extent=[-d_img/2.,d_img/2.,-d_img/2.,d_img/2.],
                       cmap='gray', vmin=0, vmax=155)

            # generate 2d map of B field strength for the plot
            x_table   = np.linspace(-20., 20., 300)    # x/y range of B field map for plot
            r_table   = np.sqrt(x_table[:,None]**2 + x_table[None,:]**2)
            phi_table = np.arctan2(x_table[None,:], x_table[:,None])
            B_table   = np.zeros((len(x_table), len(x_table))) 
        
            ii          = ((3<r_table) & (r_table<5))  # molecular ring
            B_table[ii] = B_coh_table[0] * r_B_table[0] / r_table[ii]

            k = 0                                      # spiral arms
            while True:
                for phi0, B, r_B in zip(phi0_table, B_coh_table[1:], r_B_table[1:]):
                    # find l.o.s. points outside current spiral (but still within the galaxy)
                    r_spiral_table = 5*np.exp(pitch*(phi_table - phi0 + k*2*np.pi))
                    ii = (  (r_table > 5) & (r_table < 20) & (r_table >= r_spiral_table) )
                    B_table[ii] = B * r_B / r_table[ii]
                k = k + 1
                if np.count_nonzero(r_table[ii]) == 0:
                    break

#            clip_path  = matplotlib.path.Path([[0,-20],[0,0],[-20,25],[-20,-20],[0,-20]])
            clip_path  = matplotlib.path.Path([[0,-20],[0,0],[20,25],[20,-20],[0,-20]])
            clip_patch = matplotlib.patches.PathPatch(clip_path, fc='None', ec='#00000077')
            ax1.add_patch(clip_patch)
            B_plot = ax1.imshow(B_table.T*1e10, vmin=-2., vmax=2., origin='lower',
                                extent=[min(x_table), max(x_table), min(x_table), max(x_table)],
                                cmap='RdYlBu', clip_path=clip_patch, clip_on=True, alpha=0.7)

            # draw contours of spiral arms and other decorations
            phi_table_plot = np.linspace(0, 3*np.pi, 100)
            for phi0 in phi0_table:
                ax1.plot(5*np.exp(pitch*phi_table_plot) * np.cos(phi_table_plot + phi0), 
                         5*np.exp(pitch*phi_table_plot) * np.sin(phi_table_plot + phi0),
                         color='#99000077', lw=1)
            ax1.add_artist(plt.Circle((0,0), 5, ec='#99000077', color='None', lw=1))
            ax1.add_artist(plt.Circle((0,0), 3, ec='#99000077', color='None', lw=1))

            ax1.annotate(r'$\boldsymbol{\pmb\bigoplus}$', X_Earth[:2], color='#00eeee',
                         ha='center', va='center', size=20)
            ax1.arrow(X_Earth[0], X_Earth[1],
                      r_los_table[-1]*np.cos(phi_los_table[-1]) - X_Earth[0],
                      r_los_table[-1]*np.sin(phi_los_table[-1]) - X_Earth[1],
                      color='#44ffff', length_includes_head=True, width=0.1,
                      head_width=0.8, zorder=5)
            ax1.set_xlim(-15,15)
            ax1.set_ylim(-15,15)
#            ax1.axis('off')
            ax1.xaxis.set_ticks(np.arange(-15, 15.1, 5))
            ax1.yaxis.set_ticks(np.arange(-15, 15.1, 5))
            ax1.set_xlabel('x [kpc]')  # note: this plot is rotated 90 degrees clockwise
            ax1.set_ylabel('y [kpc]')  #   compared to fig. 4 of https://arxiv.org/abs/0704.0458
            ax1.grid()
            fig.colorbar(B_plot, ax=ax1, shrink=0.87, aspect=18, alpha=0.7, extend='both')
            ax1.annotate(r'$\vec{B}$~[$\mu$G]', (15,16.3), annotation_clip=False)

            ax2 = plt.subplot(122)
            ax2.plot(r_los_table,np.sqrt(Bcoh_map[0]**2+Bcoh_map[1]**2), label=r'$|B|$')
            ax2.plot(r_los_table, 1e10*Bcoh_map[0], label=r'$B_x$ (coh.)')
            ax2.plot(r_los_table, 1e10*Bcoh_map[1], label=r'$B_y$ (coh.)', ls='--')
#            ax2.plot(r_los_table, 1e10*Bx, label=r'$B_x$ (turb.)')
#            ax2.plot(r_los_table, 1e10*By, label=r'$B_y$ (turb.)', ls='--')
            ax2.set_ylim(-2.0,2.0)
            ax2.set_xlabel('galactic radius [kpc]')
            ax2.set_ylabel('B field [$\mu$G]')
            ax2.legend(loc='upper right')
            ax2.grid()
            plt.show()        
            
        # tabulate survival probabilities
        rho0 = qp.basis(2, 0) * qp.basis(2, 0).dag() # initial density matrix
        op1  = qp.basis(2, 0) * qp.basis(2, 0).dag() # operators to track the diagonal elements
        op2  = qp.basis(2, 1) * qp.basis(2, 1).dag() #    of the density matrix
        H_dx = 0.5 * (-1j) * qp.Qobj([[ 0, 1],
                                      [ -1, 0]])*kpc/(hbar*c)
        H_dy = 0.5 * (-1j) * qp.Qobj([[ 0, 1j],
                                      [ 1j, 0]])*kpc/(hbar*c)

        dist = np.linspace(0, d/kpc, 100)
        options = qp.Options(nsteps=1E6)

        def B_varx(t, args):
            "Generate variable B-field"
            if t <= d/kpc:
                return self.Bcoh_x(t) + self.Banx(t)
            else:
                return 0

        def B_vary(t, args):
            "Generate variable B-field"
            if t <= d/kpc:
                return self.Bcoh_y(t) + self.Bany(t)
            else:
                return 0

        def P_surv(mu):
            "Compute the survival probability"
            H_v = [[mu*H_dx, B_varx], [mu*H_dy, B_vary]]
            result = qp.mesolve(H_v, rho0, dist, e_ops=[op1, op2],
                                options=qp.Options(nsteps=1E8))
            return result.expect[0][-1]

        if cpus > 1:
            with pathos.pools.ProcessPool(nodes=cpus) as pool:
                pool.restart()  # not sure why this is necessary, but without it,
                                # the code sometimes doesn't "forget" old results
                Ps = np.array(pool.map(lambda mu: P_surv(mu), mu_range))
        else:
            Ps = np.array([P_surv(mu) for mu in mu_range])

        # mus = np.linspace(0, np.pi, 200)
        # Ps = np.array([1-np.sin(mus[i])**2 for i in range(200)])

#        if len(mu_range) >= 4:
#            self.P_app = interpolate.interp1d(mu_range, Ps, 'cubic')
#        elif len(mu_range) >= 2:
        if len(mu_range) >= 2:
            self.P_app = interpolate.interp1d(mu_range, Ps, 'linear')
        elif len(mu_range) == 1:
            self.P_app = lambda x: Ps[0]
        else:
            raise ValueError('invalid mu_range')

        if plot:
            return fig
            
    #-----------------------------------------------------------------------

    def generate_B_field_extragal(self, d=1e6*kpc, d_in_cluster=1e4*kpc, B_cluster=1.,
                                  B_extragal=0.005, Nbs=1, verbosity=0):
        """Generate new extra-galactic B field maps with the given parameters
           (and randomized turbulence).
           
           Parameters:
               d:            distance to the neutrino source
               d_in_cluster: distance traveled inside galaxy cluster
               B_cluster:    intracluster magnetic field [muG]
               B_extragal:   magnetic field in between galaxy clusters [muG]
               Nbs:          number of B-field profiles to generate
               verbosity:    if > 0, print out extra status information"""
        
        self.d     = d

        N          = 1000  # number of sampling points along line of sight
        Nc         = 100   # number of sampling points for cluster magnetic fields
        Nb         = 1000  # number of sampling points for intercluster magnetic field

        B_cluster_table  = np.random.normal(scale=B_cluster,  size=Nbs)
        B_extragal_table = np.random.normal(scale=B_extragal, size=Nbs)

        # Generate the turbulent intracluster magnetic field
        if verbosity > 0:
            print("generating intracluster field - x direction ...")
        lout = 1e3*kpc    # outer scale of turbulence
        kmin = 2*np.pi/lout
        kmax = Nc * np.pi/d_in_cluster + kmin
        R = np.linspace(kmin, kmax, int(Nc/2)+1)
        Btemp = np.array([[np.random.normal(0, R[i]**(-11/6)) if i in [0, int(Nc/2)]
                      else np.random.normal(0, R[i]**(-11/6))
                            * np.exp(1j * np.random.uniform(0, 1) * np.pi)
                      for i in range(int(Nc/2)+1)] for j in range(Nbs)])
#        Btemp = np.array([ np.random.normal(scale=R[i]**(-11/6), size=Nbs)
#                       * (1 if i in [0, int(Nc/2)]
#                            else np.exp(1j * np.random.uniform(size=Nbs) * np.pi))
#                          for i in range(int(Nc/2)+1) ]).T 
#                     # faster and more Pythonic, but not really necessary here
        Bk1 = np.array([[Btemp[j][i] if i <= Nc/2
                    else np.conjugate(Btemp[j][int(Nc//2)-1-i])
                         for i in range(Nc)] for j in range(Nbs)])
                        # FIXME: replaced N -> Nc (and similarly below)
        Bx = [np.real(np.fft.ifft(Bk1[j])) for j in range(Nbs)]
        Bx_c = [Bx[j]*np.sqrt(N)*B_cluster_table[j]*1e-10/(np.sqrt(np.sum(np.abs(Bx[j])**2)))
                  for j in range(Nbs)]
                                # factor 1e-10: B field conversion from \muG to Tesla
            
        if verbosity > 0:
            print("generating intracluster field - y direction ...")
        Btemp = np.array([[np.random.normal(0, R[i]**(-11/6)) if i in [0, int(Nc/2)]
                      else np.random.normal(0, R[i]**(-11/6))
                            * np.exp(1j * np.random.uniform(0, 1) * 2 * np.pi)
                      for i in range(int(Nc/2)+1)] for j in range(Nbs)])
        Bk2 = np.array([[Btemp[j][i] if i <= Nc/2
                    else np.conjugate(Btemp[j][int(Nc/2)-1-i])
                         for i in range(Nc)] for j in range(Nbs)])
        By = [np.real(np.fft.ifft(Bk2[j])) for j in range(Nbs)]
        By_c = [By[j]*np.sqrt(N)*B_cluster_table[j]*1e-10/(np.sqrt(np.sum(np.abs(By[j])**2)))
                  for j in range(Nbs)]
        
        # Generate the turbulent intergalactic magnetic field
        if verbosity > 0:
            print("generating intercluster field - x direction ...")
        lout = 1e4*kpc     # outer scale of turbulence
        kmin = 2*np.pi/lout
        kmax = Nb * 2 * np.pi/(d - 2*d_in_cluster) + kmin
        R = np.linspace(kmin, kmax, int(Nb/2)+1)
        Btemp = np.array([[np.random.normal(0, R[i]**(-11/6)) if i in [0, int(Nb/2)]
                      else np.random.normal(0, R[i]**(-11/6))
                            * np.exp(1j * np.random.uniform(0, 1) * np.pi)
                      for i in range(int(Nb/2)+1)] for j in range(Nbs)])
        Bk1 = np.array([[Btemp[j][i] if i <= Nb/2
                    else np.conjugate(Btemp[j][int(Nb/2)-1-i])
                         for i in range(Nb)] for j in range(Nbs)])
        Bx = [np.real(np.fft.ifft(Bk1[j])) for j in range(Nbs)]
        Bx = [Bx[j]*np.sqrt(N)*B_extragal_table[j]*1e-10/(np.sqrt(np.sum(np.abs(Bx[j])**2)))
                for j in range(Nbs)]
        
        if verbosity > 0:
            print("generating intercluster field - y direction ...")
        Btemp = np.array([[np.random.normal(0, R[i]**(-11/6)) if i in [0, int(Nb/2)]
                      else np.random.normal(0, R[i]**(-11/6))
                            * np.exp(1j * np.random.uniform(0, 1) * 2 * np.pi)
                      for i in range(int(Nb/2)+1)] for j in range(Nbs)])
        Bk2 = np.array([[Btemp[j][i] if i <= Nb/2
                    else np.conjugate(Btemp[j][int(Nb/2)-1-i])
                         for i in range(Nb)] for j in range(Nbs)])
        By = [np.real(np.fft.ifft(Bk2[j])) for j in range(Nbs)]
        By = [By[j]*np.sqrt(N)*B_extragal_table[j]*1e-10/(np.sqrt(np.sum(np.abs(By[j])**2)))
                for j in range(Nbs)]
        
        d_table = np.concatenate(( np.linspace(0,                        d_in_cluster, int(Nc/2)), 
                                   np.linspace(d_in_cluster+0.001*kpc,   d-d_in_cluster, Nb),
                                   np.linspace(d-d_in_cluster+0.001*kpc, d, int(Nc/2)) )) / kpc
        self.B_extragal_x = [ interpolate.interp1d(d_table,
                  np.concatenate((Bx_c[j][:int(Nc/2)], Bx[j], Bx_c[j][int(Nc/2):])),
                  bounds_error=False, fill_value=0.) for j in range(Nbs) ]
        self.B_extragal_y = [ interpolate.interp1d(d_table,
                  np.concatenate((By_c[j][:int(Nc/2)], By[j], By_c[j][int(Nc/2):])),
                  bounds_error=False, fill_value=0.) for j in range(Nbs) ]

        # tabulate oscillation probabilities
        # JK - we don't do this any more as it costs extra time,
        # and the sampling resolution required for extragalactic
        # B-fields is so high that it is more efficient to
        # just compute oscillation probabilities on the fly as
        # we simulate random parameter points (see P_osc_extragal below)
#        rho0 = qp.basis(2, 0) * qp.basis(2, 0).dag() # initial density matrix
#        op1  = qp.basis(2, 0) * qp.basis(2, 0).dag() # operators to track diag. elements
#        op2  = qp.basis(2, 1) * qp.basis(2, 1).dag() #    of the density matrix
#        H_dx = 0.5 * (-1j) * qp.Qobj([[ 0, 1],
#                                      [ -1, 0]])*kpc/(hbar*c)
#        H_dy = 0.5 * (-1j) * qp.Qobj([[ 0, 1j],
#                                      [ 1j, 0]])*kpc/(hbar*c)
#        dist = np.linspace(0, d/kpc, 1000)
#
#        def P_surv(mu, i):
#            """Compute the survival probability"""
#
#            print("mu = ", mu/mu_b) # FIXME
#            def B_varx(t, args):
#                """Generate variable B-field in x-direction"""
#                return self.B_extragal_x[i](t)
#
#            def B_vary(t, args):
#                """Generate variable B-field in y-direction"""
#                return self.B_extragal_y[i](t)
#
#            H_v = [[mu*H_dx, B_varx], [mu*H_dy, B_vary]]
#            result = qp.mesolve(H_v, rho0, dist, e_ops=[op1, op2],
#                                options=qp.Options(nsteps=1E8))
#            return result.expect[0][-1]
#
#        if verbosity > 0:
#            print("tabulating oscillation probabilities ...")
#        Ps = np.array([[P_surv(mu, j) for mu in mu_range] for j in range(Nbs)])
#        if len(mu_range) >= 4:
#            self.P_app = [interpolate.interp1d(mu_range, Ps[j], 'cubic')
#                          for j in range(Nbs)]
#        elif len(mu_range) >= 2:
#            self.P_app = [interpolate.interp1d(mu_range, Ps, 'linear')
#                          for j in range(Nbs)]
#        elif len(mu_range) == 1:
#            self.P_app = [lambda x: Ps[0] for j in range(Nbs)]
#        else:
#            raise ValueError('invalid mu_range')

    #-----------------------------------------------------------------------

    def P_osc_extragal(self, mu, idx=0):
        """compute the oscillation probabilities of neutrino in the
           extragalctic magnetic fields

           Parameters:
               mu:  the neutrino magnetic moment
               idx: the index of the pre-computed B-field configuration
                    to use"""

        if not hasattr(self, 'B_extragal_x'):
            raise ValueError('extragalactic B field configuration not initialized.')
        if idx > len(self.B_extragal_x):
            raise ValueError('invalid B field configuration index: {:d}'.format(idx))

        rho0 = qp.basis(2, 0) * qp.basis(2, 0).dag() # initial density matrix
        op1  = qp.basis(2, 0) * qp.basis(2, 0).dag() # operators to track diagonal
        op2  = qp.basis(2, 1) * qp.basis(2, 1).dag() #   elements of the density matrix
        H_dx = 0.5 * (-1j) * qp.Qobj([[ 0, 1],
                                      [ -1, 0]])*kpc/(hbar*c)
        H_dy = 0.5 * (-1j) * qp.Qobj([[ 0, 1j],
                                      [ 1j, 0]])*kpc/(hbar*c)
        dist = np.linspace(0, self.d/kpc, 1000)

        def B_varx(t, args):
            """Generate variable B-field in x-direction"""
            return self.B_extragal_x[idx](t)

        def B_vary(t, args):
            """Generate variable B-field in y-direction"""
            return self.B_extragal_y[idx](t)

        H_v = [[mu*H_dx, B_varx], [mu*H_dy, B_vary]]
        result = qp.mesolve(H_v, rho0, dist, e_ops=[op1, op2],
                            options=qp.Options(nsteps=1E8))
        return result.expect[0][-1]
    
    #-----------------------------------------------------------------------

    def propagate_sn_neutrinos(self, mu, d=None, theta_los=None, phi_los=None, mh='NH',
                               a_B_coh=None, Bturb=None, outer_scale=None,
                               a_norm=0., random_B_coh=False,
                               dn=True, hk=True, return_rates=False):
        """Propagate supernova neutrinos with nonzero magnetic moments
           through the Galactic magnetic fields, compute event rates
           at Earth, and compare to the rates expected for zero magnetic moment.
           If B field parameters (a_B_coh, Bturb, and outer_scale) are given,
           the magnetic field map is regenerated prior to the computation.

           Parameters:
               mu:           neutrino magnetic moment
               d:            distance to the neutrino source
               phi_los:      orientation of the line of sight along the Galactic plane.
                             0 corresponds to a l.o.s. pointing away from the Galactic Center
               theta_los:    orientation of the line of sight relative to the Galactic plane
               mh:           the neutrino mass ordering ('NH' or 'IH'), relevant
                             for propagating neutrinos out of the SN.
               a_B_coh:      nuisance parameter describing the shift in the strength
                             of thw homogeneous (large-scale) magnetic field relative
                             to the fiducial model from https://arxiv.org/abs/0704.0458
               Bturb:        field strength of turbulent magnetic field [muG]
               outer_scale:  outer scale of B-field turbulence in kpc
               a_norm:       relative flux normalization bias to apply before
                             computing chi^2.
                             If a_bias=None or 'minimize', minimize chi^2 over
                             this nuisance parameter
               random_B_coh: if True, pick random B fields for each of the regions
                             in the coherent B field model. Values are chosen from
                             a Gaussian of width 2 muG
               dn:           (bool) compute event rates at DUNE?
               hk:           (bool) compute event rates at HyperK?
               return_rates: return event rates in addition to chi^2?

           Return Value:
               the chi^2 resulting from the comparison of rates
               with and without magnetic moment

               if return_rates=True, also a dictionary containing the event rates
               at DUNE and HyperK is returned"""
        
        U = PMNS(theta12, theta13, theta23)    # PMNS matrix

        binning = np.linspace(-5e-3, 2e-2, 6)  # time bins [sec]
        
        # Emission data from simulation. Col 0: time in s, col 1:luminosity in 1e51 ergs,
        # col 2: average energy in MeV, col 3: alpha parameter"
        nu_e = np.loadtxt('sn-data/Sf/neutrino_signal_nu_e', usecols=(0,1,2,5)).T
        nubar_e = np.loadtxt('sn-data/Sf/neutrino_signal_nubar_e', usecols=(0,1,2,5)).T
        nu_x = np.loadtxt('sn-data/Sf/neutrino_signal_nu_x', usecols=(0,1,2,5)).T

        t      = nu_e[0]  # time series
        idx_in = np.where(t>=-5e-3)[0][0]
        idx_fn = np.where(t<=2e-2)[0][-1]

        # regenerate magnetic field map
        if d == None:
            d = self.d
        if theta_los == None:
            theta_los = self.theta_los
        if phi_los == None:
            phi_los = self.phi_los
        if (a_B_coh != None and Bturb != None and outer_scale != None) \
                or d != self.d or theta_los != self.theta_los or phi_los != self.phi_los:
            print("regenerating galactic B-fields.")
            self.generate_B_field_gal(d=d, theta_los=theta_los, phi_los=phi_los,
                                      a_B_coh=a_B_coh, Bturb=Bturb, outer_scale=outer_scale,
                                      random_B_coh=random_B_coh)
            self.d         = d
            self.theta_los = theta_los
            self.phi_los   = phi_los

        def spec(x, mean, alph):
            """Energy spectrum of SN neutrinos in MeV"""
            return (x**alph * np.exp(-(alph+1)*x/mean)
                    * ((1+alph)/mean)**(1+alph)/special.gamma(1+alph))

        def frac(mean, alph, sigma):
            """Integrate the flux times the cross section"""
            if mean == 0:
                return 0
            else:
                return integrate.quad(lambda x: spec(x, mean, alph)
                                              * self.sigma_extract(sigma, x),
                                      0, 100, args=(), limit=100)[0]

        # Number of emitted neutrinos (Luminosity/(Mean Energy * Total Surface))"
        factor = 1e51 * erg/(4*np.pi*d**2*1e6*eV)    # Conversion from erg to Mev/area

        N_e    = [nu_e[1][i] * factor/nu_e[2][i] if nu_e[2][i] != 0 else 0
                                                  for i in range(idx_in, idx_fn+1)]
        N_ebar = [nubar_e[1][i] * factor/nubar_e[2][i] if nubar_e[2][i] != 0 else 0
                                                  for i in range(idx_in, idx_fn+1)]
        N_x    = [nu_x[1][i] * factor/nu_x[2][i]  if nu_x[2][i] != 0 else 0
                                                  for i in range(idx_in, idx_fn+1)]

        # index gymnastics for the two mass hierarchies
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

        # event rate at DUNE
        if dn:
            # the structure of the following arrays is
            # (flavor, detection channels, time bins)
            frac_e_dn = np.array([[[frac(nu_e[2][i], nu_e[3][i], self.sigma['DN'][j][l])
                                    for i in range(idx_in, idx_fn+1)]  # time
                                    for l in range(2)]                 # channel
                                    for j in range(2)])                # flavor
            frac_bare_dn = np.array([[[frac(nubar_e[2][i], nubar_e[3][i], self.sigma['bar_DN'][j][l])
                                    for i in range(idx_in, idx_fn+1)]
                                    for l in range(2)]
                                    for j in range(2)])
            frac_x_dn = np.array([[[frac(nu_x[2][i], nu_x[3][i], self.sigma['DN'][j][l])
                                    for i in range(idx_in, idx_fn+1)]
                                    for l in range(2)]
                                    for j in range(2)])
            frac_barx_dn = np.array([[[frac(nu_x[2][i], nu_x[3][i], self.sigma['bar_DN'][j][l])
                                    for i in range(idx_in, idx_fn+1)]
                                    for l in range(2)]
                                    for j in range(2)])

            # fold with oscillation probabilities both inside the SN and outside,
            # including magnetic moments
            frac_emu_dn      = frac_e_dn * self.P_app(np.linalg.multi_dot(
                            [np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_e-1, i_e-1])
            frac_baremu_dn   = frac_bare_dn * self.P_app(np.linalg.multi_dot(
                            [np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_be-1, i_be-1])
            frac_mumu_dn     = frac_x_dn * self.P_app(np.linalg.multi_dot(
                            [np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_mu-1, i_mu-1])
            frac_barmumu_dn  = frac_barx_dn * self.P_app(np.linalg.multi_dot(
                            [np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_bmu-1, i_bmu-1])
            frac_taumu_dn    = frac_x_dn * self.P_app(np.linalg.multi_dot(
                            [np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_tau-1, i_tau-1])
            frac_bartaumu_dn = frac_barx_dn * self.P_app(np.linalg.multi_dot(
                            [np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_btau-1, i_btau-1])

            # Number of neutrinos interacting with the detector for times t
            # in absence of magnetic moments.
            # The structure of these arrays is (flavor, time bin)
            dec_dn = [[((U[j,i_mu-1]**2+U[j,i_tau-1]**2)
                        * (frac_x_dn[min(j,1)][0][i] * n_ar_DN
                         + frac_x_dn[min(j,1)][1][i] * n_e_DN) * N_x[i]
                       + U[j,i_e-1]**2 * (frac_e_dn[min(j,1)][0][i] * n_ar_DN
                                        + frac_e_dn[min(j,1)][1][i] * n_e_DN) * N_e[i])
                    * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)]
                                                  for j in range(3)]

            dec_bar_dn = [[(U[j,i_be-1]**2 * (frac_bare_dn[min(j,1)][0][i] * n_ar_DN
                                            + frac_bare_dn[min(j,1)][1][i] * n_e_DN) * N_ebar[i]
                         + (U[j,i_bmu-1]**2 + U[j,i_btau-1]**2)
                                       * (frac_barx_dn[min(j,1)][0][i] * n_ar_DN
                                        + frac_barx_dn[min(j,1)][1][i] * n_e_DN) * N_x[i])
                    * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)]
                                                  for j in range(3)]

            # Now the same with magnetic conversion included
            dec_mu_dn = [[((U[j,i_mu-1]**2 * (frac_mumu_dn[min(j,1)][0][i] * n_ar_DN
                                            + frac_mumu_dn[min(j,1)][1][i] * n_e_DN)
                          + U[j,i_tau-1]**2 * (frac_taumu_dn[min(j,1)][0][i] * n_ar_DN
                                             + frac_taumu_dn[min(j,1)][1][i] * n_e_DN)) * N_x[i]
                        + U[j,i_e-1]**2 * (frac_emu_dn[min(j,1)][0][i] * n_ar_DN
                                         + frac_emu_dn[min(j,1)][1][i] * n_e_DN) * N_e[i])
                        * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)]
                                                      for j in range(3)]

            dec_bar_mu_dn = [[(U[j,i_be-1]**2 * (frac_baremu_dn[min(j,1)][0][i] * n_ar_DN
                                               + frac_baremu_dn[min(j,1)][1][i] * n_e_DN) * N_ebar[i]
                            + (U[j,i_bmu-1]**2 * (frac_barmumu_dn[min(j,1)][0][i] * n_ar_DN
                                                + frac_barmumu_dn[min(j,1)][1][i] * n_e_DN)
                                + U[j,i_btau-1]**2 * (frac_bartaumu_dn[min(j,1)][0][i] * n_ar_DN
                                                    + frac_bartaumu_dn[min(j,1)][1][i] * n_e_DN)) * N_x[i])
                            * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)]
                                                          for j in range(3)]

            # Bins
            bins_dn = np.array([[sum(dec_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                          :np.where(t<=binning[i+1])[0][-1]-idx_in])
                                   for i in range(5)]       # time
                                   for j in range(3)])      # flavor

            bins_bar_dn = np.array([[sum(dec_bar_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                                  :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                  for j in range(3)])

            bins_mu_dn = np.array([[sum(dec_mu_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                                :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                for j in range(3)])

            bins_bar_mu_dn = np.array([[sum(dec_bar_mu_dn[j][np.where(t>=binning[i])[0][0]-idx_in
                                                        :np.where(t<=binning[i+1])[0][-1]-idx_in]) for i in range(5)]
                                    for j in range(3)])

        else:
            bins_dn = np.zeros((3,6))
            bins_bar_dn = np.zeros((3,6))
            bins_mu_dn = np.zeros((3,6))
            bins_bar_mu_dn = np.zeros((3,6))

        # event rates at HyperKamiokande
        if hk:
            frac_e_hk = np.array([[[frac(nu_e[2][i], nu_e[3][i], self.sigma['HK'][j][l]) for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)])
            frac_bare_hk = np.array([[[frac(nubar_e[2][i], nubar_e[3][i], self.sigma['bar_HK'][j][l]) for i in range(idx_in, idx_fn+1)] for l in range(3)] for j in range(2)])
            frac_x_hk = np.array([[[frac(nu_x[2][i], nu_x[3][i], self.sigma['HK'][j][l]) for i in range(idx_in, idx_fn+1)] for l in range(2)] for j in range(2)])
            frac_barx_hk = np.array([[[frac(nu_x[2][i], nu_x[3][i], self.sigma['bar_HK'][j][l]) for i in range(idx_in, idx_fn+1)]
                          for l in range(3)] for j in range(2)])

            frac_emu_hk = frac_e_hk * self.P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_e-1, i_e-1])
            frac_baremu_hk = frac_bare_hk * self.P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_be-1, i_be-1])
            frac_mumu_hk = frac_x_hk * self.P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_mu-1, i_mu-1])
            frac_barmumu_hk = frac_barx_hk * self.P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_bmu-1, i_bmu-1])
            frac_taumu_hk = frac_x_hk * self.P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_tau-1, i_tau-1])
            frac_bartaumu_hk = frac_barx_hk * self.P_app(np.linalg.multi_dot([np.transpose(U[:3,:3]), mu, U[:3,:3]])[i_btau-1, i_btau-1])

            # Number of neutrinos interacting with the detector for times t
            dec = [[((U[j,i_mu-1]**2+U[j,i_tau-1]**2) * (frac_x_hk[min(j,1)][0][i] * n_e_HK + frac_x_hk[min(j,1)][1][i] * n_O_HK) * N_x[i]
                      + U[j,i_e-1]**2 * (frac_e_hk[min(j,1)][0][i] * n_e_HK + frac_e_hk[min(j,1)][1][i] * n_O_HK) * N_e[i])
                    * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]

            dec_bar = [[(U[j,i_be-1]**2 * ((frac_bare_hk[min(j,1)][0][i] + frac_bare_hk[min(j,1)][2][i]) * n_e_HK
                                           + frac_bare_hk[min(j,1)][1][i] * n_O_HK) * N_ebar[i]
                          + (U[j,i_bmu-1]**2 + U[j,i_btau-1]**2) * ((frac_barx_hk[min(j,1)][0][i] + frac_barx_hk[min(j,1)][2][i]) * n_e_HK
                                                                    + frac_barx_hk[min(j,1)][1][i] * n_O_HK) * N_x[i])
                        * (t[i+1+idx_in]-t[i+idx_in]) for i in range(idx_fn+1-idx_in)] for j in range(3)]

            # With magnetic conversion
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

            # Bins
            bins_hk = np.array([[sum(dec[j][np.where(t>=binning[i])[0][0]-idx_in
                                           :np.where(t<=binning[i+1])[0][-1]-idx_in])
                              for i in range(5)] for j in range(3)])

            bins_bar_hk = np.array([[sum(dec_bar[j][np.where(t>=binning[i])[0][0]-idx_in
                                                   :np.where(t<=binning[i+1])[0][-1]-idx_in])
                              for i in range(5)] for j in range(3)])

            bins_mu_hk = np.array([[sum(dec_mu[j][np.where(t>=binning[i])[0][0]-idx_in
                                                :np.where(t<=binning[i+1])[0][-1]-idx_in])
                              for i in range(5)] for j in range(3)])

            bins_bar_mu_hk = np.array([[sum(dec_bar_mu[j][np.where(t>=binning[i])[0][0]-idx_in
                                                        :np.where(t<=binning[i+1])[0][-1]-idx_in])
                              for i in range(5)] for j in range(3)])

        else:
            bins_hk = np.zeros((3,6))
            bins_bar_hk = np.zeros((3,6))
            bins_mu_hk = np.zeros((3,6))
            bins_bar_mu_hk = np.zeros((3,6))

        # chi^2 computation
        def chi_temp(a):
            c = a**2/0.1**2
            if hk:
                c += np.sum((np.sum(bins_hk + bins_bar_hk, axis=0)[:-1]
                           - (1+a)*np.sum(bins_mu_hk + bins_bar_mu_hk, axis=0)[:-1])**2
                           / ((1+a)*np.sum(bins_mu_hk + bins_bar_mu_hk, axis=0)[:-1]))
                    # FIXME why is the last time bin removed?
            if dn:
                c += np.sum((np.sum(bins_dn + bins_bar_dn, axis=0)[:-1]
                           - (1+a)*np.sum(bins_mu_dn + bins_bar_mu_dn, axis=0)[:-1])**2
                           / ((1+a)*np.sum(bins_mu_dn + bins_bar_mu_dn, axis=0)[:-1]))
            return c
        
        if a_norm == None or a_norm == 'minimize' or a_norm == 'min':
            chi = optimize.minimize(chi_temp, 0).fun
        else:
            chi             = chi_temp(a_norm)
            bins_dn        *= 1+a_norm
            bins_bar_dn    *= 1+a_norm
            bins_mu_dn     *= 1+a_norm
            bins_bar_mu_dn *= 1+a_norm
            bins_hk        *= 1+a_norm
            bins_bar_hk    *= 1+a_norm
            bins_mu_hk     *= 1+a_norm
            bins_bar_mu_hk *= 1+a_norm

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


