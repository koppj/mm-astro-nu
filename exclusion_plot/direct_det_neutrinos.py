import scipy, os, subprocess, shutil # scipy for integrating, the rest are for calling programs outside of python, changing directories ect
import numpy as np                   # used for dealing with arrays
import matplotlib.pyplot as plt      # plotting package
from Header import *                 # this contains all my ploting options for matplotlib
import pandas as pd                  # reading in all data using pandas to make life easy
import pickle                        # for importing and saving the interpolated muon width grid
# set the default printing option so I don't
# have to count the number of digits all the time
# Unit conversion and physical constants



class my_units:
    # Energy and mass
    eV     = 1.
    keV    = 1.e3
    MeV    = 1.e6
    GeV    = 1.e9
    TeV    = 1.e12
    PeV    = 1.e15
    kg     = 5.609588603e35*eV   # (exact)
    grams  = 0.001*kg
    tons   = 1000*kg
    Kelvin = 8.617333262e-5                  # PDG 2019
    Joule  = 1/1.602176634e-19               # PDG 2019 (exact)
    erg    = 1e-7*Joule
    u_atom  =  931.49410242 * MeV            # atomic mass PDG 2019
    
    # Length and time
    m     = 1/197.3269804e-9    # (exact)
    meter = m
    km    = 1000*m
    cm    = 0.01*m
    nm    = 1.e-9*m
    fm    = 1.e-15*m
    barn  = 1.e-28*m**2
    AU    = 1.4960e11*m
    pc    = 30.857e15*m
    kpc   = 1.e3*pc
    Mpc   = 1.e6*pc
    Gpc   = 1.e9*pc
    ly    = 9460730472580800*m  # light year (exact)
    sec   = 299792458*meter     # (exact)
    hours = 3600*sec
    days  = 24*hours
    yrs   = 365*days
    Hz    = 1./sec
    kHz   = 1.e3*Hz
    MHz   = 1.e6*Hz
    GHz   = 1.e9*Hz
    
    # particle physics
    alpha_em = (1./137.035999139)            # electromagnetic fine structure constant (PDG 2018)
    m_e      = 0.5109989461 * MeV            # electron mass (PDG 2018)
    m_mu     = 105.6583745 * MeV             # muon mass (Wikipedia 28.10.2019)
    m_n      = 939.5654133 * MeV             # neutron mass (Wikipedia 29.10.2019)
    m_p      = 938.2720813 * MeV             # proton mass (Wikipedia 29.10.2019)
    tau_mu   = 2.1969811e-6 * sec            # muon lifetime (PDG 2019)
    GF       = 1.1663787e-5 / GeV**2         # Fermi constant (PDG 2019)
    sw2      = 0.23121                       # sin^2 theta_w (M_Z) (PDG 2019)
    sw       = np.sqrt(sw2)                  # sin theta_W (M_Z) (PDG 2019)

    # Various astrophysical constants
    GN    = 6.708e-39/1e18  # eV^-2, Newton's constant
    MPl   = 1.22093e19*GeV   # Planck mass, PDG 2013
    Msun  = 1.989e30*kg
    Rsun  = 6.9551e8*meter
    
    # atomic physics
    a0      = 1. / (m_e * alpha_em)          # Bohr radius
    Ry      = 0.5 * m_e * alpha_em**2        # Rydberg constant
    
    # cosmology
    h       = 0.6766                         # (Planck 2018)
    H0      = h * 100. * km / sec/ Mpc       # Hubble parameter
    rho_c0  = 3. * H0**2/(8. * np.pi * GN)   # critical density today, Kolb Turner eq. (3.14)
    Omega_m = 0.14240 / h**2                 # total matter density (Placnk 2018)
    Omega_Lambda = 0.6889 / h**2             # dark energy density (Planck 2018)
    
    # xenon constants 
    W    = 13.7     # 13.7 eV/quantum
    g1   = 0.14261  # conversion factors for 
    g2   = 11.55    # S1+S2 -> energy in Xenon1T
    A_Xe = 131.293  # average atomic number of xenon
    # taken from http://www.physics.uwo.ca/~lgonchar/courses/p9826/xdb.pdf
    binding_energies = np.array([34561., 5453., 5107., 4786., 1148.7, 1002.1, 940.6, 
                                 689.0, 676.4, 213.2, 146.7, 145.5, 69.5, 67.5, 23.3, 13.4, 12.1]) * eV 
    # from wikipedia (seems to check out based on my foggy chemisty knowledge)
    electron_multiplicites = np.array([2,2,2,4,2,2,4,4,6,2,2,4,4,6,2,2,4])
    
    muB = np.sqrt(4.*np.pi*alpha_em)/(2.*m_e)  # Bohr magneton 
    N_A = 6.02214076E+23                       # Avogadro's number

u = my_units()   


def target_material(material_string) :
    # Read in data
    elem_data = pd.read_csv('data/element_data.dat', delimiter='\t')
    
    # pick relevant element
    row = elem_data.loc[elem_data['element'] == material_string]
    
    # set the results as global variables
    global Z, A, N, effA, mNucl
    Z     = float(row["Z"])
    A     = float(row["A"])
    N     = A - Z
    effA  = float(row["effA"])
    mNucl = A * u.u_atom

nucl_target_material = 'Xe'
if nucl_target_material == 'Ge' :
    target_material('Ge')

class xsections_and_rates :

    # load units and other constants
    u = my_units()

    isotope_data = pd.read_csv('data/Xe_isotopes.csv', delimiter=',')
    
    def diff_sigma_mag_mom(args, Er = 0., Enu = 0.) :
        '''
        Output:
        Active neutrino-electron scattering differential cross section through 
        the neutrin magnetic moment [eV^2]
        
        Input:
        args is a dictionary containing:
        d                =  magnetic neutrio dipole [eV^-1] (following conventions in 1803.03262)
        mN               =  heavy neutrino mass [eV]
        nucl_or_electron =  0 = electron only, 1 = nucl only
        other paramters:
        Er               =  electron recoil energy [eV]
        Enu              =  incoming neutrino energy [eV]
        
        
        Notes:
        Cross-section taken from 1803.03262 Eq. (A.6)
        Mathematica notebook "/to/recoil_xsec.nb" containts
        details of the basic kinematics
        '''

        d                = args['d']
        mN               = args['mN']
        nucl_or_electron = args['nucl_or_electron']

        if nucl_or_electron == 1 and nucl_target_material == 'Xe' :
            isotope_name = args['isotope_name']
            isotope_row  = xsections_and_rates.isotope_data.loc[xsections_and_rates.isotope_data['isotope'] == isotope_name]
            
            global Z, N, A, mNucl

            Z     = float(isotope_row['Z'])
            N     = float(isotope_row['N'])
            A     = Z + N
            spin  = float(isotope_row['spin'])
            mNucl = (Z + N) * u.u_atom
        if nucl_or_electron == 1 and nucl_target_material == 'Ge' :
            spin = 0

        
        res = 0.

        # Er_max = ( 1./(2*Enu + u.m_e) * ( Enu**2 - 0.5*mN**2 
        #     + Enu/2.*u.m_e * (np.sqrt(mN**4 - 4.*mN**2*u.m_e*(Enu+u.m_e) + 4.*Enu**2*u.m_e**2) - mN**2) ) )

        # if Er_max < Er :
        #     return 0.

        
        def form_factor(Er) :
            ### ------
            ### See text below Eq. 3 and Eq. 9 in 1202.6073
            ### ------
            s     = 1. * u.fm
            R     = 1.2 * A**(1./3.) * u.fm
            r     = np.sqrt(R**2. - 5. * s**2)
            kappa = np.sqrt(2. * mNucl * Er)

            return 3. * np.exp(-1. * kappa**2 * s**2 / 2.) * ( np.sin(kappa * r) - kappa * r * np.cos(kappa * r) ) / (kappa * r)**3
        
        if nucl_or_electron == 0 :
            res = ( d**2 * u.alpha_em *
                   (8. * Enu * Er * u.m_e**2 * (Enu - Er) 
                    + 2. * Er * mN**2 * u.m_e * (Er - 2. * Enu - u.m_e)
                    + mN**4 * (Er - u.m_e) )
                    / (2. * Enu**2 * Er**2 * u.m_e**2.) )
            res *= (res>0)

        if nucl_or_electron == 1 :
            pre_factor = Z**2. * form_factor(Er)**2.
            
            res = ( pre_factor * d**2 * u.alpha_em *
                   (8. * Enu * Er * mNucl**2 * (Enu - Er) 
                    + 2. * Er * mN**2 * mNucl * (Er - 2. * Enu - mNucl)
                    + mN**4 * (Er - mNucl) )
                    / (2. * Enu**2 * Er**2 * mNucl**2.) )
            
        return res
    
    
    def diff_sigma_SM(args, Er = 0., Enu = 0.) :
        '''
        Output:
        Active SM neutrino-electron scattering differential cross section including both 
        electron and nuclear recoils [eV^2]
        
        Input:
        args is a dictionary containing:
        mN               =  heavy neutrino mass [eV]
        flavour          =  which neutrino flavour (0,1,2) = (e,mu,tau) flavours
        nucl_or_electron =  0 = electron only, 1 = nucl only
        other parameters:
        Er               =  electron recoil energy [eV]
        Enu              =  incoming neutrino energy [eV]
        
        
        Notes:
        Cross-section taken from paper (see references therein)
        '''

        flavour          = args['flavour']
        nucl_or_electron = args['nucl_or_electron']
        if nucl_or_electron == 1 and nucl_target_material == 'Xe' :
            isotope_name = args['isotope_name']
            isotope_row  = xsections_and_rates.isotope_data.loc[xsections_and_rates.isotope_data['isotope'] == isotope_name]
            
            global Z, N, A, mNucl

            Z     = float(isotope_row['Z'])
            N     = float(isotope_row['N'])
            A     = Z + N
            spin  = float(isotope_row['spin'])
            mNucl = (Z + N) * u.u_atom
        elif nucl_or_electron == 1 and nucl_target_material == 'Ge' :
            spin  = 0.



        
        def form_factor(Er) :
            ### ------
            ### See text below Eq. 3 and Eq. 9 in 1202.6073
            ### ------
            s     = 1. * u.fm
            R     = 1.2 * A**(1./3.) * u.fm
            r     = np.sqrt(R**2. - 5. * s**2)
            kappa = np.sqrt(2. * mNucl * Er)

            return 3. * np.exp(-1. * kappa**2 * s**2 / 2.) * ( np.sin(kappa * r) - kappa * r * np.cos(kappa * r) ) / (kappa * r)**3
    
        res = 0.
        if nucl_or_electron == 0 :
            if flavour == 0 :
                # nu_e-electron x-sec
                res = (u.GF**2. * u.m_e / (2. * np.pi * Enu**2) *
                       (4.*u.sw**4. * (2.*Enu**2. + Er**2. - Er * (2.*Enu + u.m_e) )
                        -2.*u.sw**2. * (Er*u.m_e - 2.*Enu**2.) + Enu**2 ) )
                       
            else :
                # nu_mu/nu_tau-electron x-sec
                res = (u.GF**2. * u.m_e / (2. * np.pi * Enu**2) *
                       (4.*u.sw**4. * (2.*Enu**2. + Er**2. - Er * (2.*Enu + u.m_e) )
                        +2.*u.sw**2. * (Er*u.m_e - 2.*Enu**2.) + Enu**2 ) )
        
        if nucl_or_electron == 1 :
            # nu_i-nuclear x-sec      
            Qw  = (2. * Z - A) - 4. * Z * u.sw**2. 
            res = (u.GF**2. * mNucl * form_factor(Er)**2. * Qw**2. / (8. * np.pi * Enu**2 ) *
                (2. * Enu**2 - 2 * Enu * Er - Er * mNucl) )
        
        return res
            
        
    def Enu_min(Er,mN,nucl_or_electron = 0) :
        '''
        Output:
        Minimum neutrino energy for a given electron recoil energy [eV]
        
        Input:
        Er               =  electron recoil energy [eV]
        mN               =  heavy neutrino masses [eV]
        nucl_or_electron =  0 = electron only, 1 = nucl only
        
        Notes:
        Mathematica notebook "/to/recoil_xsec.nb" contains analytical 
        expressions based on equations from 1810.03626 (Vedran's paper)
        '''
        if nucl_or_electron == 0 :
            mTarget = u.m_e 
        else :
            mTarget = mNucl
            
        Tmin_relation = ( (mN**2 + 2. * mTarget * Er) /
               (2. * np.sqrt( Er * (2. * mTarget + Er) ) - 2. * Er ) )
        mN_threshold = ( mN + mN**2./ ( 2. * mTarget ) )
        
        return np.max([Tmin_relation, mN_threshold])
    
    
    def load_solar_spectra() :
        '''
        Output: 
        arg1  =   interpolated total solar flux [eV^3] (energy argument also in eV)
        arg2  =   solar line energies [eV]
        arg3  =   normalization of the solar lines
        
        Input:
        no input (reads data files from "solar-nu/")
        
        Notes:
        Final block determining solar neutrino fluxes -- taken from
        J. Kopp's code used in the muon diffusion project.

        Spectra of continuous solar neutrino fluxes from http://www.sns.ias.edu/~jnb/
        '''
        solar_spectrum_files = [ "solar-nu/" + f
                                   for f in ["n13.dat", "f17.dat", "o15.dat", "hepspectrum.dat", 
                                             "b8spectrum.dat", "ppspectrum.dat"] ];
        solar_spectrum_labels = [ r'${}^{13}$N', r'${}^{17}$F', r'${}^{15}$O',
                                  r'hep', r'${}^{8}$B', 'pp' ]
        solar_spectrum = [ np.loadtxt(f) for f in solar_spectrum_files ]
        for x in solar_spectrum:
            x[:,0] *= u.MeV
            x[:,1] *= 1/u.MeV

     # corresponding normalizations from https://arxiv.org/abs/astro-ph/0412440, Table 2, BS05(AGS,OP) model
        solar_spectrum_norm  = np.array([ 2.01e8,         # N-13
                                          3.25e6,         # F-17
                                          1.45e8,         # O-15
                                          8.25e3,         # hep
                                          4.51e6,         # B-8
                                          6.06e10         # pp
                                        ]) / (u.cm**2 * u.sec)


        solar_spectrum_interp = [ scipy.interpolate.interp1d(s[:,0], norm * s[:,1], kind = 'linear', bounds_error=False, fill_value=0.)
                                      for s, norm in zip(solar_spectrum, solar_spectrum_norm) ]

        # energies of discrete solar neutrino lines
        solar_line_energies = np.array([ 0.8618,  # Be-7 (ground state), http://www.sns.ias.edu/~jnb/SNdata/7belineshape.html
                                         0.3843,  # Be-7 (excited state)
                                         1.44     # pep, https://en.wikipedia.org/wiki/Proton%E2%80%93proton_chain_reaction
                                       ]) * u.MeV
        solar_line_norms    = np.array([0.897 * 4.34e9, # corresponding normalization,
                                        0.103 * 4.34e9, #  from https://arxiv.org/abs/astro-ph/0412440, Table 2, BS05(AGS,OP) model
                                                 1.45e8
                                       ]) / (u.cm**2 * u.sec)
        solar_line_labels   = [ r'${}^{7}$Be (g.s.)', r'${}^{7}$Be (exc.s.)', 'hep']
        
        return [solar_spectrum_interp, solar_line_energies, solar_line_norms]

    def load_solar_spectra_borexino_fit() :
        '''
        Output: 
        arg1  =   interpolated total solar flux [eV^3] (energy argument also in eV)
        arg2  =   solar line energies [eV]
        arg3  =   normalization of the solar lines
        
        Input:
        no input (reads data files from "solar-nu/")
        
        Notes:
        Final block determining solar neutrino fluxes -- taken from
        J. Kopp's code used in the muon diffusion project.

        Spectra of continuous solar neutrino fluxes from http://www.sns.ias.edu/~jnb/
        '''
        solar_spectrum_files = [ "solar-nu/" + f
                                   for f in ["n13.dat", "f17.dat", "o15.dat", "hepspectrum.dat", 
                                             "b8spectrum.dat", "ppspectrum.dat"] ];
        solar_spectrum_labels = [ r'${}^{13}$N', r'${}^{17}$F', r'${}^{15}$O',
                                  r'hep', r'${}^{8}$B', 'pp' ]
        solar_spectrum = [ np.loadtxt(f) for f in solar_spectrum_files ]
        for x in solar_spectrum:
            x[:,0] *= u.MeV
            x[:,1] *= 1/u.MeV

 		# Borexino Best fit point
        solar_spectrum_norm  = np.array([ 2.01e8,         # N-13
                                          3.25e6,         # F-17
                                          1.45e8,         # O-15
                                          8.25e3,         # hep
                                          5.68e6,         # B-8
                                          6.1e10         # pp
                                        ]) / (u.cm**2 * u.sec)


        solar_spectrum_interp = [ scipy.interpolate.interp1d(s[:,0], norm * s[:,1], kind = 'linear', bounds_error=False, fill_value=0.)
                                      for s, norm in zip(solar_spectrum, solar_spectrum_norm) ]

        # energies of discrete solar neutrino lines
        solar_line_energies = np.array([ 0.8618,  # Be-7 (ground state), http://www.sns.ias.edu/~jnb/SNdata/7belineshape.html
                                         0.3843,  # Be-7 (excited state)
                                         1.44     # pep, https://en.wikipedia.org/wiki/Proton%E2%80%93proton_chain_reaction
                                       ]) * u.MeV
        # Borexino best fit point
        solar_line_norms    = np.array([ 0.897 * 4.99e9, # corresponding normalization,
                                         0.103 * 4.99e9, #  from https://arxiv.org/abs/astro-ph/0412440, Table 2, BS05(AGS,OP) model
                                                 1.27e8
                                       ]) / (u.cm**2 * u.sec)
        solar_line_labels   = [ r'${}^{7}$Be (g.s.)', r'${}^{7}$Be (exc.s.)', 'hep']
        
        return [solar_spectrum_interp, solar_line_energies, solar_line_norms]
    
    # load the above function for the following interpolations
    solar_spectrum_interp, solar_line_energies, solar_line_norms = load_solar_spectra()
    solar_spectrum_interp_borexino, solar_line_energies_borexino, solar_line_norms_borexino = load_solar_spectra_borexino_fit()
    

    def load_DSRN_spectra() :
        '''
        Output: 
        Diffuse supernova relic neutrino background [eV^3]
        This is a vector for the flavours [e, mu, tau] where each 
        argument is a 1D interpolating function over the range
        [1,100] MeV.

        Notes:
        taken from J. Kopp's code used in the muon diffusion project.
        '''
        def phi_sn(f, E):
            '''supernova neutrino spectrum for flavor f and energy E
               https://arxiv.org/abs/hep-ph/0408031, eq. (2.4)'''
            dsnb_T    = { 'e':    3.5*u.MeV,   # spectrum temperature
                          'ebar': 5*u.MeV,
                          'x':    8*u.MeV }
            dsnb_eta  = { 'e':    2,           # pinching parameter
                          'ebar': 2,
                          'x':    1 }

            T   = dsnb_T[f]
            eta = dsnb_eta[f]
            return 1 / T**4 * E**2 / (np.exp(E/T - eta) + 1)

        # normalization factor k_alpha from eq. 2.4
        sn_norm = 3e53*u.erg / integ.quad(lambda E: E * (phi_sn('e', E) + phi_sn('ebar', E) + 4*phi_sn('x', E)),
                                          0, 100*u.MeV)[0]

        def R_SN(z):
            '''supernova rate at redshift z
               (https://arxiv.org/abs/hep-ph/0408031, eq. (2.3))'''
            R0    = 2e-4 / u.yrs / u.Mpc**3
            beta  = 2.5
            alpha = 1.
            if z <= 1:
                return R0 * (1+z)**beta
            else:
                return R0 * 2**(beta - alpha) * (1+z)**alpha
            
        def Hubble(z):
            '''Hubble parameter at redshift z
               (https://arxiv.org/abs/hep-ph/0408031, eq. (2.2))'''
            H0           = 70 * u.km / u.sec / u.Mpc
            Omega_m      = 0.3
            Omega_Lambda = 0.7
            return H0 * np.sqrt( Omega_m*(1+z)**3 + Omega_Lambda )
            
        def phi_dsnb_unosc(f, E):
            '''unoscillated discrete supernova neutrino background at flavor f and energy E
               (https://arxiv.org/abs/hep-ph/0408031, eq. (2.1))'''
            return integ.quad(lambda z: R_SN(z) * sn_norm * phi_sn(f, E*(1+z)) / Hubble(z),
                              0, 5)[0]

        def phi_dsnb_osc(f, E):
            '''discrete supernova neutrino background at flavor f and energy E
               including neutrino oscillations on the way out of the supernova.
               (https://arxiv.org/abs/hep-ph/0408031, eq. (2.1) and Table I)'''
            
            # mixing angles (NuFit 4.1)
            th12 = 33.82 * np.pi/180.
            th13 =  8.60 * np.pi/180.
            th23 = 48.6  * np.pi/180.
            
            # construct leptonic mixing matrix
            U = trafo.Rotation.from_euler('x', -th23).as_matrix() @ \
                trafo.Rotation.from_euler('y',  th13).as_matrix() @ \
                trafo.Rotation.from_euler('z', -th12).as_matrix()
            
            # note: at very high density, for normal mass ordering:
            #   \nu_e=\nu_3,         \nu_\mu=\nu_1,         \nu_\tau=\nu_2
            #   \bar\nu_e=\bar\nu_1, \bar\nu_\mu=\bar\nu_2, \bar\nu_\tau=\bar\nu_3
            if f == 'e':
                return np.abs(U[0,2])**2 * phi_dsnb_unosc('e', E) \
                    + (np.abs(U[0,0])**2 + np.abs(U[0,1])**2) * phi_dsnb_unosc('x', E)
            elif f == 'ebar':
                return np.abs(U[0,0])**2 * phi_dsnb_unosc('e', E) \
                    + (np.abs(U[0,1])**2 + np.abs(U[0,2])**2) * phi_dsnb_unosc('x', E)
            elif f == 'x':
                return (np.abs(U[1,2])**2 + np.abs(U[2,2])**2) * phi_dsnb_unosc('e', E) \
                     + (np.abs(U[1,0])**2 + np.abs(U[2,0])**2) * phi_dsnb_unosc('ebar', E) \
                     + (np.abs(U[1,0])**2 + np.abs(U[2,0])**2 + np.abs(U[1,1])**2 + np.abs(U[2,1])**2 \
                     + np.abs(U[1,1])**2 + np.abs(U[2,1])**2 + np.abs(U[1,2])**2 + np.abs(U[2,2])**2) \
                     * phi_dsnb_unosc('x', E)
            else:
                raise ValueError("invalid flavor: {:s}".format(f))

            E_range = np.linspace(1, 100, 50) * u.MeV
            flavours = ['e','x','x']

            DSRN_spectrum_interp = [ scipy.interpolate.interp1d(E_range,  phi_dsnb_osc(f,E_range), kind = 'linear', bounds_error=False, fill_value=0.)
                                      for f in flavours ]

            return DSRN_spectrum_interp

    def load_atmospheric_spectra() :
        '''
        Output: 
        Atmospheric neutrino flux [eV^3]

        Notes:
        taken from J. Kopp's code used in the muon diffusion project.
        Based on the Battistoni fluxes below 100 MeV
        '''
        # atm_spectrum_file   =  'data/battistoni.dat'

        # atm_spectrum        = np.loadtxt(atm_spectrum_file)
        # atm_spectrum[:,0]  *= u.GeV
        # atm_spectrum[:,1:] *= 1 / (u.meter**2 * u.sec * u.GeV)

        # atm_flavors         = ['mu','mubar','e','ebar']
        # atm_spectrum_interp = { f: interp.interp1d(atm_spectrum[:,0], atm_spectrum[:,j], bounds_error=False, fill_value=0.)
        #                               for j, f in zip(range(1,5), atm_flavors) }

        # # plot
        # E_range = np.linspace(10,1000,100) * u.MeV
        # #E_range = np.logspace(1,2,50) * u.MeV
        # for f in atm_flavors:
        #     plt.plot(E_range/u.MeV, atm_spectrum_interp[f](E_range) * (u.meter**2 * u.sec * u.GeV),
        #              label=f)
        atm_nu_data = pd.read_csv('data/atm_nu_sky_avg.dat', delimiter=' ', skiprows = 2, names=['Enu','numu','barnumu','nue','barnue'])

        atm_nu_data['Enu'] *= u.GeV

        # print(np.max(atm_nu_data['Enu'])/u.MeV)

        for key in ['numu','barnumu','nue','barnue'] :
            atm_nu_data[key] *= 4. * np.pi / (u.m**2 * u.sec * u.GeV) 
            # * np.sin(4.*np.pi*np.pi/360.)**2. FIXME

        return scipy.interpolate.interp1d(atm_nu_data['Enu'],  atm_nu_data['numu'], kind = 'linear', bounds_error=False, fill_value=0.)
                    

    atm_neutrino_flux = load_atmospheric_spectra()

    def osc_prob() :
        ''' 
        Output: 
        Probability of nu_i in solar neutrino flux 
        (be careful this assumes adiabaticity)
        valid in energy range 1E-3 - 1E+2 MeV
        
        Notes: 
        Based on J. Kopp's mathematica notebook "xenon.nb"
        '''
        Pee = pd.read_csv('data/Pee.dat', delimiter='\t', skiprows = 1, names=['energy','probability'])
        Pem = pd.read_csv('data/Pem.dat', delimiter='\t', skiprows = 1, names=['energy','probability'])
        Pet = pd.read_csv('data/Pet.dat', delimiter='\t', skiprows = 1, names=['energy','probability'])
        
        Pee_interp = scipy.interpolate.interp1d(Pee['energy'],Pee['probability'], fill_value = 0., bounds_error = False)
        Pem_interp = scipy.interpolate.interp1d(Pem['energy'],Pem['probability'], fill_value = 0., bounds_error = False)
        Pet_interp = scipy.interpolate.interp1d(Pet['energy'],Pet['probability'], fill_value = 0., bounds_error = False)
        
        return np.array([Pee_interp, Pem_interp, Pet_interp]) 
    
    osc_prob_interp = osc_prob()
        
    
    def total_solar_spec(Enu_vector) :
        '''
        Output:
        Vector of the total solar neutrino flux [eV^3]
        
        Input: 
        Enu_vector = Vector of Enu energies [eV]
        
        Notes:
        This just sums the continuous solar neutrino sources with a vector 
        of energies as the input.
        '''
        return np.sum([s(Enu_vector) for s in xsections_and_rates.solar_spectrum_interp], axis=0)
    
    def event_rate(Er_vec, d, mN,
                   exposure = u.tons * u.yrs, 
                   flux = total_solar_spec,
                   flux_line_energies = solar_line_energies, 
                   flux_line_norms = solar_line_norms,
                   cross_section = diff_sigma_mag_mom, 
                   nucl_or_electron = 0,
                   flavour = 1,
                   include_binding_energies = True) :
                   
        xsec_args = {}
        xsec_args['d']                = d
        xsec_args['mN']               = mN
        xsec_args['nucl_or_electron'] = nucl_or_electron
        xsec_args['flavour']          = flavour
                   
        def integrate_flux(Er) :
            # integrate over the entire continuous solar neutrino flux
            log_Enu_min = np.log10(xsections_and_rates.Enu_min(Er,mN, nucl_or_electron = nucl_or_electron)/u.MeV) 
            Enu_vals = np.logspace(log_Enu_min,2.,100) * u.MeV

            # Use quad rather than trapz
            res = np.trapz(
                    xsections_and_rates.osc_prob_interp[xsec_args['flavour']](Enu_vals) 
                          * cross_section(xsec_args, Er = Er, Enu = Enu_vals) 
                          * flux(Enu_vals),
                x = Enu_vals)

            return res
        
        def integrate_flux_lines(Er) :
            # Sum over the three monochromatic solar neutrino lines
            # select out the lines that have high enough neutrino energies
            line_indicies = np.where(flux_line_energies > xsections_and_rates.Enu_min(Er,mN, nucl_or_electron = nucl_or_electron))[0]

            res =np.sum(
                np.array([ \
                    xsections_and_rates.osc_prob_interp[xsec_args['flavour']](flux_line_energies[i]) \
                    * cross_section(xsec_args, Er = Er, Enu = flux_line_energies[i]) \
                    * flux_line_norms[i]  \
                    for i in line_indicies ]) )

            return res
            
        if include_binding_energies == True :
            integration =  np.array([ \
                np.sum(np.array([  \
                    u.electron_multiplicites[i] * np.heaviside( Er - u.binding_energies[i], 1.) \
                    * (integrate_flux(Er) + integrate_flux_lines(Er)) for i in range(0,len(u.binding_energies))])) for Er in Er_vec])

            prefactor =  exposure * u.keV
        else :
            integration =  np.array([ 
                (integrate_flux(Er) + integrate_flux_lines(Er) ) for Er in Er_vec])
                # (integrate_flux(Er)  ) for Er in Er_vec])

            prefactor =  exposure * u.keV

        return integration * prefactor

    def nuclear_event_rate(Er_vec, d, mN,
                   exposure = u.tons * u.yrs, 
                   flux = total_solar_spec,
                   cross_section = diff_sigma_mag_mom, 
                   nucl_or_electron = 1,
                   flavour = 1) :
                   
        xsec_args = {}
        xsec_args['d']                = d
        xsec_args['mN']               = mN
        xsec_args['nucl_or_electron'] = nucl_or_electron
        xsec_args['flavour']          = flavour
                   
        def integrate_flux(Er) :
            # integrate over the entire continuous solar neutrino flux
            log_Enu_min = np.log10(xsections_and_rates.Enu_min(Er,mN, nucl_or_electron = nucl_or_electron)/u.MeV) 
            Enu_vals = np.logspace(log_Enu_min,2.,100) * u.MeV

            # Use quad rather than trapz
            res = np.trapz(
                    xsections_and_rates.osc_prob_interp[xsec_args['flavour']](Enu_vals) 
                          * cross_section(xsec_args, Er = Er, Enu = Enu_vals) 
                          * flux(Enu_vals),
                x = Enu_vals)

            return res
        
        def integrate_flux_lines(Er) :
            # Sum over the three monochromatic solar neutrino lines
            # select out the lines that have high enough neutrino energies
            line_indicies = np.where(xsections_and_rates.solar_line_energies > xsections_and_rates.Enu_min(Er,mN, nucl_or_electron = nucl_or_electron))[0]
            return np.sum(
                np.array([ \
                    xsections_and_rates.osc_prob_interp[xsec_args['flavour']](xsections_and_rates.solar_line_energies[i]) \
                    * cross_section(xsec_args, Er = Er, Enu = xsections_and_rates.solar_line_energies[i]) \
                    * xsections_and_rates.solar_line_norms[i]  \
                    for i in line_indicies ]) )
            
        res_array = []

        if nucl_target_material == 'Xe' :
            for isotope_name in xsections_and_rates.isotope_data['isotope'] :

                xsec_args['isotope_name'] = isotope_name

                isotope_row = xsections_and_rates.isotope_data.loc[xsections_and_rates.isotope_data['isotope'] == isotope_name]
                
                global Z, N, A, mNucl

                Z         = float(isotope_row['Z'])
                N         = float(isotope_row['N'])
                A         = Z + N
                mNucl     = (Z + N) * u.u_atom
                abundance = float(isotope_row['abundance'])

                res_array.append(abundance / A / u.u_atom * exposure * u.keV * np.array([ 
                (integrate_flux(Er) + integrate_flux_lines(Er) ) for Er in Er_vec]))

            return np.sum(res_array, axis = 0)

        elif nucl_target_material == 'Ge' :
            spin  = 0.
            return  ( 1./ A / u.u_atom * exposure * u.keV * np.array([ 
                (integrate_flux(Er) + integrate_flux_lines(Er) ) for Er in Er_vec]) )



class xenon1t_detector_effects :
    
    def effeciency_data() :
        ''' 
        Output: 
        Efficiency data interpolated wrt energy [eV]
        
        Notes:
        Use official Xenon efficiency data (from data dump)
        '''
        efficiency_data = pd.read_csv('data/efficiency.txt', delimiter=',', skiprows = 8)
        efficiency_data['energy'] *= u.keV
        return scipy.interpolate.interp1d(efficiency_data['energy'],efficiency_data['totaleff'],bounds_error=False,fill_value = 0)
    
    efficiency_interp = effeciency_data()
    
    def energy_resolution(Er) :
        ''' 
        Output: 
        Energy resolution [eV]
        
        Input: 
        Er = recoil energy [eV]
        
        Notes: 
        Based on the fit in the subsection above 
        using the data from Xenon1T's nature paper 
        "Observation of two-neutrino double electron capture in 124Xe with XENON1T"
        '''
        a = 1014.3240405978196
        b = 0.42982967424679536
        return ( (a*Er**(-0.5) + b)/100. * Er)
    
    def spectrum_smearing(Er_vec, signal_vec) :
        
        if len(signal_vec) == len(Er_vec) :
            
            length_Er = len(Er_vec)
            energy_res_disc = xenon1t_detector_effects.energy_resolution(Er_vec)

            return np.array([
                       np.trapz(
                               scipy.stats.norm.pdf(Er,scale = energy_res_disc, loc = Er_vec) * signal_vec,
                       x = Er_vec) 
                   for Er in Er_vec]) * xenon1t_detector_effects.efficiency_interp(Er_vec) 
        else :
            print("Input arrays shape incompatible")
            return 0

class borexino_detector_effects :

    def Er_to_Nh_fit(Er) :
        a = 4.38198680e-04
        b = -3.79762744e-11
        c = 3.28883891e-18
        d = 9.87407700e-01
        return a*Er + b*Er**2 + c*Er**3 + d

    def Nh_to_Er_fit(Nh) :
        a = 2.25220982e+03
        b = 5.41708252e-01
        c = -7.01974716e-09
        d = 3.96223638e+02
        return a*Nh + b*Nh**2 + c*Nh**3 + d

    def derivative_Er_to_Nh_fit(Er) :
        a = 4.38198680e-04
        b = -3.79762744e-11
        c = 3.28883891e-18
        d = 9.87407700e-01
        return a + 2*b*Er + 3*c*Er**2

    def derivative_Nh_to_Er_fit(Nh) :
        a = 2.25220982e+03
        b = 5.41708252e-01
        c = -7.01974716e-09
        d = 3.96223638e+02
        return a + 2*b*Nh + 3*c*Nh**2
    

        return ( (a*Er**(-0.5) + b)/100. * Er)

    def energy_resolution(Nh) :
        ''' 
        Output: 
        Energy resolution [eV]
        
        Input: 
        Er = recoil energy [eV]
        
        Notes: 
        Based on the fit in the subsection above 
        using the data from Xenon1T's nature paper 
        "Observation of two-neutrino double electron capture in 124Xe with XENON1T"
        '''
        # a = 1.41254955
        # b = -0.01780431
        a = 1.41254955
        b = -0.01780431
        return ( (a*Nh**(-0.5) + b) * Nh)
    

    def spectrum_smearing(Er_vec, signal_vec) :
        
        if len(signal_vec) == len(Er_vec) :
            
            Nh_vec          = borexino_detector_effects.Er_to_Nh_fit(Er_vec)
            energy_res_disc = 1. * borexino_detector_effects.energy_resolution(Nh_vec)
            signal_vec      = signal_vec * borexino_detector_effects.derivative_Nh_to_Er_fit(Nh_vec) / u.keV
            # energy_res_disc = 0.075 * Nh_vec

            res = np.array([
                       np.trapz(
                               scipy.stats.norm.pdf(Nh, scale = energy_res_disc, loc = Nh_vec) * signal_vec,
                       x = Nh_vec) 
                   for Nh in Nh_vec]) 

            return res
        else :
            print("Input arrays shape incompatible")
            return 0
 