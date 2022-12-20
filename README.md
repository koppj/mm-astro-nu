# mm-astro-nu – Magnetic moments of astrophysical neutrinos

This is the code accompanying the paper "Magnetic Moments of Astrophysical Neutrinos" by Joachim Kopp (CERN), 
Toby Operfkuch (UC Berkeley), and Edward Wang (TU Munich).

The centerpiece of the code is the Jupyter notebook `mm-astro.ipynb`, which has been used to produce
all the plots in the paper. The first part of the notebook deals with *supernova neutrinos*, in particular
the disappearance of left-handed (active) $\nu_L$ into unobservable right-handed (sterile) $N_R$ due
to interactions of neutrino magnetic moments with interstellar magnetic fields

The second part is about the flavor ratios of *ultra-high energy astrophysical neutrinos* and how they
are affected in presence of neutrino magnetic moments.

The backend (where most of the physics happens) can be found in the `neutrino_propagator` class, which
is implemented in `moment.py`.

Ancillary files can be found in the `data/`, `cross_sections`, and `an-data` subdirectories.
