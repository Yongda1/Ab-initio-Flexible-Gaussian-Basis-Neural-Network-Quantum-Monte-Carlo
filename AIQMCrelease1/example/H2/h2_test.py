import jax.numpy as jnp
from AIQMCrelease1.main.main_all_electrons import main

"""tomorrow, we test the codes for all electrons calculation H2 first."""
"""maybe we made a mistake about the wavefunction construction module. To debug the location of bug, we have to import the 
wavefunction from ferminet temporarily.11.1.2025."""
structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['H', 'H']
atoms = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
charges = jnp.array([1.0, 1.0])
spins = jnp.array([1.0, -1.0])
output = main(atoms=atoms,
              charges=charges,
              spins=spins,
              tstep=0.1,
              nsteps=10,
              nelectrons=2,
              natoms=2,
              ndim=3,
              batch_size=4,
              iterations=1000,
              save_path='F:\Ab-initio-Flexible-Gaussian-Basis-Neural-Network-Quantum-Monte-Carlo\AIQMCrelease1\example\H2',
              restore_path=None,
              save_frequency=0.01,
              structure=structure,)
"""to be continued ..."""

