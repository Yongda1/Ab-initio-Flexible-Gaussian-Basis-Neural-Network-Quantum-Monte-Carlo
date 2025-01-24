import jax.numpy as jnp
from AIQMCrelease2.VMC.VMCmain import main

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
              nsteps=1,
              nelectrons=2,
              natoms=2,
              ndim=3,
              batch_size=4,
              iterations=1000,
              save_path='Save',
              restore_path='Restore',
              save_frequency=0.01,
              structure=structure,)
"""to be continued ..."""

