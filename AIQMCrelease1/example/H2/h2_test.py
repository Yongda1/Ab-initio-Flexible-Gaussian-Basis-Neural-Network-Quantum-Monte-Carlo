import jax.numpy as jnp
from AIQMCrelease1.main.main_kfac_all_electrons import main

"""tomorrow, we test the codes for all electrons calculation H2 first."""
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
              tstep=0.02,
              nelectrons=2,
              natoms=2,
              ndim=3,
              batch_size=4,
              iterations=1,
              structure=structure,)

