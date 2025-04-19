import jax.numpy as jnp
from AIQMCpretrain1.main.main_pretain import main


structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['C']
atoms = jnp.array([[0.0, 0.0, 0.0]])
charges = jnp.array([6.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

output = main(atoms=atoms,
              charges=charges,
              spins=spins,
              nelectrons=6,
              natoms=1,
              ndim=3,
              batch_size=100,
              iterations=1000,
              tstep=0.02,
              nspins=(3, 3),
              nsteps=10,
              list_l=2,
              save_path='save', #/root/save
              restore_path=None,
              structure=structure,)