import jax.numpy as jnp
from AIQMCrelease3.DMC.main_dmc import main

structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['C']
atoms = jnp.array([[0.0, 0.0, -1.0]])
charges = jnp.array([4.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0,])

Rn_local = jnp.array([[1.0, 3.0, 2.0]])
Rn_non_local = jnp.array([[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]])
Local_coes = jnp.array([[4.00000, 57.74008, -25.81955]])
Local_exps = jnp.array([[14.43502, 8.39889, 7.38188]])

Non_local_coes = jnp.array([[[52.13345, 0], [0, 0], [0, 0]]])

Non_local_exps = jnp.array([[[7.76079, 0], [0, 0], [0, 0]]])

output = main(atoms=atoms,
              charges=charges,
              spins=spins,
              nelectrons=4,
              natoms=1,
              ndim=3,
              batch_size=4,
              iterations=2,
              tstep=0.05,
              nspins=(2, 2),
              nsteps=5,
              nblocks=2,
              feedback=1.0,
              save_path='save',
              restore_path='restore_DMC',
              save_frequency=0.01,
              structure=structure,
              Rn_local=Rn_local,
              Local_coes=Local_coes,
              Local_exps=Local_exps,
              Rn_non_local=Rn_non_local,
              Non_local_coes=Non_local_coes,
              Non_local_exps=Non_local_exps,)