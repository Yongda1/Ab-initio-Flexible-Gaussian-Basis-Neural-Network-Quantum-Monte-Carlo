import jax.numpy as jnp
from AIQMCrelease3.main.main_pp_adam_muti_GPU import main


structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['C', 'C']
atoms = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
charges = jnp.array([4.0, 4.0])
spins = jnp.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

Rn_local = jnp.array([[1.0, 3.0, 2.0],
                      [1.0, 3.0, 2.0]])

Rn_non_local = jnp.array([[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                          [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],])

Local_coes = jnp.array([[4.00000, 57.74008, -25.81955],
                        [4.00000, 57.74008, -25.81955]])

Local_exps = jnp.array([[14.43502, 8.39889, 7.38188],
                        [14.43502, 8.39889, 7.38188],])

Non_local_coes = jnp.array([[[52.13345, 0], [0, 0], [0, 0]],
                            [[52.13345, 0], [0, 0], [0, 0]],])

Non_local_exps = jnp.array([[[7.76079, 0], [0, 0], [0, 0]],
                            [[7.76079, 0], [0, 0], [0, 0]],])

output = main(atoms=atoms,
              charges=charges,
              spins=spins,
              nelectrons=8,
              natoms=2,
              ndim=3,
              batch_size=4,
              iterations=10,
              tstep=0.05,
              nspins=(4, 4),
              nsteps=10,
              list_l=2,
              save_path="save",
              restore_path=None,
              save_frequency=0.01,
              structure=structure,
              Rn_local=Rn_local,
              Local_coes=Local_coes,
              Local_exps=Local_exps,
              Rn_non_local=Rn_non_local,
              Non_local_coes=Non_local_coes,
              Non_local_exps=Non_local_exps,)