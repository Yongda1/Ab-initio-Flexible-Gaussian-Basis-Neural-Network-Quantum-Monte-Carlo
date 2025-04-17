import jax.numpy as jnp
from AIQMCrelease3.DMC.main_dmc_correlated_samples import main

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

primary_weights = jnp.array([[[1.0356745,  0.9655544,  1.0356745,  0.9655544 ],
                    [1.2351745,  0.8096022,  1.2351745,  0.8096022 ],
                    [1.4008662,  0.68584216, 1.4008662,  0.68584216],
                    [1.3275279,  0.6499369,  1.3275279,  0.6499369 ]],
                   [[0.7056982,  0.7056982,  0.7056982,  0.7056982 ],
                    [0.75946516, 0.43923092, 0.43923092, 0.43923092],
                    [1.2790449,  0.26111063, 0.26111063, 0.26111063],
                    [1.7451661,  0.19158685, 0.22207397, 0.19158685]]])
new_atoms = jnp.array([[[0, 0, -1.0 + 0.1], [0 - 0.1, 0 + 0.1, 1.0]],
                           [[0, 0, -1.0 - 0.1], [0 + 0.1, 0 - 0.1, 1.0]]])

output = main(atoms=atoms,
              charges=charges,
              spins=spins,
              nelectrons=8,
              natoms=2,
              ndim=3,
              batch_size=4,
              iterations=4,
              tstep=0.05,
              nspins=(4, 4),
              nsteps=5,
              nblocks=2,
              feedback=1.0,
              save_path='None',
              restore_path='restore_DMC',
              save_frequency=0.01,
              structure=structure,
              Rn_local=Rn_local,
              Local_coes=Local_coes,
              Local_exps=Local_exps,
              Rn_non_local=Rn_non_local,
              Non_local_coes=Non_local_coes,
              Non_local_exps=Non_local_exps,
              primary_weights=primary_weights,
              new_atoms=new_atoms,)