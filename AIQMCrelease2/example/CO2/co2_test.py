import jax.numpy as jnp
from AIQMCrelease2.main.main_pp import main

structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['C', 'O', 'O']
atoms = jnp.array([[1.33, 1.0, 1.0], [0.0, 1.0, 1.0], [2.66, 1.0, 1.0]])
charges = jnp.array([4.0, 6.0, 6.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
Rn_local = jnp.array([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
Rn_non_local = jnp.array([[[2.0], [2.0], [2.0]]])
Local_coes = jnp.array([[4.00000, 57.74008, -25.81955],
                        [6.000000, 73.85984, -47.87600],
                        [6.000000, 73.85984, -47.87600]])
Local_exps = jnp.array([[14.43502, 8.39889, 7.38188],
                        [12.30997, 14.76962, 13.71419],
                        [12.30997, 14.76962, 13.71419]])
Non_local_coes = jnp.array([[52.13345], [85.86406], [85.86406]])
Non_local_exps = jnp.array([[7.76079], [13.65512], [13.65512]])
output = main(atoms=atoms,
              charges=charges,
              spins=spins,
              nelectrons=16,
              natoms=3,
              ndim=3,
              batch_size=4,
              iterations=100,
              structure=structure,
              Rn_local=Rn_local,
              Local_coes=Local_coes,
              Local_exps=Local_exps,
              Rn_non_local=Rn_non_local,
              Non_local_coes=Non_local_coes,
              Non_local_exps=Non_local_exps,)