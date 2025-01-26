import jax.numpy as jnp
from AIQMCrelease2.VMC.VMC_optimize_main_pp import main

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
Rn_local = jnp.array([[1, 3, 2],
                      [1, 3, 2],
                      [1, 3, 2]])
Rn_non_local = jnp.array([[2],
                          [2],
                          [2]])
Local_coes = jnp.array([[4.00000, 57.74008, -25.81955],
                               [6.000000, 73.85984, -47.87600],
                               [6.000000, 73.85984, -47.87600]])
Nonlocal_coes = jnp.array([[52.13345],
                                  [85.86406],
                                  [85.86406]])
Local_exps = jnp.array([[14.43502, 8.39889, 7.38188],
                            [12.30997, 14.76962, 13.71419],
                            [12.30997, 14.76962, 13.71419]])

Nonlocal_exps = jnp.array([[7.76079],
                               [13.65512],
                               [13.65512]])

output = main(atoms=atoms,
              charges=charges,
              spins=spins,
              Rn_local=Rn_local,
              Local_coes=Local_coes,
              Local_exps=Local_exps,
              Rn_non_local=Rn_non_local,
              Non_local_coes=Nonlocal_coes,
              Non_local_exps=Nonlocal_exps,
              tstep=0.02,
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

