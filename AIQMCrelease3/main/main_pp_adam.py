"""To solve the compatibility between jax-0.5.3 with our codes,
we write this part to check where we need do the correction.
Because the kfac optimizer is quite complicated, we choose adam optimizer first to test everything.
21.3.2025."""
import jax
import chex
import optax
import logging
import time
import jax.numpy as jnp
import numpy as np
import kfac_jax
from typing_extensions import Protocol
from typing import Optional, Tuple, Union
from AIQMCrelease3 import checkpoint
from jax.experimental import multihost_utils
from AIQMCrelease3.VMC import VMCmcstep
from AIQMCrelease3.wavefunction_Ynlm import nn
from AIQMCrelease3.Energy import pphamiltonian
from AIQMCrelease3.Loss import pploss as qmc_loss_functions
from AIQMCrelease3 import constants
from AIQMCrelease3 import curvature_tags_and_blocks
from AIQMCrelease3.Optimizer.kfac import make_kfac_training_step
from AIQMCrelease3.utils import writers
from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.spin_indices import jastrow_indices_ee
import functools

def main(atoms: jnp.array,
         charges: jnp.array,
         spins: jnp.array,
         tstep: float,
         nelectrons: int,
         nsteps: int,
         natoms: int,
         ndim: int,
         batch_size: int,
         iterations: int,
         list_l: int, #for the angular momentum order in the pp file. It depends on the detial of the correpsonding pp file.
         nspins: Tuple,
         save_path: Optional[str],
         restore_path: Optional[str],
         #pp parameters
         Rn_local: jnp.array,
         Local_coes: jnp.array,
         Local_exps: jnp.array,
         Rn_non_local: jnp.array,
         Non_local_coes: jnp.array,
         Non_local_exps: jnp.array,
         save_frequency: float,
         structure: jnp.array,):
    """the main function for the pp calculation."""
    logging.info('Quantum Monte Carlo Start running')
    num_devices = jax.local_device_count()  # the amount of GPU per host
    num_hosts = jax.device_count() // num_devices  # the amount of host
    jax.debug.print("num_devices:{}", num_devices)
    jax.debug.print("num_hosts:{}", num_hosts)
    logging.info('Start QMC with $i devices per host, across %i hosts.', num_devices, num_hosts)
    if batch_size % (num_devices * num_hosts) != 0:
        raise ValueError('Batch size must be divisible by number of devices!')
    host_batch_size = batch_size // num_hosts  # how many configurations we put on one host
    device_batch_size = host_batch_size // num_devices  # how many configurations we put on one GPU
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    key = jax.random.PRNGKey(seed)
    ckpt_save_path = checkpoint.create_save_path(save_path=save_path)
    ckpt_restore_path = checkpoint.get_restore_path(restore_path=restore_path)

    ckpt_restore_filename = (checkpoint.find_last_checkpoint(ckpt_save_path) or
                             checkpoint.find_last_checkpoint(ckpt_restore_path))

    if ckpt_restore_filename:
        (t_init,
         data,
         params,
         opt_state_ckpt,) = checkpoint.restore(ckpt_restore_filename, host_batch_size)
    else:
        logging.info('No checkpoint found. Training new model.')
        t_init = 0
        opt_state_ckpt = None
        """to be continued...21.3.2025."""



    return None




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
              save_path=None,
              restore_path=None,
              save_frequency=0.01,
              structure=structure,
              Rn_local=Rn_local,
              Local_coes=Local_coes,
              Local_exps=Local_exps,
              Rn_non_local=Rn_non_local,
              Non_local_coes=Non_local_coes,
              Non_local_exps=Non_local_exps,)