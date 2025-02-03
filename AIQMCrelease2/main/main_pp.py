"""this main.py can execute the pseudopotential calculation. 2.2.2025"""
import jax
import chex
import optax
import logging
import time
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol
from typing import Optional, Tuple, Union
from AIQMCrelease2.initial_electrons_positions.init import init_electrons
from AIQMCrelease2 import checkpoint
from jax.experimental import multihost_utils
from AIQMCrelease2.VMC import VMCmcstep
from AIQMCrelease2.wavefunction import nn
from AIQMCrelease2.Energy import pphamiltonian
from AIQMCrelease2.Loss import pploss as qmc_loss_functions
from AIQMCrelease2 import constants
from AIQMCrelease2 import curvature_tags_and_blocks
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

        key, subkey = jax.random.split(key)
        data_shape = (num_devices, device_batch_size)
        batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
        batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
        batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
        batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
        pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                                    electrons=spins,
                                    batch_size=host_batch_size, init_width=1.0)

        batch_pos = jnp.reshape(pos, data_shape + (-1,))
        batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
        batch_spins = jnp.repeat(spins[None, ...], batch_size, axis=0)
        batch_spins = jnp.reshape(batch_spins, data_shape + (-1,))
        batch_spins = kfac_jax.utils.broadcast_all_local_devices(batch_spins)
        data = nn.AINetData(positions=batch_pos, spins=batch_spins, atoms=batch_atoms, charges=batch_charges)

    feature_layer = nn.make_ferminet_features(natoms=natoms, nspins=nspins, ndim=ndim, )
    network = nn.make_fermi_net(ndim=ndim,
                                nspins=nspins,
                                determinants=1,
                                feature_layer=feature_layer,
                                charges=charges,
                                full_det=True)

    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    logging.info('--------------Main training--------------')
    logging.info('--------------Build Monte Carlo engine--------------')
    mc_step = VMCmcstep.main_monte_carlo(
        f=signed_network,
        tstep=tstep,
        ndim=ndim,
        nelectrons=nelectrons,
        nsteps=nsteps,
        batch_size=batch_size)
    mc_step_parallel = jax.pmap(mc_step)
    logging.info('--------------Create Hamiltonian--------------')

    localenergy = pphamiltonian.local_energy(f=signed_network,
                                             lognetwork=log_network,
                                             charges=charges,
                                             nspins=spins,
                                             rn_local=Rn_local,
                                             local_coes=Local_coes,
                                             local_exps=Local_exps,
                                             rn_non_local=Rn_non_local,
                                             non_local_coes=Non_local_coes,
                                             non_local_exps=Non_local_exps,
                                             natoms=natoms,
                                             nelectrons=nelectrons,
                                             ndim=ndim,
                                             list_l=0,
                                             use_scan=False)

    """so far, we have not constructed the pp module. Currently, we only execute all electrons calculation.  """
    logging.info('--------------Build loss function--------------')
    evaluate_loss = qmc_loss_functions.make_loss(network=log_network,
                                                 local_energy=localenergy,
                                                 clip_local_energy=5.0,
                                                 clip_from_median=False,
                                                 center_at_clipped_energy=True,
                                                 complex_output=True,
                                                 )