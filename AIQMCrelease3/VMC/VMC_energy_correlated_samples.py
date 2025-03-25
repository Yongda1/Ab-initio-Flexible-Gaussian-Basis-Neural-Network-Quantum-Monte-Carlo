"""here, we add correlated samples for VMC calculation 25.3.2025."""
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
from AIQMCrelease3.correlatedsamples.corrsamples import correlated_samples
from AIQMCrelease3.wavefunction_Ynlm import nn
from AIQMCrelease3.Energy import pphamiltonian
from AIQMCrelease3.Loss import pploss as qmc_loss_functions
from AIQMCrelease3 import constants
from AIQMCrelease3 import curvature_tags_and_blocks
from AIQMCrelease3.Optimizer import adam
from AIQMCrelease3.utils import writers
from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.spin_indices import jastrow_indices_ee
from AIQMCrelease3.correlatedsamples import corrsamples
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
    logging.info('Quantum Monte Carlo Start running')
    num_devices = jax.local_device_count()  # the amount of GPU per host
    num_hosts = jax.device_count() // num_devices  # the amount of host
    # jax.debug.print("num_devices:{}", num_devices)
    # jax.debug.print("num_hosts:{}", num_hosts)
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
        raise ValueError('correlated samples VMC must use the wave function from VMC!')

    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins, nelectrons=8)
    network = nn.make_ai_net(ndim=ndim,
                             nelectrons=nelectrons,
                             natoms=natoms,
                             nspins=nspins,
                             determinants=1,
                             charges=charges,
                             parallel_indices=parallel_indices,
                             antiparallel_indices=antiparallel_indices,
                             n_parallel=n_parallel,
                             n_antiparallel=n_antiparallel)

    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    key = jax.random.PRNGKey(seed)
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    #jax.debug.print("params：{}", params)
    signed_network = network.apply
    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    mc_step = VMCmcstep.main_monte_carlo(
        f=signed_network,
        tstep=tstep,
        ndim=ndim,
        nelectrons=nelectrons,
        nsteps=nsteps,
        batch_size=batch_size)

    mc_step_parallel = jax.pmap(mc_step, donate_argnums=1)
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
                                             list_l=list_l,
                                             use_scan=False)
    batch_local_energy = jax.pmap(
        jax.vmap(localenergy, in_axes=(None, None, nn.AINetData(positions=0, spins=0, atoms=0, charges=0)),
                 out_axes=(0, 0)))

    batch_local_energy_correlated_samples = jax.vmap(jax.pmap(
        jax.vmap(localenergy, in_axes=(None, None, nn.AINetData(positions=0, spins=0, atoms=0, charges=0)),
                 out_axes=(0, 0))), in_axes=(None, None, nn.AINetData(positions=0, spins=None, atoms=None, charges=None)))

    Energy = []
    new_atoms = jnp.array([[[0, 0, -1.0 + 0.1], [0 - 0.1, 0 + 0.1, 1.0]],
                           [[0, 0, -1.0 - 0.1], [0 + 0.1, 0 - 0.1, 1.0]]])
    correlatedsamples_parallel = jax.pmap(jax.vmap(jax.vmap(corrsamples.correlated_samples, in_axes=(None, None, 0)), in_axes=(None, 0, None)), in_axes=(None, None, 0))

    #for t in range(t_init, t_init + iterations):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    data = mc_step_parallel(params, data, subkeys)
    e_l, e_l_mat = batch_local_energy(params, subkeys, data)
    #jax.debug.print("atoms:{}", atoms)
    #jax.debug.print("newatoms:{}", new_atoms)
    #jax.debug.print("data:{}", data)
    pos = data.positions
    jax.debug.print("pos:{}", pos)
    newpos = correlatedsamples_parallel(atoms, new_atoms, pos)
    jax.debug.print("newpos:{}", newpos)
    #jax.debug.print("e_l:{}", e_l)
    newpos = jnp.reshape(newpos, (2, 1, 4, -1)) # 2 is the number of new atoms, 1 is fixed, 4 is the number of batch size
    jax.debug.print("newpos:{}", newpos)
    data.positions = newpos
    jax.debug.print("data:{}", data)
    new_energy, new_energy_mat = batch_local_energy_correlated_samples(params, subkeys, data)
    jax.debug.print("new_energy:{}", new_energy)
    Energy.append(e_l)
    #loss = constants.pmean(jnp.mean(e_l))
        #jax.debug.print("energy:{}", loss)

    #jax.debug.print("Energy:{}", Energy)
    mean_value = jnp.mean(jnp.array(Energy))
    #jax.debug.print("mean_value:{}", mean_value)