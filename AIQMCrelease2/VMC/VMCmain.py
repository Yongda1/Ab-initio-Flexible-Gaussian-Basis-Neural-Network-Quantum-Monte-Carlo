"""VMC.
One small idea, could we do the same thing like DLSS for Quantum Monte Carlo?"""
import jax
import logging
import time
import jax.numpy as jnp
import kfac_jax
from typing import Optional, Tuple, Union
from AIQMCrelease2 import checkpoint
from jax.experimental import multihost_utils
from AIQMCrelease2.VMC import VMCmcstep
from AIQMCrelease2.wavefunction import nn
from AIQMCrelease2.Energy import hamiltonian
from AIQMCrelease2 import constants

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
         save_path: Optional[str],
         restore_path: Optional[str],
         save_frequency: float,
         structure: jnp.array,):
    logging.info('Variational Quantum Monte Carlo Start running')
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
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    ckpt_save_path = checkpoint.create_save_path(save_path=save_path)
    ckpt_restore_path = checkpoint.get_restore_path(restore_path=restore_path)
    #jax.debug.print("ckpt_restore_path:{}", ckpt_restore_path)

    ckpt_restore_filename = checkpoint.find_last_checkpoint(ckpt_restore_path)
    #jax.debug.print("ckpt_restore_filename:{}", ckpt_restore_filename)

    (t_init,
     data,
     params,
     opt_state_ckpt,) = checkpoint.restore(ckpt_restore_filename, host_batch_size)
    #jax.debug.print("t_init:{}", t_init)
    #jax.debug.print("data:{}", data)
    #jax.debug.print("params:{}", params)
    feature_layer = nn.make_ferminet_features(natoms=natoms, nspins=(1, 1), ndim=ndim, )
    network = nn.make_fermi_net(ndim=ndim,
                                nspins=(1, 1),
                                determinants=1,
                                feature_layer=feature_layer,
                                charges=charges,
                                full_det=True)
    signed_network = network.apply

    # jax.debug.print("charges:{}", charges)

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
    mc_step_parallel = jax.pmap(mc_step)
    localenergy = hamiltonian.local_energy(f=signed_network, charges=charges, nspins=spins, use_scan=False)
    batch_local_energy = jax.pmap(jax.vmap(localenergy, in_axes=(None, None, nn.AINetData(positions=0, spins=0, atoms=0, charges=0)), out_axes=(0, 0)))
    Energy = []
    for t in range(t_init, t_init+iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        data = mc_step_parallel(params, data, subkeys)
        jax.debug.print("data.positions:{}", data.positions)
        e_l,  e_l_mat = batch_local_energy(params, subkeys, data)
        jax.debug.print("e_l:{}", e_l)
        Energy.append(e_l)
        loss = constants.pmean(jnp.mean(e_l))
        jax.debug.print("energy:{}", loss)

    jax.debug.print("Energy:{}", Energy)
    mean_value = jnp.mean(jnp.array(Energy))
    jax.debug.print("mean_value:{}", mean_value)