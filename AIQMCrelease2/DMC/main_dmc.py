"""This module is the main function for DMC.
we also need make the dmc engine be suitable for the large scale run."""
import jax.numpy as jnp
import time
import jax
import kfac_jax
from jax.experimental import multihost_utils
from typing import Optional, Tuple, Union
from absl import logging
from AIQMCrelease2 import checkpoint
from AIQMCrelease2.wavefunction import nn
from AIQMCrelease2.utils import utils
from AIQMCrelease2.DMC.drift_diffusion import propose_drift_diffusion
from AIQMCrelease2.DMC.Tmoves import compute_tmoves
from AIQMCrelease2.pseudopotential import pseudopotential
from AIQMCrelease2.DMC.S_matrix import comput_S


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
    logging.info('Diffusion Quantum Monte Carlo Start running')
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
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
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
        raise ValueError('DMC must use the wave function from VMC!')

    feature_layer = nn.make_ferminet_features(natoms=natoms, nspins=nspins, ndim=ndim, )
    network = nn.make_fermi_net(ndim=ndim,
                                nspins=nspins,
                                determinants=1,
                                feature_layer=feature_layer,
                                charges=charges,
                                full_det=True)
    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    logabs_f = utils.select_output(signed_network, 1)

    drift_diffusion = propose_drift_diffusion(
        logabs_f=logabs_f,
        tstep=tstep,
        ndim=ndim,
        nelectrons=nelectrons,
        batch_size=batch_size)
    drift_diffusion_pmap = jax.pmap(drift_diffusion)
    tmoves = compute_tmoves(list_l=2,
                            tstep=0.05,
                            nelectrons=nelectrons,
                            natoms=natoms,
                            ndim=ndim,
                            lognetwork=log_network,
                            Rn_non_local=Rn_non_local,
                            Non_local_coes=Non_local_coes,
                            Non_local_exps=Non_local_exps)

    #jax.debug.print("subkeys:{}", subkeys)
    tmoves_pmap = jax.pmap(jax.vmap(tmoves, in_axes=(nn.AINetData(positions=0, spins=0, atoms=0, charges=0), None, None)))
    """we need check the drift_diffusion process is correct or not.5.2.2025."""
    #jax.debug.print("new_data:{}", new_data)
    pos, acceptance = tmoves_pmap(data, params, subkeys)
    data = nn.AINetData(**(dict(data) | {'positions': pos}))
    jax.debug.print("pos:{}", pos)
    new_data, newkey, tdamp, grad_eff, grad_new_eff_s = drift_diffusion_pmap(params, sharded_key, data)
    """we need do a summary for t-moves and drift-diffusion process."""
    branchcut_start = 10
    """we need debug the local energy module.10.2.2025."""

    S_old = comput_S(e_trial=etrial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_eff), tau=tstep,
                     eloc=eloc_old, nelec=nelectrons)
    S_new = comput_S(e_trial=etrial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_new_eff_s), tau=tstep,
                     eloc=eloc_new, nelec=nelectrons)
