import jax.numpy as jnp
import time
import jax
import kfac_jax
from jax.experimental import multihost_utils
from typing import Optional, Tuple, Union
from absl import logging
from GaussianNet import checkpoint # this check point needs to be rewritten.
from GaussianNet.wavefunction import networks
from GaussianNet.wavefunction import envelopes
from GaussianNet.wavefunction import spin_indices
import ml_collections
from jaqmcmain.jaqmc.dmc.dmc import run


def main(cfg: ml_collections.ConfigDict,
         charges: jnp.array,
         tstep: float,
         nelectrons: int,
         nsteps: int, #meaningless parameters.
         natoms: int,
         ndim: int,
         batch_size: int,
         iterations: int, #means the steps in DMC run.
         nblocks: int,
         feedback: float,
         nspins: Tuple,
         save_path: Optional[str],
         restore_path: Optional[str],):

    logging.info('Diffusion Quantum Monte Carlo Start running')
    num_devices = jax.local_device_count()  # the amount of GPU per host
    num_hosts = jax.device_count() // num_devices  # the amount of host
    logging.info(f'Start QMC with {num_devices} devices per host, across {num_hosts} hosts.')
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
    """here, the restore calculation can be read from save path or restore path."""
    ckpt_restore_filename = (checkpoint.find_last_checkpoint(ckpt_save_path))
    jax.debug.print("ckpt_restore_filename:{}", ckpt_restore_filename)
    if ckpt_restore_filename:
        (t_init,
         _data,
         params,
         opt_state_ckpt,
         mcmc_width_ckpt,
         density_state_ckpt) = checkpoint.restore(ckpt_restore_filename, host_batch_size)
        jax.debug.print("data.position:{}", _data.positions)
    else:
        raise ValueError('DMC must use the wave function from VMC!')

    spins_test = jnp.array([[1., 1., 1, - 1., - 1., -1]])
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = \
        spin_indices.jastrow_indices_ee(spins=spins_test,
                                        nelectrons=cfg.system.nelectrons)
    spins_test = jnp.array([[1., 1., 1, - 1., - 1., -1]])
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = \
        spin_indices.jastrow_indices_ee(spins=spins_test,
                                        nelectrons=6)
    _network = networks.make_gaussian_net(nspins=(3, 3),
                                          charges=charges,
                                          parallel_indices=parallel_indices,
                                          antiparallel_indices=antiparallel_indices,
                                          n_parallel=n_parallel,
                                          n_antiparallel=n_antiparallel, )

    data = _data.positions
    position = data.reshape((-1, data.shape[-1]))
    spins = _data.spins[0]
    # Get a single copy of network params from the replicated one
    single_params = jax.tree_map(lambda x: x[0], params)
    network = lambda params, pos: _network.apply(params, pos, spins, atoms, charges)
    network_wrapper = lambda x: network(params=single_params, pos=x)
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    output = run(position,
        200,
        network_wrapper,
        0.01,
        key,
        atoms,
        charges,
        t_init=0,
        local_energy_func=None,
        velocity_func=None,
        mixed_estimator_num_steps=5000,
        energy_window_size=1000,
        weight_branch_threshold=(0.3, 2),
        anchor_energy=None,
        update_energy_offset_interval=1,
        energy_offset_update_amplitude=1,
        energy_cutoff_alpha=0.2,
        effective_time_step_update_period=-1,
        energy_clip_pair=None,
        energy_outlier_rel_threshold=-1,
        fix_size=False,
        ebye_move=False,
        block_size=5000,
        max_restore_nums=3,
        num_hosts=1,
        host_idx=0,
        debug_mode=False,)