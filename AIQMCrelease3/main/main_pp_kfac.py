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
from AIQMCrelease3.Optimizer import adam
from AIQMCrelease3.utils import writers
from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.spin_indices import jastrow_indices_ee
from AIQMCrelease3.spin_indices import spin_indices_h
from AIQMCrelease3.Optimizer import kfac
import functools
#logging.basicConfig(level = logging.INFO)

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
        key, subkey = jax.random.split(key)
        data_shape = (num_devices, device_batch_size)
        batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
        batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
        batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
        batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
        pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                                    electrons=spins,
                                    batch_size=host_batch_size, init_width=1.0)
        generate_spin_indices = spins
        batch_pos = jnp.reshape(pos, data_shape + (-1,))
        batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
        batch_spins = jnp.repeat(spins[None, ...], batch_size, axis=0)
        batch_spins = jnp.reshape(batch_spins, data_shape + (-1,))
        batch_spins = kfac_jax.utils.broadcast_all_local_devices(batch_spins)
        data = nn.AINetData(positions=batch_pos, spins=batch_spins, atoms=batch_atoms, charges=batch_charges)

    #jax.debug.print("data:{}", data)
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins, nelectrons=nelectrons)
    spin_up_indices, spin_down_indices = spin_indices_h(generate_spin_indices)
    network = nn.make_ai_net(ndim=ndim,
                             nelectrons=nelectrons,
                             natoms=natoms,
                             nspins=nspins,
                             determinants=1,
                             charges=charges,
                             parallel_indices=parallel_indices,
                             antiparallel_indices=antiparallel_indices,
                             n_parallel=n_parallel,
                             n_antiparallel=n_antiparallel,
                             spin_up_indices=spin_up_indices,
                             spin_down_indices=spin_down_indices
                             )

    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    # jax.debug.print("params:{}", params)
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
        batch_size=int(batch_size / (num_devices * num_hosts)))
    jax.debug.print("batch_size_run:{}", int(batch_size / (num_devices * num_hosts)))
    mc_step_parallel = jax.pmap(mc_step, donate_argnums=1)
    logging.info('--------------Create Hamiltonian--------------')
    """tomorrow, we start to check the pseudopotential part."""
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

    evaluate_loss = qmc_loss_functions.make_loss(network=log_network,
                                                 local_energy=localenergy,
                                                 clip_local_energy=5.0,
                                                 clip_from_median=False,
                                                 center_at_clipped_energy=True,
                                                 complex_output=True,
                                                 )

    def learning_rate_schedule(t_: jnp.array, rate=0.05, delay=1.0, decay=10000) -> jnp.array:
        return rate * jnp.power(1.0 / (1.0 + (t_ / delay)), decay)

    """we try adam optimzier here."""
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

    def learning_rate_schedule(t_: jnp.array, rate=0.05, delay=1.0, decay=10000) -> jnp.array:
        return rate * jnp.power(1.0 / (1.0 + (t_ / delay)), decay)

    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=0.0,
        norm_constraint=0.001,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=0.95,
        inverse_update_period=1,
        min_damping=1.0e-4,
        num_burnin_steps=0,
        register_only_generic=False,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS)
    )

    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params=params, rng=subkeys, batch=data)

    step_kfac = kfac.make_kfac_training_step(
        damping=0.001,
        optimizer=optimizer,
        reset_if_nan=False)

    train_schema = ['step', 'energy']
    writer_manager = writers.Writer(
        name='train_states',
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False
    )
    time_of_last_ckpt = time.time()
    """main training loop"""
    with writer_manager as writer:
        for t in range(t_init, t_init + iterations):
            """we need do more to deal with amda optimzier. especially for saving module. 23.3.2025."""
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            #jax.debug.print("subkeys:{}", subkeys)
            #jax.debug.print("before_run_data:{}", data)
            data = mc_step_parallel(params, data, subkeys)
            #jax.debug.print("after_run_data:{}", data)
            data, params, opt_state, loss, aux_data =  step_kfac(data, params, opt_state, subkeys)
            loss = loss[0]
            logging_str = ('Step %05d: ', '%03.4f E_h,')
            logging_args = t, loss,
            writer_kwargs = {
                'step': t,
                'energy': np.asarray(loss),
            }
            # jax.debug.print("loss:{}", loss)
            logging.info(logging_str, *logging_args)
            writer.write(t, **writer_kwargs)
            #jax.debug.print("time.time{}", time.time())
            #jax.debug.print("time_of_last_ckpt:{}", time_of_last_ckpt)
            #if time.time() - time_of_last_ckpt > save_frequency * 60:
            if t % 1000 == 0:
                """here, we store every step optimization."""
                save_params = np.asarray(params)
                save_opt_state = np.asarray(opt_state, dtype=object)
                # jax.debug.print("save_params:{}", save_params)
                # jax.debug.print("ckpt_save_path:{}", ckpt_save_path)
                checkpoint.save(ckpt_save_path, t, data, save_params, save_opt_state)
                # time_of_last_ckpt = time.time()
