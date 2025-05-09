"""This module is the main function for DMC.
we also need make the dmc engine be suitable for the large scale run."""
import numpy as np
import jax.numpy as jnp
import time
import jax
import kfac_jax
from jax.experimental import multihost_utils
from typing import Optional, Tuple, Union
from absl import logging
from AIQMC import checkpoint # this check point needs to be rewritten.
from AIQMC.wavefunction import networks
from AIQMC.wavefunction import envelopes
from AIQMC.tools import utils
from AIQMC.hamiltonian import hamiltonian
from AIQMC.DMC.total_energy import calculate_total_energy
from AIQMC.DMC.branch import branch
from AIQMC.DMC.estimate_energy import estimate_energy
from AIQMC.tools.utils import writers
from AIQMC.wavefunction import spin_indices
from AIQMC import constants
import ml_collections
from AIQMC.DMC.drift_diffusion import propose_drift_diffusion_new
from AIQMC.tools.utils import utils
from AIQMC.DMC.S_matrix import comput_S_new


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
         data,
         params,
         opt_state_ckpt,
         mcmc_width_ckpt,
         density_state_ckpt) = checkpoint.restore(ckpt_restore_filename, host_batch_size)
        jax.debug.print("data.position:{}", data.positions)
    else:
        raise ValueError('DMC must use the wave function from VMC!')

    spins_test = jnp.array([[1., 1., 1, - 1., - 1., -1]])
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = \
        spin_indices.jastrow_indices_ee(spins=spins_test,
                                        nelectrons=cfg.system.nelectrons)
    envelope = envelopes.make_isotropic_envelope()
    feature_layer = networks.make_ferminet_features(
        natoms=1,
        nspins=(3, 3),
        ndim=3,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
    )
    use_complex = cfg.network.get('complex', False)
    network = networks.make_fermi_net(
        nspins,
        charges,
        nelectrons=cfg.system.nelectrons,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=cfg.system.states,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        full_det=cfg.network.full_det,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=use_complex,
        parallel_indices=parallel_indices,
        antiparallel_indices=antiparallel_indices,
        n_parallel=n_parallel,
        n_antiparallel=n_antiparallel,
        **cfg.network.ferminet,
    )

    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]

    laplacian_method = cfg.optim.get('laplacian', 'default')
    local_energy_fn = hamiltonian.local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        complex_output=use_complex,
        laplacian_method=laplacian_method)
    local_energy = local_energy_fn
    
    total_e = calculate_total_energy(local_energy=local_energy)
    total_e_parallel = jax.pmap(total_e)
    e_est, variance_est = total_e_parallel(params, subkeys, data)
    loss = constants.pmean(jnp.mean(e_est))
    e_trial = loss.real
    e_est = loss.real
    jax.debug.print("e_trial:{}", e_trial)
    jax.debug.print("e_est:{}", e_est)
    jax.debug.print("loss:{}", loss)

    """we need think more about the parallel strategy. So later, we have to modify the shape of weights and branchcut."""
    weights = jnp.ones(shape=(num_devices * num_hosts, int(batch_size / (num_devices * num_hosts))))
    esigma = jnp.std(e_est)

    branchcut_start = jnp.ones(shape=(num_devices * num_hosts, int(batch_size / (num_devices * num_hosts)))) * 10
    branch_parallel = jax.pmap(branch, in_axes=(0, 0, 0))
    energy_data = jnp.zeros(shape=(nblocks, iterations, batch_size))
    weights_data = jnp.zeros(shape=(nblocks, iterations, batch_size))

    """Start the main loop."""
    opt_state = opt_state_ckpt

    """the writer module need be modified. 14.2.2025."""
    train_schema = ['block', 'energy', 'positions']
    writer_manager = writers.Writer(
        name='DMC_states',
        schema=train_schema,
        directory=ckpt_restore_path,
        iteration_key=None,
        log=False
    )

    """to debug drift-diffusion process"""
    drift_diffusion = propose_drift_diffusion_new(f=signed_network,
                                tstep=0.02,
                                ndim=3,
                                nelectrons=6,
                                batch_size=device_batch_size,
                                complex_output=True)

    drift_diffusion_parallel = jax.pmap(jax.vmap(drift_diffusion, in_axes=(None, 0, 0)))

    phase_f = utils.select_output(signed_network, 0)
    logabs_f = utils.select_output(signed_network, 1)

    def calculate_gradient(params, data: networks.FermiNetData):
        grad_f = jax.grad(logabs_f, argnums=1)
        def grad_f_closure(x):
            return grad_f(params, x, data.spins, data.atoms, data.charges)

        grad_phase = jax.grad(phase_f, argnums=1)

        def grad_phase_closure(x):
            return grad_phase(params, x, data.spins, data.atoms, data.charges)

        primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)
        phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, data.positions)
        O = primal + 1.j * phase_primal
        return O

    calculate_gradient_parallel = jax.pmap(jax.vmap(calculate_gradient, in_axes=(None, 0)))
    compute_S_kernel = comput_S_new(tstep, nelectrons)
    comput_S_new_parallel = jax.pmap(jax.vmap(compute_S_kernel, in_axes=(0, 0, None, None, 0)), in_axes=(0, 0, None, None, 0))
    key_array = []
    for i in range(len(subkeys)):
        key_array_elements = jax.random.split(subkeys[i], num=device_batch_size)
        key_array.append(key_array_elements)
    key_array = jnp.array(key_array)

    with writer_manager as writer:
        for block in range(0, nblocks):
            for t in range(t_init, t_init+iterations):
                next_data, new_key = drift_diffusion_parallel(params, key_array, data)
                e_loc_old, variance_est = total_e_parallel(params, subkeys, data)
                e_loc_new, variance_est_next = total_e_parallel(params, subkeys, next_data)
                O_old = calculate_gradient_parallel(params, data)
                O_new = calculate_gradient_parallel(params, next_data)
                O_old = jnp.sum(jnp.square(O_old), axis=-1)
                O_new = jnp.sum(jnp.square(O_new), axis=-1)
                S_old = comput_S_new_parallel(O_old, e_loc_old, e_trial, e_est, branchcut_start)
                S_new = comput_S_new_parallel(O_new, e_loc_new, e_trial, e_est, branchcut_start)
                wmult = jnp.exp(tstep * (0.5 * S_new + 0.5 * S_old))
                mean_eloc_new = jnp.mean(e_loc_new)
                jax.debug.print("mean_eloc_new:{}", mean_eloc_new)
                weights = wmult * weights
                data = next_data
                key_array = new_key
                #jax.debug.print("e_loc_new:{}", e_loc_new.shape)

                energy = jnp.reshape(e_loc_new, batch_size)
                weights_step = jnp.reshape(weights, batch_size)
                temp_energy = energy_data[block].at[t-t_init].set(energy.real)
                temp_weights = weights_data[block].at[t-t_init].set(weights_step)
                energy_data = energy_data.at[block].set(temp_energy)
                weights_data = weights_data.at[block].set(temp_weights)


            e_est = estimate_energy(energy_data, weights_data)
            #jax.debug.print("e_est:{}", e_est)
            logging_str = ('Block %05d:', '%03.4f E_h,')
            logging_args = block, e_est,
            logging.info(logging_str, *logging_args)

            """rewrite this part. Here something is wrong. 25.4.2025."""
            if block % 2 == 0:
                """np.savez cannot save inhomogeneous array. So, we have to use the following line to convert the format of the arrays."""
                save_params = np.asarray(params)
                save_opt_state = np.asarray(opt_state, dtype=object)
                checkpoint.save(ckpt_restore_path, block, data, save_params, save_opt_state, mcmc_width=0.1)

            #jax.debug.print('weights:{}', weights)
            #jax.debug.print("subkeys:{}", subkeys)
            weights, newindices, subkeys = branch_parallel(data, weights.real, subkeys)
            x1 = data.positions
            x2 = []

            for i in range(len(x1)):
                unique, counts = jnp.unique(newindices[i], return_counts=True)
                temp = x1[i][unique]
                number_branches = jnp.max(counts)
                """here, we need think what if the walker is killed. 13.2.2025.
                we need add one more walker into the configurations.
                But now the problem is how to generate the new walker?
                Actually, this part belong to branch function. Currently, it is not a good solution. Maybe improve it later.
                we need find how to generate the new walkers. 26.4.2025."""
                if len(unique) < batch_size:
                    n = batch_size - len(unique)
                    extra_walkers = temp[-1] + tstep * jax.random.uniform(key, (n, nelectrons * ndim))
                    temp = jnp.concatenate([temp, extra_walkers], axis=0)
                    logging.info(f"max branches {number_branches} and number of walkers killed {n}")
                else:
                    logging.info(f"max branches {number_branches} and number of walkers killed {0}")

                x2.append(temp)

            x2 = jnp.array(x2)
            #jax.debug.print("x2:{}", x2)
            x2 = jnp.reshape(x2, (num_devices * num_hosts, int(batch_size/(num_devices * num_hosts)), -1))
            #jax.debug.print("x2:{}", x2)
            data = networks.FermiNetData(**(dict(data) | {'positions': x2}))

            """leave this to tomorrow. 13.2.2025. we also need update the branchcut."""
            #jax.debug.print("e_est:{}", e_est)
            #jax.debug.print("e_trial:{}", e_trial)
            e_trial = e_est - feedback * jnp.log(jnp.mean(weights)).real
            #jax.debug.print("e_trial_new:{}", e_trial_new)
            #jax.debug.print("e_est:{}", e_est)
            """probably, we made some mistakes about these number storage."""

            writer_kwargs = {
                'block': block,
                'energy': np.asarray(e_est),
                #'weights_data': np.asarray(weights_data),
                'positions': np.asarray(x2),
            }
            writer.write(block, **writer_kwargs)
