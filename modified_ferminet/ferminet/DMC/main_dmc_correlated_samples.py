"""
1. we generate secondary walks from the reference walk according to the space-warp transformation.
2. In the averages, we retain the ratios of the secondary and primary wave functions as in VMC.
3. The secondary weights are the primary ones multiplied by the product of the factors exp[S_s(R^s', R^s, tau_s) - S(R',R, tau)] for the last N_proj generations.
N_proj is chosen large enough to project out the secondary ground state, but small enough to avoid a considerable increase in the fluctuations.
In the exponential factors, we introduced tau_s because the secondary moves are effectively proposed with a different time step, tau_s.
"""
import numpy as np
import jax.numpy as jnp
import time
import jax
import kfac_jax
from jax.experimental import multihost_utils
from typing import Optional, Tuple, Union
from absl import logging
from AIQMCrelease3 import checkpoint
from AIQMCrelease3.wavefunction_Ynlm import nn
from AIQMCrelease3.utils import utils
from AIQMCrelease3.Energy import pphamiltonian
from AIQMCrelease3.DMC.total_energy import calculate_total_energy
from AIQMCrelease3.DMC.dmc import dmc_propagate
from AIQMCrelease3.DMC.branch import branch
from AIQMCrelease3.DMC.estimate_energy import estimate_energy
from AIQMCrelease3.utils import writers
from AIQMCrelease3.spin_indices import jastrow_indices_ee
from AIQMCrelease3.correlatedsamples import corrsamples
from AIQMCrelease3.correlatedsamples import jacobianWeights


def main(atoms: jnp.array,
         charges: jnp.array,
         spins: jnp.array,
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
         restore_path: Optional[str],
         #pp parameters
         Rn_local: jnp.array,
         Local_coes: jnp.array,
         Local_exps: jnp.array,
         Rn_non_local: jnp.array,
         Non_local_coes: jnp.array,
         Non_local_exps: jnp.array,
         save_frequency: float,
         structure: jnp.array,
         primary_weights: jnp.array,
         new_atoms: jnp.array,):
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
        jax.debug.print("data:{}", data)
    else:
        raise ValueError('correlated samples calculation of DMC must use the wave function from DMC!')

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
    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    logabs_f = utils.select_output(signed_network, 1)

    log_network_parallel = jax.pmap(jax.vmap(log_network, in_axes=(None, 0, None, None, None,)),
                                    in_axes=(0, 0, None, None, None,))

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
                                             list_l=2,
                                             use_scan=False)

    total_e = calculate_total_energy(local_energy=localenergy)
    total_e_parallel = jax.pmap(total_e)
    e_trial, variance_trial = total_e_parallel(params, subkeys, data)
    e_est, variance_est = total_e_parallel(params, subkeys, data)
    """we need think more about the parallel strategy. So later, we have to modify the shape of weights and branchcut."""
    #weights = jnp.ones(shape=(num_devices * num_hosts, batch_size))
    esigma = jnp.std(e_est)
    dmc_run = dmc_propagate(signed_network=signed_network,
                            log_network=log_network,
                            logabs_f=logabs_f,
                            list_l=2,
                            nelectrons=nelectrons,
                            natoms=natoms,
                            ndim=ndim,
                            batch_size=batch_size,
                            tstep=tstep,
                            nsteps=nsteps,
                            charges=charges,
                            spins=spins,
                            Rn_local=Rn_local,
                            Local_coes=Local_coes,
                            Local_exps=Local_exps,
                            Rn_non_local=Rn_non_local,
                            Non_local_coes=Non_local_coes,
                            Non_local_exps=Non_local_exps)
    branchcut_start = jnp.ones(shape=(num_devices * num_hosts, batch_size)) * 10

    branch_parallel = jax.pmap(branch, in_axes=(0, 0, 0))
    energy_data = jnp.zeros(shape=(nblocks, iterations, batch_size))
    weights_data = primary_weights

    """Start the main loop."""
    time_of_last_ckpt = time.time()
    opt_state = opt_state_ckpt

    """the writer module need be modified. 14.2.2025."""
    train_schema = ['block', 'energy', 'weights_data', 'positions']
    writer_manager = writers.Writer(
        name='DMC_states',
        schema=train_schema,
        directory=ckpt_restore_path,
        iteration_key=None,
        log=False
    )

    correlatedsamples_parallel = jax.pmap(
        jax.vmap(jax.vmap(corrsamples.correlated_samples, in_axes=(None, None, 0)), in_axes=(None, 0, None)),
        in_axes=(None, None, 0))
    pos = data.positions
    newpos = correlatedsamples_parallel(atoms, new_atoms, pos)
    number_new_atoms = new_atoms.shape[0]
    """we also need change the data.new_atoms. 27.3.2025."""

    newpos = jnp.reshape(newpos, (
    number_new_atoms, 1, batch_size, -1))  # 2 is the number of new atoms, 1 is fixed, 4 is the number of batch size

    data.positions = newpos #temporaly, we only take one new configuration. Ok, it is working currently.
    #jax.debug.print("data.positions:{}", data.positions)
    #jax.debug.print("new_atoms:{}", new_atoms)
    temp = jnp.repeat(new_atoms, batch_size, axis=0)
    #jax.debug.print("temp:{}", temp)
    final_new_atoms = jnp.reshape(temp, (2, 1, 4, 2, 3)) # 2 is the number of new configurations. 1 is the number of devices. 4 is the batch size, 2 is the number of atoms, 3 is the dimension.
    #jax.debug.print("final_new_atoms:{}", final_new_atoms)
    data.atoms = final_new_atoms
    #jax.debug.print("data.atoms:{}", data.atoms)
    jacobianWeights_parallel = jax.pmap(
        jax.vmap(jax.vmap(jacobianWeights.weights_jacobian, in_axes=(0, None, None)), in_axes=(None, None, 0)),
        in_axes=(0, None, None))
    weights_new_atoms = jacobianWeights_parallel(pos, atoms, new_atoms)
    wave_x1 = log_network_parallel(params, pos, spins, atoms, charges)
    log_network_parallel_new_atoms = jax.vmap(log_network_parallel, in_axes=(None, None, None, 0, None))
    wave_x2 = log_network_parallel_new_atoms(params, pos, spins, new_atoms, charges)
    ratios = jnp.square(jnp.abs(wave_x1 - wave_x2))
    weights_new_atoms = jnp.reshape(weights_new_atoms, ratios.shape)  # we need be careful about this line. 2 means the number of new configurations.
    weights_final = weights_new_atoms * ratios * batch_size / jnp.sum(weights_new_atoms * ratios, axis=-1, keepdims=True)

    #jax.debug.print("weights_final:{}", weights_final)
    #jax.debug.print("weights:{}", weights)
    #jax.debug.print("primary_weights:{}", primary_weights)
    #jax.debug.print("data:{}", data)
    temp_spins = jnp.repeat(data.spins, 2, axis=0)
    temp_spins = jnp.reshape(temp_spins, (2, 1, 4, -1))
    temp_charges = jnp.repeat(data.charges, 2, axis=0)
    temp_charges = jnp.reshape(temp_charges, (2, 1, 4, -1))
    #jax.debug.print("temp_spins:{}", temp_spins)
    #jax.debug.print("temp_charges:{}", temp_charges)
    data.spins = temp_spins
    data.charges = temp_charges
    """to be continued...27.3.2025.
    forgot to reshape the data. shit."""
    #jax.debug.print("data:{}", data)
    """should we use the first one as the primary weights? 28.3.2025."""
    initial_weights = primary_weights[0][0]
    initial_weights = jnp.reshape(initial_weights, (1, -1))
    initial_weights = jnp.repeat(initial_weights, 2, axis=0)

    initial_weights = jnp.reshape(initial_weights, (2, 1, -1))
    jax.debug.print("initial_weights:{}", initial_weights)
    with writer_manager as writer:
        for block in range(0, nblocks):
            for t in range(t_init, t_init+iterations):
                #jax.debug.print("primary_weights:{}", primary_weights[block, t-t_init,])
                weights = jnp.reshape(primary_weights[block, t-t_init], (-1, batch_size))
                #jax.debug.print("weights:{}", weights)
                dmc_run_parallel = jax.vmap(dmc_run, in_axes=(None, None, nn.AINetData(positions=0, spins=0, atoms=0, charges=0), None, None, None, None))
                energy, new_weights, new_data = dmc_run_parallel(params,
                                                        subkeys,
                                                        data,
                                                        weights,
                                                        branchcut_start * esigma,
                                                        e_trial,
                                                        e_est,)
                data = new_data
                #weights = new_weights
                #jax.debug.print("new_data:{}", new_data)
                #jax.debug.print("new_weights:{}", new_weights)
                #jax.debug.print("energy:{}", energy)
                """we still need calculate the secondary weights. And understand well what is the N_proj."""
                #energy = jnp.reshape(energy, batch_size)
                secondary_weights = new_weights/weights
                jax.debug.print("secondary_weights:{}", secondary_weights)
                initial_weights = initial_weights * secondary_weights
                jax.debug.print("initial_weights:{}", initial_weights)
                jax.debug.print("secondary_weights:{}", secondary_weights)
                jax.debug.print("energy:{}", energy)
                output_energy = jnp.mean(energy * initial_weights, axis=-1, keepdims=True)
                """now, we got another problem. How do we need change the e_trial or something else? 28.3.2025."""
                jax.debug.print("output_energy:{}", output_energy)


            '''e_est = estimate_energy(energy_data, weights_data)
           
            """for the energy store part, we need rewrite it."""
            jax.debug.print("e_est:{}", e_est)
            logging_str = ('Block %05d:', '%03.4f E_h,')
            logging_args = block, t, e_est,
            logging.info(logging_str, *logging_args)
            if time.time() - time_of_last_ckpt > save_frequency * 60:
                """np.savez cannot save inhomogeneous array. So, we have to use the following line to convert the format of the arrays."""
                save_params = np.asarray(params)
                save_opt_state = np.asarray(opt_state, dtype=object)
                checkpoint.save(ckpt_restore_path, block, data, save_params, save_opt_state)
                time_of_last_ckpt = time.time()

            weights, newindices = branch_parallel(data, weights, subkeys)
            x1 = data.positions
            x2 = []
            for i in range(len(x1)):
                unique, counts = jnp.unique(newindices[i], return_counts=True)
                temp = x1[i][unique]
                """here, we need think what if the walker is killed. 13.2.2025.
                we need add one more walker into the configurations.
                But now the problem is how to generate the new walker?
                Actually, this part belong to branch function. Currently, it is not a good solution. Maybe improve it later."""
                if len(unique) < batch_size:
                    n = batch_size - len(unique)
                    extra_walkers = temp[-1] + jax.random.uniform(key, (n, nelectrons * ndim))
                    temp = jnp.concatenate([temp, extra_walkers], axis=0)
                    logging.info("max branches $i and number of walkers killed $i:", jnp.max(counts), n)
                else:
                    logging.info("max branches $i and number of walkers killed $i:", jnp.max(counts), 0)

                x2.append(temp)

            x2 = jnp.array(x2)
            data = nn.AINetData(**(dict(data) | {'positions': x2}))

            """leave this to tomorrow. 13.2.2025. we also need update the branchcut."""
            e_trial = e_est - feedback * jnp.log(jnp.mean(weights)).real

            writer_kwargs = {
                'block': block,
                'energy': np.asarray(e_est),
                'weights_data': np.asarray(weights_data),
                'positions': np.asarray(x2),
            }
            writer.write(block, **writer_kwargs)
        """we turn to the energy summary part."""
        '''