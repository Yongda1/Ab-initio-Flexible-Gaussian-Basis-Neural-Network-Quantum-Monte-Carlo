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
from AIQMCrelease3.spin_indices import spin_indices_h

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
    """here, the restore calculation can be read from save path or restore path."""
    ckpt_restore_filename = (checkpoint.find_last_checkpoint(ckpt_save_path) or
                             checkpoint.find_last_checkpoint(ckpt_restore_path))
    if ckpt_restore_filename:
        (t_init,
         data,
         params,
         opt_state_ckpt,) = checkpoint.restore(ckpt_restore_filename, host_batch_size)
        #jax.debug.print("data:{}", data)
    else:
        raise ValueError('DMC must use the wave function from VMC!')

    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins, nelectrons=nelectrons )
    spin_up_indices, spin_down_indices = spin_indices_h(spins)
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
    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    logabs_f = utils.select_output(signed_network, 1)

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
    jax.debug.print("e_trial:{}", e_trial)
    jax.debug.print("variance_trial:{}", variance_trial)
    jax.debug.print("e_est:{}", e_est)
    jax.debug.print("variance_est:{}", variance_est)
    """we need think more about the parallel strategy. So later, we have to modify the shape of weights and branchcut."""
    weights = jnp.ones(shape=(num_devices * num_hosts, int(batch_size / (num_devices * num_hosts))))
    esigma = jnp.std(e_est)
    dmc_run = dmc_propagate(signed_network=signed_network,
                            log_network=log_network,
                            logabs_f=logabs_f,
                            list_l=2,
                            nelectrons=nelectrons,
                            natoms=natoms,
                            ndim=ndim,
                            batch_size=int(batch_size/(num_devices * num_hosts)), #we also need to write the number of walker on each GPU.
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
    branchcut_start = jnp.ones(shape=(num_devices * num_hosts, int(batch_size / (num_devices * num_hosts)))) * 10
    branch_parallel = jax.pmap(branch, in_axes=(0, 0, 0))
    energy_data = jnp.zeros(shape=(nblocks, iterations, batch_size))
    weights_data = jnp.zeros(shape=(nblocks, iterations, batch_size))
    #jax.debug.print("energy_data:{}", energy_data)

    """Start the main loop."""
    time_of_last_ckpt = time.time()
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

    with writer_manager as writer:
        for block in range(0, nblocks):
            for t in range(t_init, t_init+iterations):
                #jax.debug.print("weights:{}", weights)
                #jax.debug.print("branchcut_start * esigma:{}", branchcut_start* esigma)
                #jax.debug.print("e_trial:{}", e_trial)
                #jax.debug.print("e_est:{}", e_est)
                energy, new_weights, new_data = dmc_run(params,
                                                        subkeys,
                                                        data,
                                                        weights,
                                                        branchcut_start * esigma,
                                                        e_trial,
                                                        e_est,)
                #jax.debug.print("energy:{}", energy)
                #jax.debug.print("new_weights:{}", new_weights)
                #jax.debug.print("new_data:{}", new_data)

                data = new_data
                weights = new_weights
                energy = jnp.reshape(energy, batch_size)
                weights_step = jnp.reshape(weights, batch_size)
                #temp = energy_data[block]
                jax.debug.print("block:{}", block)
                jax.debug.print("t:{}", t-t_init)
                #jax.debug.print("weights:{}", weights)
                temp_energy = energy_data[block].at[t-t_init].set(energy.real)
                temp_weights = weights_data[block].at[t-t_init].set(weights_step)
                energy_data = energy_data.at[block].set(temp_energy)
                weights_data = weights_data.at[block].set(temp_weights)


            e_est = estimate_energy(energy_data, weights_data)
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
            #jax.debug.print("x1:{}", x1)
            #jax.debug.print("newindices:{}", newindices)
            x2 = []
            """we need change the parallel rules here, 4.4.2025. Fortunately, the above code is running smoothly.
            after we finish the pseudopotential part, we start to rewrite the following lines. 9.4.2025."""
            #x1 = jnp.reshape(x1, (batch_size, -1))
            #jax.debug.print("x1:{}", x1)
            #newindices = jnp.reshape(newindices, -1)
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
            #jax.debug.print("x2:{}", x2)
            x2 = jnp.reshape(x2, (num_devices * num_hosts, int(batch_size/(num_devices * num_hosts)), -1))
            #jax.debug.print("x2:{}", x2)
            data = nn.AINetData(**(dict(data) | {'positions': x2}))

            """leave this to tomorrow. 13.2.2025. we also need update the branchcut."""
            e_trial = e_est - feedback * jnp.log(jnp.mean(weights)).real

            writer_kwargs = {
                'block': block,
                'energy': np.asarray(e_est),
                #'weights_data': np.asarray(weights_data),
                'positions': np.asarray(x2),
            }
            writer.write(block, **writer_kwargs)
