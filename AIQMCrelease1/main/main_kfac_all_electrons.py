"""we test kfac for the all electrons' calculation here."""

import time
from typing import Optional, Tuple, Union
from absl import logging
import chex
import jax
import kfac_jax
import optax
import numpy as np
import jax.numpy as jnp
from typing_extensions import Protocol
from jax.experimental import multihost_utils
from AIQMCrelease1.wavefunction import nn
#from AIQMCrelease1.MonteCarloSample import mcstep
from AIQMCrelease1.Loss import pploss as qmc_loss_functions
from AIQMCrelease1 import constants
from AIQMCrelease1.Energy import hamiltonian
from AIQMCrelease1 import curvature_tags_and_blocks
import functools


def _assign_spin_configuration(nalpha: int, nbeta: int, batch_size: int = 1) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(key, structure: jnp.array, atoms: jnp.array, charges: jnp.array,
                   electrons: jnp.array, batch_size: int, init_width: float) -> Tuple[jnp.array, jnp.array]:
    """Initializes electron positions around each atom.
    structure: the crystal structure, (lattice parameters, cell size).
    atoms: positions of the atoms.
    electrons: the array of alpha and beta electrons, i.e. spin configurations.
    batch_size: total number of Monte Carlo configurations to generate across all devices.
    init_width: width of atom-centred Gaussian used to generate initial electron configurations.
    This function needs be finished."""
    electrons_positions_batch = []
    for _ in range(batch_size):
        for i in range(len(atoms)):
            electrons_positions_batch.append(np.tile(atoms[i], int(charges[i])))

    """the following line has some problems. But it is still working now. We can make it better later."""
    electrons_positions_batch = np.hstack(np.array(electrons_positions_batch))
    electrons_positions_batch = jnp.reshape(jnp.array(electrons_positions_batch), (batch_size, -1))
    key, subkey = jax.random.split(key, num=2)
    electrons_positions_batch += (jax.random.normal(subkey, shape=electrons_positions_batch.shape) * init_width)
    "we need think about this. We need assign the spin configurations to electrons.12.08.2024."
    spins_no_batch = electrons
    # spins = jnp.repeat(jnp.reshape(spins, (1, -1)), batch_size, )
    return electrons_positions_batch, spins_no_batch


OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]

OptUpdateResults = Tuple[
    nn.ParamTree, Optional[OptimizerState], jnp.array, Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):
    def __call__(self, params: nn.ParamTree, data: nn.AINetData, opt_state: optax.OptState,
                 key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters accordingly."""


StepResults = Tuple[nn.ParamTree, Optional[optax.OptState], jnp.ndarray, qmc_loss_functions.AuxiliaryLossData,]


class Step(Protocol):
    def __call__(self, data: nn.AINetData, params: nn.ParamTree, state: OptimizerState, key: chex.PRNGKey) \
            -> StepResults:
        """Performs one set of MCMC moves and an optimization step."""


def make_kfac_training_step(damping: float, optimizer: kfac_jax.Optimizer,) -> Step:
    """Factory to create training step for Kfac optimizers."""
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(damping))

    def step(data: nn.AINetData, params: nn.ParamTree, state: kfac_jax.Optimizer.State, key: chex.PRNGKey) \
            -> StepResults:

        mc_keys, loss_keys = kfac_jax.utils.p_split(key)
        jax.debug.print("--------------------")
        jax.debug.print("data:{}", data)
        jax.debug.print("--------------------")
        new_params, new_state, stats = optimizer.step(params=params, state=state, rng=loss_keys, batch=data,
                                                      momentum=shared_mom, damping=shared_damping)
        return new_params, new_state, stats['loss'], stats['aux']

    return step


def main(atoms: jnp.array,
         charges: jnp.array,
         spins: jnp.array,
         tstep: float,
         nelectrons: int,
         natoms: int,
         ndim: int,
         batch_size: int,
         iterations: int,
         structure: jnp.array,):
    print("Quantum Monte Carlo Start running")
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
    key, subkey = jax.random.split(key)
    data_shape = (num_devices, device_batch_size)
    """Here, we use [None, ...] to enlarge one dimension of the array 'atoms'. """
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
    pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                                electrons=spins,
                                batch_size=host_batch_size, init_width=1.0)

    batch_pos = jnp.reshape(pos, data_shape + (-1,))
    batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
    data = nn.AINetData(positions=batch_pos, atoms=batch_atoms, charges=batch_charges)
    #jax.debug.print("batch_charges:{}", batch_charges)
    spins_total = jnp.reshape(spins, (1, nelectrons)) * jnp.reshape(spins, (nelectrons, 1))
    spins_total_uptriangle = jnp.triu(spins_total, k=1)
    sample = jnp.zeros_like(a=spins_total_uptriangle)
    parallel = jnp.where(spins_total_uptriangle > sample, spins_total_uptriangle, sample)
    antiparallel = jnp.where(spins_total_uptriangle < sample, spins_total_uptriangle, sample)
    parallel_indices = jnp.nonzero(parallel)
    antiparallel_indices = jnp.nonzero(antiparallel)
    parallel_indices = jnp.array(parallel_indices)
    antiparallel_indices = jnp.array(antiparallel_indices)
    n_parallel = len(parallel_indices[0])
    n_antiparallel = len(antiparallel_indices[0])
    jax.debug.print("n_parallel:{}", n_parallel)

    """we already write the envelope function in the nn.py."""
    network = nn.make_ai_net(ndim=ndim,
                             natoms=natoms,
                             nelectrons=nelectrons,
                             num_angular=3,
                             n_parallel=n_parallel,
                             n_antiparallel=n_antiparallel,
                             parallel_indices=parallel_indices,
                             antiparallel_indices=antiparallel_indices,
                             charges=charges,
                             full_det=True)

    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    print('''--------------Main training-------------''')
    print('''--------------Start Monte Carlo process------------''')
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    '''
    mc_step = mcstep.main_monte_carlo(f=signed_network,
                                      tstep=tstep,
                                      ndim=ndim,
                                      nelectrons=nelectrons)
    """to be continued...24.12.2024."""

    """we need add the pseudopotential module into the hamiltonian module."""
    """be aware of the list_l, this variable means the angular momentum function max indice in pseudopotential file.7.1.2025."""

    localenergy = hamiltonian.local_energy(
        signed_network=signed_network,
        natoms=natoms,
        nelectrons=nelectrons,
        ndim=ndim)

    evaluate_loss = qmc_loss_functions.make_loss(log_network, local_energy=localenergy)

    def learning_rate_schedule(t_: jnp.array, rate=0.05, delay=1.0, decay=10000) -> jnp.array:
        return rate * jnp.power(1.0/(1.0 + (t_/delay)), decay)

    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

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
    )  # this line to be done!!! we need read the paper of Kfac.

    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params=params, rng=subkeys, batch=data)
    # jax.debug.print("opt_state:{}", opt_state)
    """the default option is Kfac. It could be the problem of parallirization. 23.10.2024."""
    if isinstance(optimizer, kfac_jax.Optimizer):
        step_kfac = make_kfac_training_step(damping=0.001, optimizer=optimizer)

    """main training loop"""
    for t in range(0, iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        data = mc_step(params, data)
        # jax.debug.print("data_before:{}", data)
        # jax.debug.print("params:{}", params)
        # sharded_key1, subkeys1 = kfac_jax.utils.p_split(subkeys)
        params, opt_state, loss, aux_data, = step_kfac(data=data, params=params, state=opt_state, key=subkeys, )
        loss = loss[0]
    '''
    return signed_network, data, params, log_network