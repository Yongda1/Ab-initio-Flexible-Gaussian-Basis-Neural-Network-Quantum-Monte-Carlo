"""here, we learn how to use kfac and test it."""


import time
from typing import Optional, Tuple, Union
from absl import logging
from AIQMCbatch2 import nn
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import optax
from typing_extensions import Protocol
import chex
from AIQMCbatch2 import mcstep
from AIQMCbatch2 import loss as qmc_loss_functions
from AIQMCbatch2 import constants
from AIQMCbatch2 import hamiltonian
from AIQMCbatch2 import curvature_tags_and_blocks
import functools




def _assign_spin_configuration(nalpha: int, nbeta: int, batch_size: int=1) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))



structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
atoms = jnp.array([[1.0, 1.0, 1.0], [0.2, 0.2, 0.2]])
charges = jnp.array([2.0, 2.0])
pos = jnp.array([1.5, 1.5, 1.5, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 0.5])


def init_electrons(key, structure: jnp.ndarray, atoms: jnp.ndarray, charges: jnp.ndarray,
                   electrons: jnp.ndarray, batch_size: int, init_width: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initializes electron positions around each atom.
    structure: the crystal structure, (lattice parameters, cell size).
    atoms: positions of the atoms.
    electrons: the array of number of alpha and beta electrons, i.e. spin configurations.
    batch_size: total number of Monte Carlo configurations to generate across all devices.
    init_width: width of atom-centred Gaussian used to generate initial electron configurations.
    This function needs be finished."""
    electrons_positions_batch = []
    for _ in range(batch_size):
        electron_positions = []
        for i in range(len(atoms)):
            electron_positions.append(jnp.tile(atoms[i], int(charges[i])))
        electrons_positions_batch.append(electron_positions)
    electrons_positions_batch = jnp.reshape(jnp.array(electrons_positions_batch), (batch_size, len(charges), -1, 3))
    key, subkey = jax.random.split(key, num=2)
    electrons_positions_batch += (jax.random.normal(subkey, shape=electrons_positions_batch.shape) * init_width)
    electrons_positions_batch = jnp.reshape(electrons_positions_batch, (batch_size, 12))
    "we need think about this. We need assign the spin configurations to electrons.12.08.2024."
    electrons = jnp.repeat(electrons, batch_size, axis=0)
    return electrons_positions_batch, electrons



OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
'''
OptUpdateResults = Tuple[nn.ParamTree, Optional[OptimizerState], jnp.array, Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):
    def __call__(self, params: nn.ParamTree, data: nn.AINetData, opt_state: optax.OptState, key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters accordingly."""

'''
StepResults = Tuple[nn.ParamTree, Optional[optax.OptState], jnp.ndarray, qmc_loss_functions.AuxiliaryLossData,]


class Step(Protocol):
    def __call__(self, data: nn.AINetData, params: nn.ParamTree, state: OptimizerState, key: chex.PRNGKey) \
            -> StepResults:
        """Performs one set of MCMC moves and an optimization step."""

    
    """Returns an OptUpdate function for performing a parameter update.
    So far ,we have not solved the spin configuration problem yet. But we got one more task about writing the loss function.
    Let's go back to main.py 14.08.2024. We cannot finished all functions now. Because we need guarrante all input data format fixed and
    Loss.py, hamiltonian_wrong.py, utils.py and pseudopotential.py form an entire part. So, next fews steps, we need move stepy by step."""


def make_kfac_training_step(damping: float, optimizer: kfac_jax.Optimizer,) -> Step:
    """Factory to create training step for Kfac optimizers."""
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(damping))

    def step(data: nn.AINetData, params: nn.ParamTree, state: kfac_jax.Optimizer.State, key: chex.PRNGKey) \
            -> StepResults:

        mc_keys, loss_keys = kfac_jax.utils.p_split(key)
        jax.debug.print("--------------------")
        #jax.debug.print("params:{}", params)
        jax.debug.print("data:{}", data)
        jax.debug.print("loss_keys:{}", loss_keys)
        jax.debug.print("shared_mom:{}", shared_mom)
        jax.debug.print("shared_damping:{}", shared_damping)
        jax.debug.print("--------------------")
        new_params, new_state, stats = optimizer.step(params=params, state=state, rng=loss_keys, batch=data,
                                                      momentum=shared_mom, damping=shared_damping)
        return new_params, new_state, stats['loss'], stats['aux']

    return step

"""we can start the main function first, the solve every module we need in the calculation."""


def main(batch_size=4, structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]]), atoms=jnp.array([[0, 0, 0], [0.2, 0.2, 0.2]]), charges=jnp.array([2.0, 2.0]), nelectrons=4, ndim=3,
         iterations=1):
    num_devices = jax.local_device_count() #the amount of GPU per host
    num_hosts = jax.device_count() // num_devices #the amount of host
    jax.debug.print("num_devices:{}", num_devices)
    jax.debug.print("num_hosts:{}", num_hosts)
    logging.info('Start QMC with $i devices per host, across %i hosts.', num_devices, num_hosts)
    if batch_size % (num_devices * num_hosts) != 0:
        raise ValueError('Batch size must be divisible by number of devices!')
    host_batch_size = batch_size // num_hosts # how many configurations we put on one host
    device_batch_size = host_batch_size // num_devices # how many configurations we put on one GPU
    data_shape = (num_devices, device_batch_size)
    """Here, we use [None, ...] to enlarge one dimension of the array 'atoms'. """
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    #jax.debug.print("seed:{}", seed)
    key = jax.random.PRNGKey(seed)
    #feature_layer1 = nn.make_ainet_features(natoms=2, nelectrons=4, ndim=3)
    """we already write the envelope function in the nn_wrong.py."""
    network = nn.make_ai_net(ndim=3, full_det=True)
    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j*phase


    pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                                electrons=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
                                batch_size=host_batch_size, init_width=0.1)
    batch_pos = jnp.reshape(pos, data_shape+(-1,))
    batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
    data = nn.AINetData(positions=batch_pos, atoms=batch_atoms, charges=batch_charges)
    '''--------------Main training-------------'''
    #Construct MC step

    mc_step = mcstep.make_mc_step(signed_network, nsteps=10)
    localenergy = hamiltonian.local_energy(f=signed_network)
    """so far, we have not constructed the pp module. Currently, we only execute all electrons calculation.  """
    evaluate_loss = qmc_loss_functions.make_loss(log_network, local_energy=localenergy)
    """18.10.2024, we will continue later."""

    #we have some problems about kfac optimizer. We dont understand the mechanism behind it. Leave more time for it.
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
                                    ) #this line to be done!!! we need read the paper of Kfac.


    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params=params, rng=subkeys, batch=data)
    #jax.debug.print("opt_state:{}", opt_state)
    """the default option is Kfac. It could be the problem of parallirization. 23.10.2024."""
    if isinstance(optimizer, kfac_jax.Optimizer):
        step_kfac = make_kfac_training_step(damping=0.001, optimizer=optimizer)

    """main training loop"""
    for t in range(0, iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        mcstep_pmap = constants.pmap(mc_step, donate_argnums=1)
        data = mcstep_pmap(params, data, subkeys)
        #jax.debug.print("data_before:{}", data)
        #jax.debug.print("params:{}", params)
        #sharded_key1, subkeys1 = kfac_jax.utils.p_split(subkeys)
        params, opt_state, loss, aux_data, = step_kfac(data=data, params=params, state=opt_state, key=subkeys, )
        loss = loss[0]


    #return signed_network, data, params, log_network


output = main()