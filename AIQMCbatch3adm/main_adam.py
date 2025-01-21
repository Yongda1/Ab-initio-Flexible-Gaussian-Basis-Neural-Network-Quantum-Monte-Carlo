"""here, we learn how to use kfac and test it."""


import time
from typing import Optional, Tuple, Union
from absl import logging
from AIQMCbatch3adm import nn
import jax
import numpy as np
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import optax
from typing_extensions import Protocol
import chex
from AIQMCbatch3adm import mcstep
from AIQMCbatch3adm import loss as qmc_loss_functions
from AIQMCbatch3adm import constants
from AIQMCbatch3adm import hamiltonian
from AIQMCbatch3adm import curvature_tags_and_blocks
import functools




def _assign_spin_configuration(nalpha: int, nbeta: int, batch_size: int=1) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))



structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['C', 'O', 'O']
atoms = jnp.array([[1.33, 1.0, 1.0], [0.0, 1.0, 1.0], [2.66, 1.0, 1.0]])
charges = jnp.array([4.0, 6.0, 6.0])
#pos = jnp.array([1.5, 1.5, 1.5, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 0.5])


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
    spins = electrons
    spins = jnp.repeat(jnp.reshape(spins, (1, -1)), batch_size, )
    return electrons_positions_batch, spins



OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]

OptUpdateResults = Tuple[nn.ParamTree, Optional[OptimizerState], jnp.array, Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):
    def __call__(self, params: nn.ParamTree, data: nn.AINetData, opt_state: optax.OptState, key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters accordingly."""


StepResults = Tuple[nn.AINetData, nn.ParamTree, Optional[optax.OptState], jnp.ndarray, qmc_loss_functions.AuxiliaryLossData,]


class Step(Protocol):
    def __call__(self, data: nn.AINetData, params: nn.ParamTree, state: OptimizerState, key: chex.PRNGKey) \
            -> StepResults:
        """Performs one set of MCMC moves and an optimization step."""

    
    """Returns an OptUpdate function for performing a parameter update.
    So far ,we have not solved the spin configuration problem yet. But we got one more task about writing the loss function.
    Let's go back to main.py 14.08.2024. We cannot finished all functions now. Because we need guarrante all input data format fixed and
    Loss.py, hamiltonian_wrong.py, utils.py and pseudopotential.py form an entire part. So, next fews steps, we need move stepy by step."""


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossAINet, optimizer: optax.GradientTransformation) -> OptUpdate:
    loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

    def opt_update(params: nn.ParamTree, data: nn.AINetData, opt_state: Optional[optax.OptState], key: chex.PRNGKey) -> OptUpdateResults:
        (loss, aux_data), grad = loss_and_grad(params, key, data)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux_data
    return opt_update


def make_training_step(mcmcstep, optimizer_step: OptUpdate) -> Step:

    @functools.partial(constants.pmap, donate_argnums=(0, 2))
    def step(data: nn.AINetData,
             params: nn.ParamTree,
             state: Optional[optax.OptState],
             key: chex.PRNGKey,) -> StepResults:
        mcmc_key, loss_key = jax.random.split(key, num=2)
        data = mcmcstep(params, data, mcmc_key)
        new_params, new_state, loss, aux_data = optimizer_step(params, data, state, loss_key)
        return data, new_params, new_state, loss, aux_data
    return step


def main(batch_size=4, structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]]), atoms=atoms, charges=charges, nelectrons=16, natoms =3, ndim=3,
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
    key = jax.random.PRNGKey(seed)
    """we already write the envelope function in the nn_wrong.py."""
    network = nn.make_ai_net(ndim=3, natoms=3, nelectrons=16, num_angular=4, charges=charges, full_det=True)
    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j*phase


    pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                                electrons=jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
                                batch_size=host_batch_size, init_width=1.5)
    batch_pos = jnp.reshape(pos, data_shape+(-1,))
    batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
    #jax.debug.print("batch_pos:{}", batch_pos)
    data = nn.AINetData(positions=batch_pos, atoms=batch_atoms, charges=batch_charges)
    '''--------------Main training-------------'''
    #jax.debug.print("data:{}", data)
    mc_step = mcstep.make_mc_step(signed_network, nsteps=10)
    localenergy = hamiltonian.local_energy(f=signed_network, batch_size=batch_size, natoms=natoms, nelectrons=nelectrons)
    """so far, we have not constructed the pp module. Currently, we only execute all electrons calculation.  """
    evaluate_loss = qmc_loss_functions.make_loss(log_network, local_energy=localenergy)
    """18.10.2024, we will continue later."""

    #we have some problems about kfac optimizer. We dont understand the mechanism behind it. Leave more time for it.
    def learning_rate_schedule(t_: jnp.array, rate=0.05, delay=1.0, decay=10000) -> jnp.array:
        return rate * jnp.power(1.0/(1.0 + (t_/delay)), decay)

    """the setup of adam optimzier."""
    optimizer = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1.0e-9, eps_root=0.0),
                            optax.scale_by_schedule(learning_rate_schedule),
                            optax.scale(-1.))

    opt_state = jax.pmap(optimizer.init)(params)
    step = make_training_step(mcmcstep=mc_step, optimizer_step=make_opt_update_step(evaluate_loss, optimizer))
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    '''
    we comment on this part for the test of t-moves.
    """main training loop"""
    for t in range(0, iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        jax.debug.print("data:{}", data)
        jax.debug.print("params:{}", params)
        jax.debug.print("opt_state:{}", opt_state)
        jax.debug.print("subkeys:{}", subkeys)
        data, params, opt_state, loss, aux_data, = step(data, params, opt_state, subkeys)

    '''
    return signed_network, data, params, log_network


#output = main()