"""This is the main part of AIQMC."""
import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union
from absl import logging
import chex
from AIQMC import envelopes
from AIQMC import nn
from AIQMC import mcstep
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from hamiltonian import local_energy
from typing_extensions import Protocol


def _assign_spin_configuration(nalpha: int, nbeta: int, batch_size: int=1) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))


#electrons = _assign_spin_configuration(nalpha=2, nbeta=2, batch_size=3)


structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
atoms = jnp.array([[1, 1, 1], [0.2, 0.2, 0.2]])
charges = jnp.array([2, 2])
pos = jnp.array([1.5, 1.5, 1.5, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 0.5])


def init_electrons(key, structure: jnp.ndarray, atoms: jnp.ndarray, charges: jnp.ndarray,
                   electrons: jnp.ndarray, batch_size: int, init_width: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initializes electron positions around each atom.
    structure: the crystal structure, (lattice parameters, cell size).
    atoms: positions of the atoms.
    electrons: the array of number of alpha and beta electrons, i.e. spin configurations.
    batch_size: total number of Monte Carlo configurations to generate across all devices.
    init_width: width of atom-centred Gaussian used to generate initial electron configurations."""
    electrons_positions_batch = []
    for _ in range(batch_size):
        electron_positions = []
        for i in range(len(atoms)):
            electron_positions.append(jnp.tile(atoms[i], charges[i]))
        electrons_positions_batch.append(electron_positions)
    electrons_positions_batch = jnp.reshape(jnp.array(electrons_positions_batch), (batch_size, len(charges), -1, 3))
    key, subkey = jax.random.split(key, num=2)
    electrons_positions_batch += (jax.random.normal(subkey, shape=electrons_positions_batch.shape) * init_width)
    electrons_positions_batch = jnp.reshape(electrons_positions_batch, (batch_size, 12))
    "we need think about this. We need assign the spin configurations to electrons.12.08.2024."
    electrons = jnp.repeat(electrons, batch_size, axis=0)
    return electrons_positions_batch, electrons


#key = jax.random.PRNGKey(seed=1)
#a = init_electrons(key=key, structure=structure, atoms=atoms, charges=charges, electrons=electrons, batch_size=3, init_width=0.5)

OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[nn.ParamTree, Optional[OptimizerState], jnp.ndarray]


class OptUpdate(Protocol):
    def __call__(self, params: nn.ParamTree, data: nn.AINetData, opt_state: optax.OptState, key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters accordingly."""


StepResults = Tuple[nn.AINetData, nn.ParamTree, Optional[optax.OptState], jnp.ndarray, jnp.ndarray]


class Step(Protocol):
    def __call__(self, data: nn.AINetData, params: nn.ParamTree, state: OptimizerState, key: chex.PRNGKey, mcmc_width: jnp.ndarray) -> StepResults:
        """Performs one set of MCMC moves and an optimization step."""


#def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn, optimizer: optax.GradientTransformation) -> OptUpdate:
    """Returns an OptUpdate function for performing a parameter update.
    So far ,we have not solved the spin configuration problem yet. But we got one more task about writing the loss function.
    Let's go back to main.py 14.08.2024. We cannot finished all functions now. Because we need guarrante all input data format fixed and
    Loss.py, hamiltonian.py, utils.py and pseudopotential.py form an entire part. So, next fews steps, we need move stepy by step."""


"""we can start the main function first, the solve every module we need in the calculation."""


def main(batch_size=4, structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]]), atoms=jnp.array([[0, 0, 0], [0.2, 0.2, 0.2]]), charges=jnp.array([2, 2]), nelectrons=4, ndim=3):
    num_devices = jax.local_device_count() #the amount of GPU per host
    num_hosts = jax.device_count() // num_devices #the amount of host
    #print("num_devices", num_devices)
    #print("num_hosts", num_hosts)
    logging.info('Start QMC with $i devices per host, across %i hosts.', num_devices, num_hosts)
    if batch_size % (num_devices * num_hosts) != 0:
        raise ValueError('Batch size must be divisible by number of devices!')
    host_batch_size = batch_size // num_hosts # how many configurations we put on one host
    device_batch_size = host_batch_size // num_devices # how many configurations we put on one GPU
    data_shape = (num_devices, device_batch_size)
    #print("data_shape", data_shape)
    """we continue tommorrow, 14.08.2024.
    Here, we use [None, ...] to enlarge one dimension of the array 'atoms'. """
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    #print("batch_atoms", batch_atoms)
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
    #print("batch_atoms", batch_atoms)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    #print("batch_charges", batch_charges)
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
    #print("batch_charges", batch_charges)
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    key = jax.random.PRNGKey(seed)
    #print("key", key)
    feature_layer1 = nn.make_ainet_features(natoms=2, nelectrons=4, ndim=3)
    """we already write the envelope function in the nn.py."""
    network = nn.make_ai_net(charges=jnp.array([2, 2]), ndim=3, full_det=True)
    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    #print("params", params)
    batch_params = kfac_jax.utils.replicate_all_local_devices(params)
    #print("params", params)
    '''here, we have one problem about complex number orbitals. So far, we have not deal with it.
    16.08.2024, we solve the complex number problem later.
    For the complex number problem, we dont need change any part of the nn.py. Because we have angular momentum functions to generate complex orbitals naturally.
    If we have to introduce the complex number later, we can use two envelope layers, one as real part, the other one as imaginary part.
    So the output dimensions of envelope layer will be two times. 16.08.2024.'''
    signed_network = network.apply
    phase_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[0]
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    "notes: how many 0 in the in_axes? It depends on the input parameters."
    "Notes: here, batch_network is a function but not a class."
    """we have more problems about this batch calculation at 21.08.2021."""
    """The good news is that we probably know what is happening to the parallel control over the function logabs_network. 
    The bad news is that i dont know how to solve it completely. Currently, I only use one way to make it work temporarily.
    We need improve it later. 22.08.2024."""
    phase_network = jax.vmap(phase_network, in_axes=(None, 1, 1, 1), out_axes=0)
    batch_network = jax.vmap(logabs_network, in_axes=(None, 1, 1, 1), out_axes=0)
    "for the complex wave function, we need a new function."
    "This is correct. no problem. 19.08.2024."
    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges, electrons=jnp.array([[1, 0], [1, 0]]), batch_size=host_batch_size, init_width=0.1)
    """this operation means add one extra dimension to the array."""
    batch_pos = jnp.reshape(pos, data_shape+(-1,))
    """here, we need be sure that the array pos must be compatible with the input of AInet and hamiltonian."""
    batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
    spins = jnp.reshape(spins, data_shape+(-1,))
    batch_spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    print("batch_pos", batch_pos)
    print("batch_atoms", batch_atoms)
    print("batch_charges", batch_charges)
    """here, we tested the parallel calculation method. Currently, after our heavy effort, it is working well."""
    #test_output_no_batch = logabs_network(params=params, pos=batch_pos[0][1], atoms=atoms, charges=charges)
    #test_output = batch_network(batch_params, batch_pos, batch_atoms, batch_charges)
    #jax.debug.print("test_output:{}", test_output)
    #test_output_no_batch = signed_network(params=params, pos=batch_pos[0][1], atoms=atoms, charges=charges)
    data = nn.AINetData(positions=batch_pos, spins=batch_spins, atoms=batch_atoms, charges=batch_charges)
    """here, we have one problem about the format of pos, spins, batch_atoms, batch_charges.
    These formats can be easily changed in nn.py. so, before we do this, we need know which format should be used in the loss function 16.08.2024."""
    #print("data.positions", data.positions.shape)
    "currently, we dont need check points. So, we ignore this part."
    '''--------------Main training-------------'''
    #Construct MC step
    mc_step = mcstep.make_mc_step(phase_network, batch_network, signed_network)
    '''Construct loss and optimizer, local energy calculation. we are gonna deal with it at 28.08.2024.'''
    localenergy = local_energy(f=signed_network, complex_number=True) 
    """so far, we have not constructed the pp module. Currently, we only execute all electrons calculation.  """
    return signed_network, data, batch_params, phase_network, batch_network, mc_step, localenergy


output = main()