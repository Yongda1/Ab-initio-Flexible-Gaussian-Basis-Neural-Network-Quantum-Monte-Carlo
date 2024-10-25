"""This is the main part of AIQMC."""
import functools
import time
from typing import Optional, Tuple, Union
from absl import logging
import chex
from AIQMC import nn
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import optax
#from hamiltonian import local_energy
from typing_extensions import Protocol
#from AIQMC import loss as qmc_loss_function
from AIQMC import constants
#from AIQMC import curvature_tags_and_blocks
#from AIQMC import mcstep


def _assign_spin_configuration(nalpha: int, nbeta: int, batch_size: int=1) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))


#electrons = _assign_spin_configuration(nalpha=2, nbeta=2, batch_size=3)


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
    init_width: width of atom-centred Gaussian used to generate initial electron configurations."""
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


    """Returns an OptUpdate function for performing a parameter update.
    So far ,we have not solved the spin configuration problem yet. But we got one more task about writing the loss function.
    Let's go back to main.py 14.08.2024. We cannot finished all functions now. Because we need guarrante all input data format fixed and
    Loss.py, hamiltonian.py, utils.py and pseudopotential.py form an entire part. So, next fews steps, we need move stepy by step."""



def make_kfac_training_step(mc_step, damping:float, optimizer: kfac_jax.Optimizer, reset_if_nan: bool = False) -> Step:
    """Factory to create training step for Kfac optimizers."""
    mc_step = constants.pmap(mc_step, donate_argums=1)
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(damping))
    copy_tree = constants.pmap(functools.partial(jax.tree_util.tree_map, lambda x: (1.0 * x).astype(x.dtype)))

    def step(data: nn.AINetData, params: nn.ParamTree, state: kfac_jax.Optimizer.State, key: chex.PRNGKey) -> StepResults:
        mc_keys, loss_keys = kfac_jax.utils.p_split(key)
        data = mc_step(params, data, mc_keys)
        new_params, new_state, stats = optimizer.step(params=params, state=state, rng=loss_keys, batch=data, momentum=shared_mom, damping=shared_damping)
        return data, new_params, new_state, stats['loss'], stats['aux']

    return step

"""we can start the main function first, the solve every module we need in the calculation."""


def main(batch_size=4, structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]]), atoms=jnp.array([[0, 0, 0], [0.2, 0.2, 0.2]]), charges=jnp.array([2.0, 2.0]), nelectrons=4, ndim=3,
         iterations=10):
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
    #print("data_shape", data_shape)
    """we continue tommorrow, 14.08.2024.
    Here, we use [None, ...] to enlarge one dimension of the array 'atoms'. """
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    #jax.debug.print("batch_atoms:{}", batch_atoms)
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
    jax.debug.print("batch_atoms:{}", batch_atoms)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    #jax.debug.print("batch_charges:{}", batch_charges)
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
    jax.debug.print("batch_charges:{}", batch_charges)
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
    #phase_network = jax.vmap(phase_network, in_axes=(None, 1, 1, 1), out_axes=0)
    #batch_network = jax.vmap(logabs_network, in_axes=(None, 1, 1, 1), out_axes=0)
    "for the complex wave function, we need a new function."
    "This is correct. no problem. 19.08.2024."
    #def log_network(*args, **kwargs):
    #    phase, mag = signed_network(*args, **kwargs)
    #    return mag + 1.j * phase

    pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges, electrons=jnp.array([[1.0, 0.0], [1.0, 0.0]]), batch_size=host_batch_size, init_width=0.1)
    """this operation means add one extra dimension to the array."""
    batch_pos = jnp.reshape(pos, data_shape+(-1,))
    """here, we need be sure that the array pos must be compatible with the input of AInet and hamiltonian."""
    batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)

    spins = jnp.reshape(spins, data_shape+(-1,))
    batch_spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    print("batch_pos", batch_pos)
    print("batch_atoms", batch_atoms)
    print("batch_charges", batch_charges)
    "*********************************************************************************************"
    '''we need change the strategy of parallel. So we need test the non-batch atoms and charges.'''
    jax.debug.print("charges:{}", charges)
    jax.debug.print("atoms:{}", atoms)
    jax.debug.print("spins:{}", spins)
    non_batch_charges = kfac_jax.utils.replicate_all_local_devices(charges)
    non_batch_atoms = kfac_jax.utils.replicate_all_local_devices(atoms)
    non_batch_spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    jax.debug.print("non_batch_charges:{}", non_batch_charges)
    jax.debug.print("non_batch_atoms:{}", non_batch_atoms)
    """here, we tested the parallel calculation method. Currently, after our heavy effort, it is working well."""
    #test_output_no_batch = logabs_network(params=params, pos=batch_pos[0][1], atoms=atoms, charges=charges)
    #test_output = batch_network(batch_params, batch_pos, batch_atoms, batch_charges)
    #jax.debug.print("test_output:{}", test_output)
    #test_output_no_batch = signed_network(params=params, pos=batch_pos[0][1], atoms=atoms, charges=charges)
    data = nn.AINetData(positions=batch_pos, spins=batch_spins, atoms=batch_atoms, charges=batch_charges)

    data_non_batch = nn.AINetData(positions=batch_pos, spins=non_batch_spins, atoms=non_batch_atoms, charges=non_batch_charges)
    non_phase_network = jax.vmap(phase_network, in_axes=(None, 1, None, None), out_axes=0)
    non_batch_network = jax.vmap(logabs_network, in_axes=(None, 1, None, None), out_axes=0)
    """here, we have one problem about the format of pos, spins, batch_atoms, batch_charges.
    These formats can be easily changed in nn.py. so, before we do this, we need know which format should be used in the loss function 16.08.2024."""
    #print("data.positions", data.positions.shape)
    "currently, we dont need check points. So, we ignore this part."
    '''--------------Main training-------------'''
    #Construct MC step
    #mc_step = mcstep.make_mc_step(phase_network, batch_network, signed_network)
    '''Construct loss and optimizer, local energy calculation. we are gonna deal with it at 28.08.2024.'''
    #localenergy = local_energy(f=signed_network, params=batch_params, complex_number=True)
    """so far, we have not constructed the pp module. Currently, we only execute all electrons calculation.  """
    #evaluate_loss = qmc_loss_function.make_loss(signed_network, local_energy=local_energy, params=batch_params, data=data, complex_output=True)
    """18.10.2024, we will continue later."""

    '''
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
                                    pmap_axis_name=constants.PMAP_AXIS_NAME, #this line to be done
                                    auto_register_kwargs=dict(graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERN,)) #this line to be done!!! we need read the paper of Kfac.


    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    """we got a bug here. Maybe we dont pass the correct data to the optimizer.init."""
    jax.debug.print("batch_params:{}", batch_params)
    jax.debug.print("data:{}", data)
    opt_state = optimizer.init(params=batch_params, rng=subkeys, batch=data)
    """the default option is Kfac. It could be the problem of parallirization. 23.10.2024."""
    step = make_kfac_training_step(mc_step=mc_step, damping=0.001, optimizer=optimizer)
    """main training loop"""
    for t in range(0, iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        data, batch_params, opt_state, loss, aux_data = step(data, batch_params, opt_state, subkeys)

    '''

    return signed_network, data_non_batch, batch_params, non_batch_network, non_phase_network #mc_step, local_energy


output = main()