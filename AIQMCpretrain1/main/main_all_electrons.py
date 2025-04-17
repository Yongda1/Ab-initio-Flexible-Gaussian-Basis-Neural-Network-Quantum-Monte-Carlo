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
from AIQMCrelease3.wavefunction_Ynlm import nn
#from AIQMCrelease1.MonteCarloSample import VMCmcstep
from AIQMCrelease3.MonteCarloSample import mcstep
from AIQMCrelease3.Loss import loss as qmc_loss_functions
from AIQMCrelease3 import constants
from AIQMCrelease3.Energy import hamiltonian
from AIQMCrelease3 import checkpoint
from AIQMCrelease3 import curvature_tags_and_blocks
from AIQMCrelease3.utils import writers
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
    #jax.debug.print("spins_no_batch:{}", spins_no_batch[None, ...])
    #spins_batch = jnp.repeat(spins_no_batch[None, ...], batch_size, axis=0)
    #jax.debug.print("spins_batch:{}", spins_batch)
    return electrons_positions_batch, spins_no_batch


OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]

OptUpdateResults = Tuple[
    nn.ParamTree, Optional[OptimizerState], jnp.array, Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):
    def __call__(self, params: nn.ParamTree, data: nn.AINetData, opt_state: optax.OptState,
                 key: chex.PRNGKey) -> OptUpdateResults:
        """Evaluates the loss and gradients and updates the parameters accordingly."""


StepResults = Tuple[
    nn.AINetData,
    nn.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: nn.AINetData,
      params: nn.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def make_kfac_training_step(
    mcmc_step,
    damping: float,
    optimizer: kfac_jax.Optimizer,
    reset_if_nan: bool = False) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))
  # Due to some KFAC cleverness related to donated buffers, need to do this
  # to make state resettable
  copy_tree = constants.pmap(
      functools.partial(jax.tree_util.tree_map,
                        lambda x: (1.0 * x).astype(x.dtype)))

  def step(
      data: nn.AINetData,
      params: nn.ParamTree,
      state: kfac_jax.Optimizer.State,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    if reset_if_nan:
      old_params = copy_tree(params)
      old_state = copy_tree(state)

    # Optimization step
    new_params, new_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        momentum=shared_mom,
        damping=shared_damping,
    )
    #jax.debug.print("new_params:{}", new_params)
    if reset_if_nan and jnp.isnan(stats['loss']):
      new_params = old_params
      new_state = old_state
    return data, new_params, new_state, stats['loss'], stats['aux'], pmove

  return step

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
         save_path: Optional[str],
         restore_path: Optional[str],
         save_frequency: float,
         structure: jnp.array,):
    logging.info('Quantum Monte Carlo Start running')
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
        #jax.debug.print("spins:{}", spins)
        batch_spins = jnp.repeat(spins[None, ...], batch_size, axis=0)
        batch_spins = jnp.reshape(batch_spins, data_shape + (-1,))
        batch_spins = kfac_jax.utils.broadcast_all_local_devices(batch_spins)
        #jax.debug.print("batch_spins:{}", batch_spins)
        data = nn.AINetData(positions=batch_pos, spins=batch_spins, atoms=batch_atoms, charges=batch_charges)
    '''to be continued...20.1.2025. today, we rebuild the wavefunction. Tomorrow, we rebuild the VMCmcstep to check the engine.'''



    """we already write the envelope function in the nn_wrong.py."""
    feature_layer = nn.make_ferminet_features(natoms=natoms, nspins=(1, 1), ndim=ndim, )
    network = nn.make_fermi_net(ndim=ndim,
                                nspins=(1, 1),
                                determinants=1,
                                feature_layer=feature_layer,
                                charges=charges,
                                full_det=True)


    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply
    #jax.debug.print("charges:{}", charges)

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    logging.info('--------------Main training--------------')
    logging.info('--------------Start Monte Carlo process--------------')

    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    batch_network = jax.vmap(logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    """we solve the following problem tomorrow. 20.1.2025."""
    '''
    mc_step = VMCmcstep.main_monte_carlo(
        f=signed_network,
        tstep=tstep,
        ndim=ndim,
        nelectrons=nelectrons,
        nsteps=nsteps,
        batch_size=batch_size)
    '''
    mc_step = mcstep.make_mcmc_step(
        batch_network,
        batch_per_device=batch_size,
        steps=10,
        atoms=atoms,
        ndim=3,
        blocks=1)

    """we need add the pseudopotential module into the hamiltonian module."""
    """be aware of the list_l, this variable means the angular momentum function max indice in pseudopotential file.7.1.2025."""
    logging.info('--------------Create Hamiltonian--------------')
    localenergy = hamiltonian.local_energy(f=signed_network, charges=charges, nspins=spins, use_scan=False)

    """so far, we have not constructed the pp module. Currently, we only execute all electrons calculation.  """
    logging.info('--------------Build loss function--------------')
    evaluate_loss = qmc_loss_functions.make_loss(network=log_network, local_energy=localenergy,
                                                 clip_local_energy=5.0,
                                                 clip_from_median=False,
                                                 center_at_clipped_energy=True,
                                                 complex_output=True,
                                                 )

    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

    def learning_rate_schedule(t_: jnp.array, rate=0.05, delay=1.0, decay=10000) -> jnp.array:
        return rate * jnp.power(1.0/(1.0 + (t_/delay)), decay)

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

    step_kfac = make_kfac_training_step(
        mcmc_step=mc_step,
        damping=0.001,
        optimizer=optimizer,
        reset_if_nan=False)

    """main training loop"""
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(0.02))
    train_schema = ['step', 'energy']
    writer_manager = writers.Writer(
        name='train_states',
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False
    )
    time_of_last_ckpt = time.time()

    with writer_manager as writer:
        for t in range(t_init, iterations):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, opt_state, loss, aux_data, pmove = step_kfac(data, params, opt_state, subkeys, mcmc_width)
            loss = loss[0]
            logging_str = ('Step %05d: '
                           '%03.4f E_h,')
            logging_args = t, loss,
            writer_kwargs = {
                'step': t,
                'energy': np.asarray(loss),
            }
            logging.info(logging_str, *logging_args)
            writer.write(t, **writer_kwargs)
            if time.time() - time_of_last_ckpt > save_frequency * 60:
                checkpoint.save(ckpt_save_path, t, data, params, opt_state)
                time_of_last_ckpt = time.time()