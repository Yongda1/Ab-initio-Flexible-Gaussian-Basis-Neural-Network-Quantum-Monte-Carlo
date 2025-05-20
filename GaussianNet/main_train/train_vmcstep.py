"""to debug the VMCmcstep.py, we reconstruct the train.py."""

import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union
from absl import logging
import chex
from GaussianNet import checkpoint
from GaussianNet import constants
from GaussianNet.wavefunction import envelopes
from GaussianNet.hamiltonian import hamiltonian
from GaussianNet.hamiltonian import pphamiltonian
from GaussianNet.loss_function import loss as qmc_loss_functions
from GaussianNet.loss_function import pploss as qmc_pploss_functions
from GaussianNet.monte_carlo_step import mcmc
from GaussianNet.monte_carlo_step import VMCmcstep
from GaussianNet.wavefunction import networks
from GaussianNet.pretainHF import pretrain
from GaussianNet.tools.utils import statistics
from GaussianNet.tools.utils import system
from GaussianNet.tools.utils import utils
from GaussianNet.tools.utils import writers
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol
from GaussianNet.wavefunction import spin_indices


def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
    core_electrons: Mapping[str, int] = {},
    max_iter: int = 10000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    niter = 0
    total_electrons = sum(atom.charge - core_electrons.get(atom.symbol, 0)
                          for atom in molecule)
    if total_electrons != sum(electrons):
        if len(molecule) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:
        atomic_spin_configs = [
            (atom.element.nalpha - core_electrons.get(atom.symbol, 0) // 2,
             atom.element.nbeta - core_electrons.get(atom.symbol, 0) // 2)
            for atom in molecule
        ]
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while (
                tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons
                and niter < max_iter
        ):
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            atomic_spin_configs[i] = nbeta, nalpha
            niter += 1

    if tuple(sum(x) for x in zip(*atomic_spin_configs)) == electrons:
        # Assign each electron to an atom initially.
        electron_positions = []
        for i in range(2):
            for j in range(len(molecule)):
                atom_position = jnp.asarray(molecule[j].coords)
                electron_positions.append(
                    jnp.tile(atom_position, atomic_spin_configs[j][i]))
        electron_positions = jnp.concatenate(electron_positions)
    else:
        logging.warning(
            'Failed to find a valid initial electron configuration after %i'
            ' iterations. Initializing all electrons from a Gaussian distribution'
            ' centred on the origin. This might require increasing the number of'
            ' iterations used for pretraining and MCMC burn-in. Consider'
            ' implementing a custom initialisation.',
            niter,
        )
        electron_positions = jnp.zeros(shape=(3 * sum(electrons),))

    # Create a batch of configurations with a Gaussian distribution about each atom.
    key, subkey = jax.random.split(key)
    electron_positions += (
            jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
            * init_width
    )

    electron_spins = _assign_spin_configuration(
        electrons[0], electrons[1], batch_size
    )

    return electron_positions, electron_spins


OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.GaussianNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.GaussianNetData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.GaussianNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
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


def null_update(
    params: networks.ParamTree,
    data: networks.GaussianNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.GaussianNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.GaussianNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers."""
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(
      data: networks.GaussianNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data

  return step


def train(cfg: ml_collections.ConfigDict,):
    num_devices = jax.local_device_count()
    num_hosts = jax.device_count() // num_devices
    num_states = cfg.system.get('states', 0) or 1  # avoid 0/1 confusion
    logging.info('Starting QMC with %i XLA devices per host '
                 'across %i hosts.', num_devices, num_hosts)
    if cfg.batch_size % (num_devices * num_hosts) != 0:
        raise ValueError('Batch size must be divisible by number of devices, '
                         f'got batch size {cfg.batch_size} for '
                         f'{num_devices * num_hosts} devices.')
    host_batch_size = cfg.batch_size // num_hosts  # batch size per host
    total_host_batch_size = host_batch_size * num_states
    device_batch_size = host_batch_size // num_devices  # batch size per device
    data_shape = (num_devices, device_batch_size)

    if cfg.system.pyscf_mol:
        cfg.update(
            system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

    # Convert mol config into array of atomic positions and charges
    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])

    nspins = cfg.system.electrons

    # Generate atomic configurations for each walker
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

    if cfg.debug.deterministic:
        seed = 23
    else:
        seed = jnp.asarray([1e6 * time.time()])
        seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    key = jax.random.PRNGKey(seed)

    # extract number of electrons of each spin around each atom removed because
    # of pseudopotentials
    if cfg.system.pyscf_mol:
        cfg.system.pyscf_mol.build()
        core_electrons = {
            atom: ecp_table[0]
            for atom, ecp_table in cfg.system.pyscf_mol._ecp.items()  # pylint: disable=protected-access
        }
        ecp = cfg.system.pyscf_mol.ecp
    else:
        ecp = {}
        core_electrons = {}

    if cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0:
        hartree_fock = pretrain.get_hf(
            pyscf_mol=cfg.system.get('pyscf_mol'),
            molecule=cfg.system.molecule,
            nspins=nspins,
            restricted=False,
            basis=cfg.pretrain.basis,
            ecp=ecp,
            core_electrons=core_electrons,
            states=cfg.system.states,
            excitation_type=cfg.pretrain.get('excitation_type', 'ordered'))
        # broadcast the result of PySCF from host 0 to all other hosts
        hartree_fock.mean_field.mo_coeff = multihost_utils.broadcast_one_to_all(
            hartree_fock.mean_field.mo_coeff
        )

    spins_test = jnp.array([[1., 1., 1, - 1., - 1., -1]])
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = \
        spin_indices.jastrow_indices_ee(spins=spins_test,
                                        nelectrons=6)
    #jax.debug.print("n_parallel:{}", n_parallel)
    #jax.debug.print("n_antiparallel:{}", n_antiparallel)
    network = networks.make_gaussian_net(nspins=(3, 3),
                                charges=charges,
                                parallel_indices=parallel_indices,
                                antiparallel_indices=antiparallel_indices,
                                n_parallel=n_parallel,
                                n_antiparallel=n_antiparallel, )

    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    signed_network = network.apply
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )

    use_complex = cfg.network.get('complex', False)

    def log_network(*args, **kwargs):
      if not use_complex:
        raise ValueError('This function should never be used if the '
                         'wavefunction is real-valued.')
      phase, mag = signed_network(*args, **kwargs)
      return mag + 1.j * phase

    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
    ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

    ckpt_restore_filename = (
            checkpoint.find_last_checkpoint(ckpt_save_path) or
            checkpoint.find_last_checkpoint(ckpt_restore_path))

    if ckpt_restore_filename:
        (t_init,
         data,
         params,
         opt_state_ckpt,
         mcmc_width_ckpt,
         density_state_ckpt) = checkpoint.restore(
            ckpt_restore_filename, host_batch_size)
    else:
        logging.info('No checkpoint found. Training new model.')
        key, subkey = jax.random.split(key)
        # make sure data on each host is initialized differently
        subkey = jax.random.fold_in(subkey, jax.process_index())
        # create electron state (position and spin)
        pos, spins = init_electrons(
            subkey,
            cfg.system.molecule,
            cfg.system.electrons,
            batch_size=total_host_batch_size,
            init_width=cfg.mcmc.init_width,
            core_electrons=core_electrons,
        )
        # jax.debug.print("spins:{}", spins)
        # For excited states, each device has a batch of walkers, where each walker
        # is nstates * nelectrons. The vmap over nstates is handled in the function
        # created in make_total_ansatz
        pos = jnp.reshape(pos, data_shape + (-1,))
        pos = kfac_jax.utils.broadcast_all_local_devices(pos)
        spins = jnp.reshape(spins, data_shape + (-1,))
        spins = kfac_jax.utils.broadcast_all_local_devices(spins)
        data = networks.GaussianNetData(
            positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
        )

        t_init = 0
        opt_state_ckpt = None
        mcmc_width_ckpt = None
        density_state_ckpt = None

    train_schema = ['step', 'energy', 'ewmean', 'ewvar',]
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    #Pretain to match Hartree-Fock
    if (
            t_init == 0
            and cfg.pretrain.method == 'hf'
            and cfg.pretrain.iterations > 0
    ):
        pretrain_spins = spins[0, 0]
        batch_orbitals = jax.vmap(
            network.orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0
        )
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        params, data.positions = pretrain.pretrain_hartree_fock(
            params=params,
            positions=data.positions,
            spins=pretrain_spins,
            atoms=data.atoms,
            charges=data.charges,
            batch_network=batch_network,
            batch_orbitals=batch_orbitals,
            #network_options=network.options,
            sharded_key=subkeys,
            electrons=cfg.system.electrons,
            scf_approx=hartree_fock,
            iterations=cfg.pretrain.iterations,
            batch_size=device_batch_size,
            scf_fraction=cfg.pretrain.get('scf_fraction', 0.0),
            states=cfg.system.states,
        )

    mcmc_step = VMCmcstep.walkers_update(f=signed_network,
                                         tstep=0.02,
                                         ndim=3,
                                         nelectrons=cfg.system.nelectrons,)

    laplacian_method = cfg.optim.get('laplacian', 'default')
    # pp_symbols = cfg.system.get('pp', {'symbols': None}).get('symbols')
    local_energy_fn = hamiltonian.local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        complex_output=use_complex,
        laplacian_method=laplacian_method)
    local_energy = local_energy_fn
    evaluate_loss = qmc_loss_functions.make_loss(
        log_network if use_complex else logabs_network,
        local_energy,
        clip_local_energy=cfg.optim.clip_local_energy,
        clip_from_median=cfg.optim.clip_median,
        center_at_clipped_energy=cfg.optim.center_at_clip,
        complex_output=use_complex,
        max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
    )

    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t_ / cfg.optim.lr.delay))), cfg.optim.lr.decay)

    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))

    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_training_step(
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
        reset_if_nan=cfg.optim.reset_if_nan)

    weighted_stats = None
    if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
        logging.info('No optimizer provided. Assuming inference run.')
        logging.info('Setting initial iteration to 0.')
        t_init = 0

    writer_manager = writers.Writer(
        name='train_stats',
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False)

    mcmc_step_parallel = jax.pmap(jax.vmap(mcmc_step,
                                           in_axes=(networks.GaussianNetData(positions=0, spins=0, atoms=0, charges=0), None, 0)),
                                  in_axes=(networks.GaussianNetData(positions=0, spins=0, atoms=0, charges=0), 0, 0))

    def generate_batch_key(batch_size: int):
        def get_keys(key: chex.PRNGKey):
            keys = jax.random.split(key, num=batch_size)
            return keys

        return get_keys

    with writer_manager as writer:
        # Main training loop
        num_resets = 0  # used if reset_if_nan is true
        for t in range(t_init, cfg.optim.iterations):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            jax.debug.print("subkeys:{}", subkeys)
            generate_mc_keys = generate_batch_key(6)
            mc_keys = jax.pmap(generate_mc_keys)(subkeys)
            jax.debug.print("data:{}", data)
            jax.debug.print("mc_keys:{}", mc_keys)
            for i in range(10):
                data, mc_keys = mcmc_step_parallel(data, params, mc_keys)
                jax.debug.print("output_data:{}", data)
                jax.debug.print("mc_keys:{}", mc_keys)
            data, params, opt_state, loss, aux_data = step(
                data,
                params,
                opt_state,
                subkeys,)

            loss = loss[0]
            jax.debug.print("loss:{}", loss)