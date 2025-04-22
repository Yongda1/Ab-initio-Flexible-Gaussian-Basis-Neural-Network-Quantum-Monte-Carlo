"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union
from absl import logging
import chex
from modified_ferminet.ferminet import checkpoint
from modified_ferminet.ferminet import constants
#from ferminet import curvature_tags_and_blocks
from modified_ferminet.ferminet import envelopes
from modified_ferminet.ferminet import hamiltonian
from modified_ferminet.ferminet import loss as qmc_loss_functions
from modified_ferminet.ferminet import mcmc
from modified_ferminet.ferminet import networks
from modified_ferminet.ferminet import observables
from modified_ferminet.ferminet import pretrain
from modified_ferminet.ferminet import psiformer
from modified_ferminet.ferminet.utils import statistics
from modified_ferminet.ferminet.utils import system
from modified_ferminet.ferminet.utils import utils
from modified_ferminet.ferminet.utils import writers
from modified_ferminet.ferminet.DMC.optimizer import adam
#from modified_ferminet.ferminet import VMCmcstep
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol
from modified_ferminet.ferminet import spin_indices


def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
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
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.
    core_electrons: mapping of element symbol to number of core electrons
      included in the pseudopotential.
    max_iter: maximum number of iterations to try to find a valid initial
        electron configuration for each atom. If reached, all electrons are
        initialised from a Gaussian distribution centred on the origin.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """
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
    electron_positions = jnp.zeros(shape=(3*sum(electrons),))

  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  electron_positions += (
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
      * init_width
  )

  electron_spins = _assign_spin_configuration(
      electrons[0], electrons[1], batch_size
  )

  return electron_positions, electron_spins

def train(cfg: ml_collections.ConfigDict, writer_manager=None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
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

  # Check if mol is a pyscf molecule and convert to internal representation
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
    hartree_fock.mean_field.mo_coeff = multihost_utils.broadcast_one_to_all(hartree_fock.mean_field.mo_coeff)

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
    jax.debug.print("spins:{}", spins)
    spins_input_nn = spins
    pos = jnp.reshape(pos, data_shape + (-1,))
    pos = kfac_jax.utils.broadcast_all_local_devices(pos)
    spins = jnp.reshape(spins, data_shape + (-1,))
    spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    #jax.debug.print("spins:{}", spins)
    data = networks.FermiNetData(positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges)

  feature_layer = networks.make_ferminet_features(
    natoms=charges.shape[0],
    nspins=cfg.system.electrons,
    ndim=cfg.system.ndim,
    rescale_inputs=cfg.network.get('rescale_inputs', False),
  )

  envelope = envelopes.make_isotropic_envelope()
  use_complex = cfg.network.get('complex', True)

  if cfg.network.network_type == 'ferminet':
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = \
      spin_indices.jastrow_indices_ee(spins=spins_input_nn[0],
                                      nelectrons=6)
    jax.debug.print("n_parallel:{}", n_parallel)
    jax.debug.print("n_antiparallel:{}", n_antiparallel)
    network = networks.make_fermi_net(
      nspins,
      charges,
      ndim=cfg.system.ndim,
      determinants=cfg.network.determinants,
      states=cfg.system.states,
      envelope=envelope,
      nelectrons=6,
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

  key, subkey = jax.random.split(key)
  params = network.init(subkey)
  params = kfac_jax.utils.replicate_all_local_devices(params)
  signed_network = network.apply
  logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0)  # batched network

  def log_network(*args, **kwargs):
    if not use_complex:
      raise ValueError('This function should never be used if the wavefunction is real-valued.')
    phase, mag = signed_network(*args, **kwargs)
    return mag + 1.j * phase
  """to be continued... 19.4.2025."""
  t_init = 0
  opt_state_ckpt = None
  mcmc_width_ckpt = None

  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove']
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

  if (t_init == 0 and cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0):
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
      network_options=network.options,
      sharded_key=subkeys,
      electrons=cfg.system.electrons,
      scf_approx=hartree_fock,
      iterations=cfg.pretrain.iterations,
      batch_size=device_batch_size,
      scf_fraction=cfg.pretrain.get('scf_fraction', 0.0),
      states=cfg.system.states,
    )

  atoms_to_mcmc = atoms
  mcmc_step = mcmc.make_mcmc_step(
    batch_network,
    device_batch_size,
    steps=cfg.mcmc.steps,
    atoms=atoms_to_mcmc,
    blocks=cfg.mcmc.blocks * num_states,)

  laplacian_method = cfg.optim.get('laplacian', 'default')
  pp_symbols = cfg.system.get('pp', {'symbols': None}).get('symbols')

  local_energy_fn = hamiltonian.local_energy(
    f=signed_network,
    charges=charges,
    nspins=nspins,
    use_scan=False,
    complex_output=use_complex,
    laplacian_method=laplacian_method,
    states=cfg.system.get('states', 0),
    state_specific=(cfg.optim.objective == 'vmc_overlap'),
    pp_type=cfg.system.get('pp', {'type': 'ccecp'}).get('type'),
    pp_symbols=pp_symbols if cfg.system.get('use_pp') else None)

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

  if cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
      optax.scale_by_adam(**cfg.optim.adam),
      optax.scale_by_schedule(learning_rate_schedule),
      optax.scale(-1.))

  opt_state = jax.pmap(optimizer.init)(params)
  opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  step = adam.make_training_step(
    mcmc_step=mcmc_step,
    optimizer_step=adam.make_opt_update_step(evaluate_loss, optimizer),
    reset_if_nan=cfg.optim.reset_if_nan)

  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = adam.make_training_step(
        mcmc_step=mcmc_step, optimizer_step=adam.null_update)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data,
          params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])

  weighted_stats = None
  if writer_manager is None:
    writer_manager = writers.Writer(
        name='train_stats',
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False)
  with writer_manager as writer:
    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, opt_state, loss, aux_data, pmove = step(
          data,
          params,
          opt_state,
          subkeys,
          mcmc_width)

      # due to pmean, loss, and pmove should be the same across
      # devices.
      loss = loss[0]
      # per batch variance isn't informative. Use weighted mean and variance
      # instead.
      weighted_stats = statistics.exponentialy_weighted_stats(
          alpha=0.1, observation=loss, previous_stats=weighted_stats)
      pmove = pmove[0]
      mcmc_width, pmoves = mcmc.update_mcmc_width(
        t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves)

      logging_str = ('Step %05d: '
                     '%03.4f E_h, exp. variance=%03.4f E_h^2, pmove=%0.2f')
      logging_args = t, loss, weighted_stats.variance, pmove
      writer_kwargs = {
        'step': t,
        'energy': np.asarray(loss),
        'ewmean': np.asarray(weighted_stats.mean),
        'ewvar': np.asarray(weighted_stats.variance),
        'pmove': np.asarray(pmove),
      }
      logging.info(logging_str, *logging_args)
      writer.write(t, **writer_kwargs)

      if t % 100 == 0:
        checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)



from modified_ferminet.ferminet import base_config

cfg = base_config.default()
cfg.system.electrons = (3, 3)  # (alpha electrons, beta electrons)
cfg.network.complex = True
cfg.system.molecule = [system.Atom('C', (0, 0, 0))]
# Set training parameters
cfg.batch_size = 100
cfg.pretrain.iterations = 20
cfg.optim.optimizer = 'adam'
cfg.optim.iterations = 1000
#cfg.system.use_pp = True
#cfg.system.pp.symbols = 'C'
cfg.network.determinants = 1
cfg.network.full_det = True
cfg.log.save_path = 'save'
train(cfg)