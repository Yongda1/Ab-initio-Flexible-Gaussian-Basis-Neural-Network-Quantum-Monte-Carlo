import jax
import chex
import optax
import logging
import time
import jax.numpy as jnp
import numpy as np
import kfac_jax
from typing_extensions import Protocol
from typing import Optional, Tuple, Union
from AIQMCpretrain1 import checkpoint
from jax.experimental import multihost_utils
#from AIQMCpretrain1.VMC import VMCmcstep
from AIQMCpretrain1.VMC import mcmc
#from AIQMCpretrain1.wavefunction_Ynlm import nn
from AIQMCpretrain1.wavefunction import networks as nn
from AIQMCpretrain1.wavefunction import envelopes
from AIQMCpretrain1.Loss import loss as qmc_loss_functions
from AIQMCpretrain1 import constants
from AIQMCpretrain1.Energy import hamiltonian
from AIQMCpretrain1.Optimizer import adam
from AIQMCpretrain1.utils import writers
from AIQMCpretrain1.initial_electrons_positions.init import init_electrons
from AIQMCpretrain1.spin_indices import jastrow_indices_ee
from AIQMCpretrain1.spin_indices import spin_indices_h
from AIQMCpretrain1.pretain import pretain
from AIQMCpretrain1.utils import system
from AIQMCpretrain1 import curvature_tags_and_blocks
from AIQMCpretrain1.Optimizer import kfac
import pyscf
import functools
OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[nn.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]
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
         list_l: int, #for the angular momentum order in the pp file. It depends on the detial of the correpsonding pp file.
         nspins: Tuple,
         save_path: Optional[str],
         restore_path: Optional[str],
         structure: jnp.array,):
    """the main function for the pp calculation."""
    logging.info('Quantum Monte Carlo Start running')
    num_devices = jax.local_device_count()  # the amount of GPU per host
    num_hosts = jax.device_count() // num_devices  # the amount of host
    jax.debug.print("num_devices:{}", num_devices)
    jax.debug.print("num_hosts:{}", num_hosts)
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
        """to be continued...21.3.2025."""
        key, subkey = jax.random.split(key)
        data_shape = (num_devices, device_batch_size)
        batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
        batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
        batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
        batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
        pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                                    electrons=spins,
                                    batch_size=host_batch_size, init_width=1.0)
        #jax.debug.print("spins:{}", spins)
        generate_spin_indices = spins
        batch_pos = jnp.reshape(pos, data_shape + (-1,))
        batch_pos = kfac_jax.utils.broadcast_all_local_devices(batch_pos)
        batch_spins = jnp.repeat(spins[None, ...], batch_size, axis=0)
        batch_spins = jnp.reshape(batch_spins, data_shape + (-1,))
        batch_spins = kfac_jax.utils.broadcast_all_local_devices(batch_spins)
        data = nn.AINetData(positions=batch_pos, spins=batch_spins, atoms=batch_atoms, charges=batch_charges)

    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins,
                                                                                            nelectrons=nelectrons)
    spin_up_indices, spin_down_indices = spin_indices_h(generate_spin_indices)
    '''
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
    '''
    feature_layer = nn.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=(3, 3),
        ndim=3,
        rescale_inputs=False,
    )
    envelope = envelopes.make_isotropic_envelope()
    network = nn.make_fermi_net(
        nspins,
        charges,
        ndim=3,
        determinants=1,
        states=0,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow='default',
        bias_orbitals=True,
        full_det=True,
        rescale_inputs=False,
        complex_output=True,
    )

    key, subkey = jax.random.split(key)
    params = network.init(subkey)
    params = kfac_jax.utils.replicate_all_local_devices(params)
    # jax.debug.print("params:{}", params)
    signed_network = network.apply
    jax.debug.print("params_before:{}", params)

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )

    mol = pyscf.gto.Mole()
    mol.build(
        atom='''C 0 0 0''',
        basis='sto-3g')

    hartree_fock = pretain.get_hf(pyscf_mol=mol,
                                  molecule=[system.Atom('C', (0, 0, 0))],
                                  nspins=nspins,
                                  restricted=False,
                                  basis='ccpvdz',
                                  )
    hartree_fock.mean_field.mo_coeff = multihost_utils.broadcast_one_to_all(
        hartree_fock.mean_field.mo_coeff
    )
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    pretain_spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    batch_orbitals = jax.vmap(network.orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    """we still dont understand the pretrain process from pyscf. 18.4.2025."""
    params, data.positions = pretain.pretain_hartree_fock(
        params=params,
        positions=data.positions,
        spins=pretain_spins,
        atoms=data.atoms,
        charges=data.charges,
        batch_network=batch_network,
        batch_orbitals=batch_orbitals,
        sharded_key=subkeys,
        electrons=(3, 3),
        scf_approx=hartree_fock,
        iterations=100,
        batch_size=device_batch_size,
        scf_fraction=1.0,
        states=0
    )
    #jax.debug.print("params:{}", params)
    mc_atoms = atoms
    jax.debug.print("mc_atoms:{}", mc_atoms)
    mcmc_step = mcmc.make_mcmc_step(
        batch_network,
        device_batch_size,
        steps=10,
        atoms=mc_atoms,
        blocks=1,
    )
    '''
    mc_step = VMCmcstep.main_monte_carlo(
        f=signed_network,
        tstep=tstep,
        ndim=ndim,
        nelectrons=nelectrons,
        nsteps=nsteps,
        batch_size=int(batch_size / (num_devices * num_hosts)))
    '''
    jax.debug.print("batch_size_run:{}", int(batch_size / (num_devices * num_hosts)))
    #mc_step_parallel = jax.pmap(mcmc_step, donate_argnums=1)
    logging.info('--------------Create Hamiltonian--------------')

    #localenergy = hamiltonian.local_energy(f=signed_network, charges=charges, nspins=spins, use_scan=False)
    '''
    evaluate_loss = qmc_loss_functions.make_loss(network=log_network, local_energy=localenergy,
                                                 clip_local_energy=5.0,
                                                 clip_from_median=False,
                                                 center_at_clipped_energy=True,
                                                 complex_output=True,
                                                 )
    '''
    local_energy = hamiltonian.local_energy(f=signed_network, charges=charges, nspins=spins,
                                            )

    evaluate_loss = qmc_loss_functions.make_loss(
        log_network,
        local_energy,
        clip_local_energy=5.0,
        clip_from_median=True,
        center_at_clipped_energy=True,
        complex_output=True,)
    """we try adam optimzier here."""
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    def learning_rate_schedule(t_: jnp.array, rate=0.05, delay=1.0, decay=10000) -> jnp.array:
        return rate * jnp.power(1.0 / (1.0 + (t_ / delay)), decay)
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
    opt_state = optimizer.init(params, subkeys, data)
    opt_state = opt_state_ckpt or opt_state
    step_kfac = make_kfac_training_step(
        mcmc_step=mcmc_step,
        damping=0.001,
        optimizer=optimizer,
        reset_if_nan=False)

    train_schema = ['step', 'energy']
    writer_manager = writers.Writer(
        name='train_states',
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False
    )
    #time_of_last_ckpt = time.time()
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray([0.02]))
    """main training loop"""
    with writer_manager as writer:
        for t in range(t_init, t_init + iterations):
            """we need do more to deal with amda optimzier. especially for saving module. 23.3.2025."""
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            # jax.debug.print("subkeys:{}", subkeys)
            # jax.debug.print("before_run_data:{}", data)
            print("--------------------------")
            #data = mc_step_parallel(params, data, subkeys)
            #data, pmove = mc_step_parallel(params, data, subkeys, width=jnp.array([0.02]))
            jax.debug.print("after_run_data:{}", data)
            print("--------------------------")

            data, params, opt_state, loss, aux_data, pmove = step_kfac(data, params, opt_state, subkeys, mcmc_width)
            #loss = loss
            logging_str = ('Step %05d: ', '%03.4f E_h,')
            logging_args = t, loss,
            writer_kwargs = {
                'step': t,
                'energy': np.asarray(loss),
            }
            # jax.debug.print("loss:{}", loss)
            logging.info(logging_str, *logging_args)
            writer.write(t, **writer_kwargs)
            # jax.debug.print("time.time{}", time.time())
            # jax.debug.print("time_of_last_ckpt:{}", time_of_last_ckpt)
            # if time.time() - time_of_last_ckpt > save_frequency * 60:
            # if t > 1:
            # jax.debug.print("opt_state:{}", opt_state)
            """here, we store every step optimization."""
            if t % 1000 == 0:
                save_params = np.asarray(params)
                save_opt_state = np.asarray(opt_state, dtype=object)
                # jax.debug.print("save_params:{}", save_params)
                # jax.debug.print("ckpt_save_path:{}", ckpt_save_path)
                checkpoint.save(ckpt_save_path, t, data, save_params, save_opt_state)
                # time_of_last_ckpt = time.time()
