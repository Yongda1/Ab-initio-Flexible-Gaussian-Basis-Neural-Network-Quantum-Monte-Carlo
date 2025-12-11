"""Currently, we only develop the single stream version.
Because the parallel strategy of JAX changed a lot since last year. We need spend a long time to reconstruct it.
one more thing is that we do not pretrain it currently.

24.10.2025."""

import ml_collections
import jax.numpy as jnp
import jax
import time
import optax
import kfac_jax
from Kolmogorov_Arnold_QMC.optimizer.opt import make_training_step, make_opt_update_step
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.kan_networks_case_one import make_kan_net, KANetsData
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.spin_indices import jastrow_indices_ee, jastrow_indices_ae
from Kolmogorov_Arnold_QMC.monte_carlo_step.mcmc import make_mcmc_step
from Kolmogorov_Arnold_QMC.hamiltonian import hamiltonian
from Kolmogorov_Arnold_QMC.loss_function import loss as qmc_loss_functions
from Kolmogorov_Arnold_QMC.initialization import electrons_initialization
from Kolmogorov_Arnold_QMC.pretrain import pretrain_HF


def train(cfg: ml_collections.ConfigDict,):
    spins_jastrow = jnp.array(cfg.spins)
    #jax.debug.print("spins:{}", spins_jastrow)
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins_jastrow,
                                                                                            nelectrons=6)
    #jax.debug.print("parallel_indices:{}", parallel_indices)
    g = jnp.array(cfg.g)
    k = jnp.array(cfg.k)
    layer_dims = jnp.array(cfg.layer_dims)
    """electron coordinates initialization. 10.11.2025."""
    seed_electrons_coords = 22
    key_electrons_coords = jax.random.PRNGKey(seed_electrons_coords)
    key_electrons_coords, subkey_electrons_coords = jax.random.split(key_electrons_coords)
    pos, spins_test = electrons_initialization.init_electrons(
        subkey_electrons_coords,
        cfg.system.molecule,
        cfg.system.electrons,
        batch_size=cfg.batch_size,
        init_width=0.1,
        core_electrons={},
    )

    #jax.debug.print("pos_test:{}", pos_test)
    """test pretrain. 10.11.2025."""
    hartree_fock = pretrain_HF.get_hf(
        pyscf_mol=cfg.system.get('pyscf_mol'),
        molecule=cfg.system.molecule,
        nspins=(3, 3),
        restricted=False,
        basis='ccpvdz',
        ecp={},
        core_electrons={},
        states=0,
        excitation_type='ordered')
    # broadcast the result of PySCF from host 0 to all other hosts
    jax.debug.print("hartree_fock:{}", hartree_fock)
    """we need check the next fitting step.10.11.2025."""

    charges = jnp.array(cfg.charges)
    atoms = jnp.array(cfg.atoms)
    #pos = jnp.array(cfg.pos)
    #jax.debug.print("g:{}", g)
    grid_range_envelope = jnp.array(cfg.envelope.grid_range_envelope)
    kan_init, kan_apply, orbitals_apply = make_kan_net(nspins=(3, 3),
                                                       charges=charges,
                                                       nelectrons=6,
                                                       nfeatures=4,
                                                       n_parallel=n_parallel,
                                                       n_antiparallel=n_antiparallel,
                                                       parallel_indices=parallel_indices,
                                                       antiparallel_indices=antiparallel_indices,
                                                       grid_range=cfg.grid_range,
                                                       g=g,
                                                       k=k,
                                                       natoms=1,
                                                       ndims=3,
                                                       layer_dims=layer_dims,
                                                       g_envelope=cfg.envelope.g_envelope,
                                                       k_envelope=cfg.envelope.k_envelope,
                                                       grid_range_envelope=grid_range_envelope,
                                                       chebyshev=cfg.chebyshev,
                                                       spline=cfg.spline,
                                                       add_residual=cfg.add_residual,
                                                       add_bias=cfg.add_bias,
                                                       external_weights=cfg.external_weights,
                                                       envelope_chebyshev=cfg.envelope_chebyshev,
                                                       envelope_spline=cfg.envelope_spline,
                                                       envelope_simple=cfg.envelope_simple,)

    seed = 42
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    params = kan_init(subkey)
    signed_network = kan_apply
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    """these are for real orbitals. not for complex orbitals. to be continued...3.12.2025.!!!"""
    """complex version can run. 5.12.2025."""
    spins = jnp.array([cfg.spins])
    #jax.debug.print("spins:{}", spins)
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, None, None, None), out_axes=0
    )
    jax.debug.print("pos:{}", pos)
    jax.debug.print("spis:{}", spins)
    jax.debug.print("atoms:{}", atoms)
    jax.debug.print("charges:{}", charges)
    #value_wavefunction = batch_network(params, pos, spins, atoms, charges)

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    key, hartree_fock_key = jax.random.split(key, 2)
    orbitals_vmap = jax.vmap(orbitals_apply, in_axes=(None, 0, None, None, None), out_axes=0)
    params, pos = pretrain_HF.pretrain_hartree_fock(
        params=params,
        positions=pos,
        spins=spins,
        charges=charges,
        atoms=atoms,
        batch_network=batch_network,
        batch_orbitals=orbitals_vmap,
        sharded_key=hartree_fock_key,
        electrons=cfg.system.electrons,
        scf_approx=hartree_fock,
        iterations=cfg.preiterations,
        batch_size=cfg.batch_size,
        scf_fraction=1.0,
        states=0,
    )

    #jax.debug.print("pos:{}", pos)
    #jax.debug.print("params:{}", params)
    #jax.debug.print("atoms:{}", atoms)

    #jax.debug.print("wavefunction_value:{}", wavefunction_value)
    """we need do batch for pos."""
    data = KANetsData(positions=pos, spins=spins, atoms=atoms, charges=charges)

    monte_carlo = make_mcmc_step(batch_network=batch_network,
                                 batch_per_device=2,
                                 steps=10,
                                 atoms=atoms,
                                 blocks=1)
    """the following two lines for testing the walker move process.31.10.2025."""
    #key, monte_carlo_key = jax.random.split(subkey)
    #new_data = monte_carlo(params, data, monte_carlo_key, 0.1)
    """now we need move to the energy calculation method."""
    """the following is energy calculation test. 31.10.2025."""
    #jax.debug.print("charges:{}", charges)
    key, energy_key = jax.random.split(key)
    local_energy = hamiltonian.local_energy(f=signed_network,
                                            nspins=(3, 3),
                                            charges=charges,
                                            use_scan=False,
                                            complex_output=False,
                                            laplacian_method='default')
    """the reason for the error is local_energy can not accept the batched input. 3.11.2025."""
    """we solved it by a simple reconstruction of input axes."""
    #local_energy_vmap = jax.vmap(local_energy, in_axes=(None, None, 0, None, None, None))
    #output = local_energy_vmap(params, energy_key, data.positions, data.spins, data.atoms, data.charges,)
    #jax.debug.print("output:{}", output)
    """next, we need construction the loss function. 3.11.2025."""
    evaluate_loss = qmc_loss_functions.make_loss(log_network,
                                                 local_energy,
                                                 clip_local_energy=5.0,
                                                 clip_from_median=True,
                                                 center_at_clipped_energy=True,
                                                 complex_output=True,
                                                 )

    def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
        return 0.05 * jnp.power(
            (1.0 / (1.0 + (t_ / 1.0))), 10000.0)

    optimizer = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.999,eps=1e-6),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
    #jax.debug.print("type_of_optimizer:{}",type(optimizer))
    if isinstance(optimizer, optax.GradientTransformation):
        opt_state = optimizer.init(params)
        #jax.debug.print("opt_state:{}", opt_state)
        """because we dont set any parallel strategy for monte carlo step. We also need rewrite the parallel strategy for optimization.4.11.2025."""
        step = make_training_step(mcmc_step=monte_carlo,
                                  optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
                                  reset_if_nan=True)

    mcmc_width = 0.1
    pmoves = None
    t_init = 0
    sharded_key = key
    jax.debug.print("sharded_key:{}", sharded_key)
    """to be continued... 3.11.2025."""
    for t in range(t_init, cfg.iterations):
        sharded_key, subkeys = jax.random.split(sharded_key, 2)

        data, params, opt_state, loss, aux_data = step(data, params, opt_state, subkeys, mcmc_width,)

        #loss = loss[0]
        jax.debug.print("loss:{}", loss)











