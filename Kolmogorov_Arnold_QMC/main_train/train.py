"""Currently, we only develop the single stream version.
Because the parallel strategy of JAX changed a lot since last year. We need spend a long time to reconstruct it.
one more thing is that we do not pretrain it currently.

24.10.2025."""

import ml_collections
import jax.numpy as jnp
import jax
import time
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.kan_networks_case_one import make_kan_net, KANetsData
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.spin_indices import jastrow_indices_ee, jastrow_indices_ae
from Kolmogorov_Arnold_QMC.monte_carlo_step.mcmc import make_mcmc_step


def train(cfg: ml_collections.ConfigDict,):
    spins_jastrow = jnp.array(cfg.spins)
    #jax.debug.print("spins:{}", spins_jastrow)
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins_jastrow,
                                                                                            nelectrons=6)
    #jax.debug.print("parallel_indices:{}", parallel_indices)
    g = jnp.array(cfg.g)
    k = jnp.array(cfg.k)
    layer_dims = jnp.array(cfg.layer_dims)
    charges = jnp.array(cfg.charges)
    atoms = jnp.array(cfg.atoms)
    pos = jnp.array(cfg.pos)
    #jax.debug.print("g:{}", g)
    kan_init, kan_apply = make_kan_net(nspins=(3, 3),
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
                                       layer_dims=layer_dims)

    seed = 42
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    params = kan_init(subkey)
    signed_network = kan_apply
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    spins = jnp.array([cfg.spins])
    jax.debug.print("spins:{}", spins)
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, None, None, None), out_axes=0
    )

    jax.debug.print("pos:{}", pos)
    jax.debug.print("atoms:{}", atoms)
    wavefunction_value = batch_network(params, pos, spins, atoms, charges)
    jax.debug.print("wavefunction_value:{}", wavefunction_value)
    """we need do batch for pos."""
    data = KANetsData(positions=pos, spins=spins, atoms=atoms, charges=charges)

    monte_carlo = make_mcmc_step(batch_network=batch_network,
                                 batch_per_device=2,
                                 steps=10,
                                 atoms=atoms,
                                 blocks=1)
    key, monte_carlo_key = jax.random.split(subkey)
    new_data = monte_carlo(params, data, monte_carlo_key, 0.1)







