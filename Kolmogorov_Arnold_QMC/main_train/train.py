"""Currently, we only develop the single stream version.
Because the parallel strategy of JAX changed a lot since last year. We need spend a long time to reconstruct it.
one more thing is that we do not pretrain it currently.

24.10.2025."""

import ml_collections
import jax.numpy as jnp
import jax
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.kan_networks_case_one import make_kan_net
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.spin_indices import jastrow_indices_ee, jastrow_indices_ae


def train(cfg: ml_collections.ConfigDict,):
    spins = jnp.array(cfg.spins)
    jax.debug.print("spins:{}", spins)

