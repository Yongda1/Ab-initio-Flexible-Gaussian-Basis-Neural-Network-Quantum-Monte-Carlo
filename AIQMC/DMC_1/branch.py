import logging
import chex
import jax
import kfac_jax
import jax.numpy as jnp
from typing import Sequence
from AIQMC.wavefunction import networks as nn


def branch(data: nn.FermiNetData, weights: jnp.array, key: chex.PRNGKey):
    """after one block run, run branch to update the information for weights and configurations."""
    key, subkey = jax.random.split(key)
    n = data.positions.shape[0]
    probability = jnp.cumsum(weights)
    wtot = probability[-1]
    #jax.debug.print("wtot:{}", wtot)
    base = jax.random.uniform(subkey) * wtot
    #jax.debug.print("base:{}", base)
    newinds = jnp.searchsorted(probability, (base + jnp.linspace(0, wtot, n, endpoint=False)) % wtot)
    weights = wtot/n
    return weights, newinds, subkey