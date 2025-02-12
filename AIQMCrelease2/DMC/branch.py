import logging
import chex
import jax
import kfac_jax
import jax.numpy as jnp
from AIQMCrelease2.wavefunction import nn


def branch(data: nn.AINetData, weights: jnp.array, key: chex.PRNGKey):
    #key, subkey = kfac_jax.utils.p_split(key)
    n = data.positions.shape[0]
    jax.debug.print("subkey:{}", key)
    jax.debug.print("data:{}", data)
    jax.debug.print("weights:{}", weights)
    probability = jnp.cumsum(weights)
    jax.debug.print("probability:{}", probability)
    wtot = probability[-1]
    jax.debug.print("wtot:{}", wtot)
    base = jax.random.uniform(key) * wtot
    jax.debug.print("base:{}", base)
    newinds = jnp.searchsorted(probability, (base + jnp.linspace(0, wtot, n, endpoint=False)) % wtot)
    jax.debug.print("newinds:{}", newinds)
    """to be continued... 12.2.2025. we need spend some time on stochastic comb"""
    #unique, conunts = jnp.unique(newinds)
    weights = wtot/n
    jax.debug.print("weights:{}", weights)
    return None