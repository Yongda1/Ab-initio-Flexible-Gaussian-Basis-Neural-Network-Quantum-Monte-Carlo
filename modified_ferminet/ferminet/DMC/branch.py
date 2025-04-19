import logging
import chex
import jax
import kfac_jax
import jax.numpy as jnp
from typing import Sequence
from modified_ferminet.ferminet import networks as nn


def branch(data: nn.FermiNetData, weights: jnp.array, key: chex.PRNGKey):
    """after one block run, run branch to update the information for weights and configurations."""
    #key, subkey = kfac_jax.utils.p_split(key)
    n = data.positions.shape[0]
    #jax.debug.print("subkey:{}", key)
    #jax.debug.print("data:{}", data)
    #jax.debug.print("weights:{}", weights)
    probability = jnp.cumsum(weights)
    #jax.debug.print("probability:{}", probability)
    wtot = probability[-1]
    #jax.debug.print("wtot:{}", wtot)
    base = jax.random.uniform(key) * wtot
    #jax.debug.print("base:{}", base)
    newinds = jnp.searchsorted(probability, (base + jnp.linspace(0, wtot, n, endpoint=False)) % wtot)
    #jax.debug.print("newinds:{}", newinds)
    """to be continued... 12.2.2025. we need spend some time on stochastic comb.
    Now, we got some problems. How to remove wrong walkers or reorder the walkers? Currently, we only put in on numpy."""

    #jax.debug.print("data.position:{}", data.positions)
    #x1 = data.positions
    #pos = jnp.transpose(x1, list(newinds))
    #jax.debug.print("pos:{}", pos)
    weights = wtot/n
    #jax.debug.print("weights:{}", weights)
    return weights, newinds