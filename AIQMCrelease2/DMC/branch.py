import logging
import jax.numpy as jnp
from AIQMCrelease2.wavefunction import nn


def branch(data: nn.AINetData, weights: jnp.array, batch_size: int):
    if jnp.any(weights > 2.0):
        logging.warning("Some weights are larger than 2")
    probability = jnp.cumsum(weights)
    wtot = probability[-1]
    base = jnp.random.rand() * wtot
    newinds = jnp.searchsorted(
        probability, (base + jnp.linspace(0, wtot, batch_size, endpoint=False)) % wtot
    )
    unique, counts = jnp.unique(newinds, return_counts=True)

    return