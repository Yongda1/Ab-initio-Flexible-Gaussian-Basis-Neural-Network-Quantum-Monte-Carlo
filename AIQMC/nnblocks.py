"""
This is the initialization module of neural network.
"""

import functools
import itertools
from typing import MutableMapping, Optional, Sequence, Tuple
#import chex. something is wrong from this library chex.
import jax
import jax.numpy as jnp


def init_linear_layer(key: jax.random.PRNGKey, in_dim: int, out_dim: int, include_bias: bool = True) \
        -> MutableMapping[str, jnp.ndarray]:
    """Initialise parameters for a linear layer, xw+b"""
    key1, key2 = jax.random.split(key)
    weight = (jax.random.normal(key1, shape=(in_dim, out_dim)) / jnp.sqrt(float(in_dim)))
    if include_bias:
        bias = jax.random.normal(key2, shape=(out_dim,))
        return {'w': weight, 'b': bias}
    else:
        return {'w': weight}


def linear_layer(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Evaluates a linear layer, xw+b. Here."""
    #jax.debug.print("vmap_w:{}", w)
    #jax.debug.print("vmap_b:{}", b)
    #jax.debug.print("vmap_x:{}", x)

    y = jnp.dot(x, w) + b
    #jax.debug.print("y:{}", y)
    return y

def linear_layer_no_b(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Evaluates a linear layer, xw. Here."""
    y = jnp.dot(x, w)
    return y


vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)
vmap_linearlayer_no_b = jax.vmap(linear_layer_no_b, in_axes=(0, None), out_axes=0)
"""we map the function along x axis, i.e. along the electrons."""


def slogdet(x):
    """computes sign and log of determinants of matrices
    different from Ferminet, we only consider the determinants which have more rows and columns."""
    sign, logabsdet = jnp.linalg.slogdet(x)
    return sign, logabsdet, jnp.angle(sign)


def logdet_matmul(xs: Sequence[jnp.ndarray], w: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """return sum_i w_i D_i
    today, we need solve this problem. Here ,we also use long exp sum trick 19.08.2024.
    notes: this function is only working for multi-determinants"""
    phase_in = []
    logabsdet = []
    for x in xs:
        sign, logabs = slogdet(x)
        phase_in.append(sign)
        logabsdet.append(logabs)

    maxlogabsdet = jnp.max(jnp.array(logabsdet))
    det = jnp.array(phase_in) * jnp.exp(jnp.array(logabsdet) - maxlogabsdet)
    result = jnp.dot(det, w)
    phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogabsdet
    return phase_out, log_out