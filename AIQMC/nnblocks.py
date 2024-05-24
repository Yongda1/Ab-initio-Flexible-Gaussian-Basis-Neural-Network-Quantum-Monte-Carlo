"""
This is the initialization module of neural network.
"""

import functools
import itertools
from typing import MutableMapping, Optional, Sequence, Tuple
import chex
import jax
import jax.numpy as jnp


def init_linear_layer(key: chex.PRNGKey, in_dim: int, out_dim: int, include_bias: bool = True) \
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
    """Evaluates a linear layer, xw+b"""
    y = jnp.dot(x, w) + b
    return y


vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)
"""we map the function along x axis"""


def slogdet(x):
    """computes sign and log of determinants of matrices
    different from Ferminet, we only consider the determinants which have more rows and columns."""
    sign, logabsdet = jnp.linalg.slogdet(x)
    return sign, logabsdet


def logdet_matmul(xs: Sequence[jnp.ndarray], w: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """return sum_i w_i D_i
    be careful, we do not use full_det. This means that we use |D_spin_up| * |D_spin_down|. So we can avoid many if.
    Here we do not consider the 1x1 determinant."""
    phase_in, logabsdet = functools.reduce(lambda a, b: (a[0] * b[0], a[1] + b[1]), [slogdet(x) for x in xs])
    maxlogabsdet = jnp.max(logabsdet)
    det = phase_in * jnp.exp(logabsdet - maxlogabsdet)
    result = jnp.matmul(det, w)[0]
    phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogabsdet
    return phase_out, log_out