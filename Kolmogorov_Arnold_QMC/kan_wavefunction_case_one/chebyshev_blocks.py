import jax.numpy as jnp
import jax
import chex
from typing import MutableMapping, Optional, Sequence, Tuple

#from setuptools.dist import check_extras

"""in this module, we finish the chebyshev polynomial as the the basis functions."""
"""let us try to remove the residual functions. It probably is not necessary for our calculation."""

def init_chebyshev(key: chex.PRNGKey,
                   n_in: int,
                   n_out: int,
                   d: int,
                   add_residual: bool = False,
                   add_bias: bool = True,
                   external_weights: bool = True,
                   ):
    """initialize the parameters for chebyshev polynomial basis functions.
    we only consider the situation that all bool types are true.17.11.2025."""
    key_basis, key_residual, key_external_weights, key_bias = jax.random.split(key, 4)
    ext_dim = d if add_bias else d+1
    std = 1.0/jnp.sqrt(n_in * ext_dim)
    c_basis = jax.nn.initializers.truncated_normal(stddev=std,)(key_basis, (n_out, n_in, ext_dim))
    bias = jnp.zeros(n_out)
    c_ext = jnp.ones((n_out, n_in))
    if add_residual:
        c_res = jax.nn.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(key_residual, (n_out, n_in))
        return {'c_basis': c_basis, 'c_res': c_res, 'c_ext': c_ext, 'bias': bias, }
    else:
        return {'c_basis': c_basis, 'c_ext': c_ext, 'bias': bias, 'c_res': None,}


def chebyshev_polynomial_each_layer(x: jnp.ndarray,
                                    n_in: int,
                                    n_out: int,
                                    d: int, ):
    batch = x.shape[0]
    x = jnp.tanh(x) # this line is important to make the net stable.
    x = jnp.expand_dims(x, axis=-1)
    x = jnp.tile(x, (1, 1, d+1))
    x = jnp.arccos(x)
    x *= jnp.arange(d+1)
    cheb_value = jnp.cos(x)
    return cheb_value[:, :, 1:]

def residual(x: jnp.ndarray,):
    return x
    #return x/(1+jnp.exp(-x))




def forward_each_layer(x: jnp.ndarray,
                       n_in: int,
                       n_out: int,
                       d: int,
                       c_basis: jnp.ndarray,
                       c_ext: jnp.ndarray,
                       bias: jnp.ndarray,
                       c_res: jnp.ndarray,):
    batch = x.shape[0]
    Bi = chebyshev_polynomial_each_layer(x, n_in, n_out, d)
    act = Bi.reshape(batch, -1)
    #jax.debug.print("act:{}", act)
    act_w = c_basis * c_ext[..., None]
    act_w = act_w.reshape(n_out, -1)
    #jax.debug.print("act_w:{}", act_w)
    y = jnp.matmul(act, act_w.T)
    if c_res is not None:
        res = residual(x)
        res_w = c_res
        full_res = jnp.matmul(res, res_w.T) # (batch, n_out)
        """consider to change + to *, also change the residual function."""
        y += full_res

    if bias is not None:
       y += bias

    return y

"""
'''this part is for debugging the chebyshev polynomial basis functions.'''
seed = 23
key = jax.random.PRNGKey(seed)
input = jnp.array([[0.1, 0.2, 0.1,], [0.2, 0.2, 0.2,]])
params = init_chebyshev(key, 3, 4, 5,)
output = forward_each_layer(x=input, n_in=3, n_out=4, d=5,
                            c_basis = params['c_basis'], c_ext = params['c_ext'], bias = params['bias'], c_res = params['c_res'] )
jax.debug.print("output:{}", output)
"""