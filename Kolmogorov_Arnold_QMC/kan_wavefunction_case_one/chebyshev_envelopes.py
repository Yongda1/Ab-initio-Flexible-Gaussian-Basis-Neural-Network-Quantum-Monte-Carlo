import jax.numpy as jnp
import jax
import chex
from typing import MutableMapping, Optional, Sequence, Tuple

#from setuptools.dist import check_extras

"""in this module, we finish the chebyshev polynomial as the the basis functions.
We need add this on the envelope functions. However, we need change the its structure.18.11.2025.
to be continued....."""

def init_chebyshev(key: chex.PRNGKey,
                   n_in: int,
                   n_out: int,
                   d: int,
                   add_residual: bool = True,
                   add_bias: bool = True,
                   external_weights: bool = True,
                   ):
    """initialize the parameters for chebyshev polynomial basis functions.
    we only consider the situation that all bool types are true.17.11.2025.
    we need rewrite the initialization."""
    key_basis, key_residual, key_external_weights, key_bias = jax.random.split(key, 4)
    ext_dim = d if add_bias else d+1
    std = 1.0/jnp.sqrt(n_in * ext_dim)
    c_basis = jax.nn.initializers.truncated_normal(stddev=std,)(key_basis, (n_in, ext_dim))
    #jax.debug.print("c_basis:{}", c_basis)
    c_res = jax.nn.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(key_residual, (n_in, 1))
    #jax.debug.print("c_res:{}", c_res)
    bias = jnp.zeros(n_in)
    #jax.debug.print("bias:{}", bias)
    c_ext = jnp.ones((n_in,))
    return {'c_basis': c_basis, 'c_res': c_res, 'c_ext': c_ext, 'bias': bias, }


def chebyshev_polynomial_each_layer(x: jnp.ndarray,
                                    n_in: int,
                                    n_out: int,
                                    d: int, ):
    batch = x.shape[0]
    x = jnp.tanh(x)
    #jax.debug.print("x:{}", x)
    x = jnp.expand_dims(x, axis=-1)
    x = jnp.tile(x, (1, 1, d+1))
    x = jnp.arccos(x)
    x *= jnp.arange(d+1)
    cheb_value = jnp.cos(x)
    #jax.debug.print("cheb_value:{}", cheb_value)
    return cheb_value[:, :, 1:]

def residual(x: jnp.ndarray,):
    """this is a part of the envelope function. it can be any format as you want."""
    return jnp.exp(-x)




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
    #jax.debug.print("Bi:{}", Bi)
    #jax.debug.print("c_basis:{}", c_basis)
    #jax.debug.print("c_ext:{}", c_ext)
    act_w = c_basis * c_ext[..., None]
    #jax.debug.print("act_w:{}", act_w[None, ...])
    y = Bi * act_w
    y = jnp.sum(y, axis=-1)
    #jax.debug.print("y:{}", y)
    if c_res is not None:
        res = residual(x)
        res_w = c_res.reshape(1, n_in)
        #jax.debug.print("res:{}", res)
        #jax.debug.print("res_w:{}", res_w)
        full_res = res_w * res
        y *= full_res
        #jax.debug.print("y:{}", y)
    #jax.debug.print("bias:{}", bias)
    if bias is not None:
       y += bias[None, ...]
    #jax.debug.print("y:{}", y)
    return y

"""
'''this part is for debugging the chebyshev polynomial basis functions.'''
seed = 23
key = jax.random.PRNGKey(seed)
input = jnp.array([[0.1, 0.2, 0.1,], [0.2, 0.2, 0.2,]])
params = init_chebyshev(key, 3, 4, 5,)
output = forward_each_layer(x=input, n_in=3, n_out=4, d=5,
                            c_basis = params['c_basis'], c_ext = params['c_ext'], bias = params['bias'], c_res = params['c_res'] )
#jax.debug.print("output:{}", output)
"""