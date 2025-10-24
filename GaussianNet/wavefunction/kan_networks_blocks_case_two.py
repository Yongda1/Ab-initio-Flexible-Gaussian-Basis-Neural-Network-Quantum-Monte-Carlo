import jax.numpy as jnp
import jax
import chex
from typing import MutableMapping, Optional, Sequence, Tuple

""" Here, currently, we set grid update function aside. 
    We need make the case two.17.10.2025.
"""

def init_ka_layer(key: chex.PRNGKey,
                  n_in: int,
                  n_out: int,
                  g: int,
                  k: int,
                  add_residual: bool = True,
                  add_bias: bool = True,
                  external_weights: bool = True,) -> MutableMapping[str, jnp.ndarray]:
    """Initialises parameters for a KA layer.
    g: the number of grid.
    k: the order of spline."""
    key_basis, key_residual, key_external_weights, key_bias = jax.random.split(key, 4)
    c_basis = jax.random.normal(key_basis, shape=(n_in * n_out, g + k))
    #jax.debug.print("c_basis:{}", c_basis)

    if add_residual and external_weights and add_bias:
        c_res = jax.random.normal(key_residual, shape=(n_out, n_in))
        c_spl = jax.random.normal(key_external_weights, shape=(n_out, n_in))
        bias = jax.random.normal(key_bias, shape=(n_out,))
        return {'c_basis': c_basis, 'c_res': c_res, 'c_spl': c_spl, 'bias': bias,}
    else:
        return {'c_basis': c_basis,}

def init_grid(n_in: int,
              n_out: int,
              g: int,
              k: int,
              grid_range: jnp.ndarray,) -> jnp.ndarray:
    """we also need initialize the gird for each layer."""
    h = (grid_range[1] - grid_range[0]) / g
    grid = jnp.arange(-k, g + k + 1) * h + grid_range[0]
    grid = jnp.expand_dims(grid, 0)
    grid = jnp.tile(grid, (n_in * n_out, 1))
    return grid

def spline_each_layer(x: jnp.ndarray,
                      n_in: int,
                      n_out: int,
                      k: int,
                      grid: jnp.ndarray,):
    """
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(n_out, )).reshape((batch, n_in * n_out))
    x_ext = jnp.transpose(x_ext, (1, 0))
    grid = jnp.expand_dims(grid, axis=2)
    x = jnp.expand_dims(x_ext, axis=1)
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    for K in range(1, k + 1):
        left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

    be careful that currently we are running the vector on each edge.
    the shape of basis_spline should be (n_in*n_out, k+g, number of features)
    And we do not need make batched version of KANets. Therefore, we remove the parts about "batch".
    """
    x_scalar = x
    '''
    x_test = x
    #jax.debug.print("x_test:{}", x_test)
    nfeatures = x_test.shape[1]
    #jax.debug.print("nfeatures:{}", nfeatures)
    x_ext_test = jnp.tile(x_test, (1, n_out)).reshape((n_in*n_out, nfeatures))
    #jax.debug.print("x_ext_test:{}", x_ext_test)
    x = jnp.expand_dims(x_ext_test, axis=1)
    grid = jnp.expand_dims(grid, axis=2)
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    '''
    """here, for each edge, we only set one scalar input. 17.20.2025."""

    jax.debug.print("x_scalar:{}", x_scalar)
    x_scalar = jnp.reshape(x_scalar, (-1, 1))
    jax.debug.print("x_scalar:{}", x_scalar)
    x = jnp.expand_dims(x_scalar, axis=1)
    grid = jnp.expand_dims(grid, axis=2)
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)


    for K in range(1, k + 1):
        left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

    #jax.debug.print("basis_splines:{}", basis_splines)
    return basis_splines


def multiply_bi_spl_w(bi: jnp.ndarray, spl_w: jnp.ndarray):
    value = jnp.dot(bi , spl_w)
    return value

"""because we are calculating the vector on every node, the map axis should be (1, None)"""
multiply_Bi_spl_w_vmap = jax.vmap(multiply_bi_spl_w, in_axes=(0, 0))

def residual_function(x: jnp.ndarray,):
    return x/(1+jnp.exp(-x))

def residual_cres(residual_value: jnp.ndarray, c_res: jnp.ndarray):
    return residual_value * c_res

residual_cres_vmap = jax.vmap(jax.vmap(residual_cres, in_axes=(0, 0)), in_axes=(0, 0))

def forward_each_layer(x: jnp.ndarray,
                       n_in: int,
                       n_out: int,
                       g: int,
                       k: int,
                       grid_range: jnp.ndarray,
                       c_basis: jnp.ndarray,
                       c_spl: jnp.ndarray,
                       bias: jnp.ndarray,
                       c_res: jnp.ndarray,):
    nfeatures = x.shape[1]
    grid = init_grid(n_in = n_in, n_out = n_out, g = g, k = k, grid_range = grid_range)
    Bi = spline_each_layer(x, n_in, n_out, k, grid)
    """to do in the afternoon."""
    jax.debug.print("c_basis:{}", c_basis)
    c_spl = jnp.reshape(c_spl, (n_in * n_out, 1))
    #jax.debug.print("c_spl:{}", c_spl)
    #spl_w = c_basis * c_spl[..., None]
    spl_w = c_basis * c_spl
    jax.debug.print("spl_w:{}", spl_w)
    Bi = jnp.reshape(Bi, (n_in * n_out, k+g))
    jax.debug.print("Bi:{}", Bi)
    value = multiply_Bi_spl_w_vmap(Bi, spl_w).reshape((n_in, n_out))
    #jax.debug.print("value:{}", value)
    #value = jnp.reshape(value, (batch, n_in, n_out, 3))
    jax.debug.print("value:{}", value)

    if c_res is not None:
        jax.debug.print('x:{}',x)
        x = jnp.reshape(x, (n_in, n_out))
        residual_value = residual_function(x)
        #jax.debug.print("residual_function:{}", residual_value)
        #jax.debug.print("c_res:{}", c_res)
        #residual_value = jnp.tile(residual_value, (1, n_out)).reshape((n_in, n_out, nfeatures))
        jax.debug.print("residual_value:{}", residual_value)
        c_res = jnp.reshape(c_res, (n_in, n_out))
        jax.debug.print("c_res:{}", c_res)
        #residual_value = residual_cres_vmap(residual_value, c_res)
        #jax.debug.print("residual_value:{}", residual_value)
        residual_value = residual_value * c_res
        #jax.debug.print("residual_value:{}", residual_value)
        #jax.debug.print("value:{}", value)
        value = residual_value + value
        #value = jnp.reshape(value, (n_in, n_out, nfeatures))
        jax.debug.print("value:{}", value)
        value = jnp.sum(value, axis=0)
        jax.debug.print("value:{}", value)
    else:
        value = jnp.sum(value, axis=0)
    """so far, I have finished the construction of one layer of KANets including forward process and parameters initialization.11.10.2025.
    We will check it again in next week."""
    return value






'''
  key1, key2 = jax.random.split(key)
  weight = (
      jax.random.normal(key1, shape=(in_dim, out_dim)) /
      jnp.sqrt(float(in_dim)))
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {'w': weight, 'b': bias}
  else:
    return {'w': weight}'''


seed = 23
key = jax.random.PRNGKey(seed)
"""for example, only one atom and three electrons. The number of features is 4. 17.10.2025."""
input = jnp.array([[0.1, 0.2, 0.1, 0.2], [0.2, 0.2, 0.2, 0.3],[0.3, 0.3, 0.3, 0.4]])

params = init_ka_layer(key=key, n_in=3, n_out=4, g=3, k=3, add_residual=True, add_bias=True, external_weights=True)
output = forward_each_layer(x=input, n_in=3, n_out=4, g=3, k=3, grid_range=jnp.array([0, 1]),
                            c_basis = params['c_basis'], c_spl = params['c_spl'], bias = params['bias'], c_res = params['c_res'] )