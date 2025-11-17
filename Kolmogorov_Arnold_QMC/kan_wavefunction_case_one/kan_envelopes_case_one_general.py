import jax.numpy as jnp
import jax
import chex
from typing import MutableMapping, Optional, Sequence, Tuple

"""we need develop such a layer. The input should be coe_eff. 
And each element of input vector only get into one spline function."""

def init_ka_layer(key: chex.PRNGKey,
                  n_in: int,
                  n_out: int, # For the envelope layer, the n_in must be same with n_out.
                  g: int,
                  k: int,
                  add_residual: bool = True,
                  add_bias: bool = True,
                  external_weights: bool = True,) -> MutableMapping[str, jnp.ndarray]:
    """Initialises parameters for an envelope layer. First, we need reconstruct the init of parameters.22.10.2025.
    g: the number of grid.
    k: the order of spline."""
    key_basis, key_residual, key_external_weights, key_bias = jax.random.split(key, 4)
    c_basis = jax.random.normal(key_basis, shape=(n_in, (g + k)))
    #jax.debug.print("c_basis:{}", c_basis)

    if add_residual and external_weights and add_bias:
        c_res = jax.random.normal(key_residual, shape=(n_in,))
        c_spl = jax.random.normal(key_external_weights, shape=(n_in,))
        bias = jax.random.normal(key_bias, shape=(n_in,))
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
    grid = jnp.tile(grid, (n_in, 1))
    return grid

def spline_each_layer(x: jnp.ndarray,
                      n_in: int,
                      n_out: int,
                      k: int,
                      grid: jnp.ndarray,):
    """
    be careful that currently we are running the vector on each edge.
    the shape of basis_spline should be (n_in*n_out, k+g, number of features)
    And we do not need make batched version of KANets. Therefore, we remove the parts about "batch".
    """
    batch = x.shape[0]
    nfeatures = x.shape[1]
    x = jnp.reshape(x, (batch, nfeatures, 1))
    #jax.debug.print("x:{}", x)
    x_ext = x
    grid = jnp.expand_dims(grid, axis=2)
    #jax.debug.print("grid:{}", grid)
    x_ext = jnp.reshape(x_ext, (n_in, -1))
    x = jnp.expand_dims(x_ext, axis=1)
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    """calculate the value of x on the spline basis functions."""
    for K in range(1, k + 1):
        left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

    return basis_splines


def multiply_bi_spl_w(bi: jnp.ndarray, spl_w: jnp.ndarray):
    value = jnp.dot(bi , spl_w)
    return value

"""because we are calculating the vector on every node, the map axis should be (1, None)"""
multiply_Bi_spl_w_vmap = jax.vmap(jax.vmap(multiply_bi_spl_w, in_axes=(1, None)), in_axes=(0, 0))

def residual_function(x: jnp.ndarray,):
    return jnp.exp(-1 * x)

def residual_cres(residual_value: jnp.ndarray, c_res: jnp.ndarray):
    return residual_value*c_res

residual_cres_vmap =jax.vmap(residual_cres, in_axes=(0, None))

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
    batch = x.shape[0]
    nfeatures = x.shape[1]
    #jax.debug.print("nfeatures:{}", nfeatures)
    grid = init_grid(n_in = n_in, n_out = n_out, g = g, k = k, grid_range = grid_range)
    #jax.debug.print("grid:{}", grid)
    Bi = spline_each_layer(x, n_in, n_out, k, grid)
    """the Bi is the value of the spline functions on the edge. The shape is the (n_in, G+K, batch) """
    #jax.debug.print("Bi:{}", Bi)
    """here, we need multiply the c_basis with Bi."""
    c_spl = jnp.reshape(c_spl, (n_in, 1))

    #jax.debug.print("c_spl:{}", c_spl)
    #jax.debug.print("c_basis:{}", c_basis)

    spl_w = c_basis * c_spl
    #jax.debug.print("spl_w:{}", spl_w)
    #jax.debug.print("Bi:{}", Bi)
    value = multiply_Bi_spl_w_vmap(Bi, spl_w)
    #jax.debug.print("value:{}", value)
    value = jnp.transpose(value).reshape(batch, n_in)
    #jax.debug.print("value:{}", value)
    if c_res is not None:
        residual_value = residual_function(x)
        #jax.debug.print("residual_value:{}", residual_value)
        c_res = jnp.reshape(c_res, (n_in))
        #jax.debug.print("c_res:{}", c_res)
        residual_value = residual_cres_vmap(residual_value, c_res)
        #jax.debug.print("residual_value:{}", residual_value)
        value = residual_value + value
        value = jnp.reshape(value, (batch, n_in))
        #jax.debug.print("value:{}", value)
        #jax.debug.print("bias:{}", bias)
        value = value + jnp.expand_dims(bias, axis=0)
        #jax.debug.print("value:{}", value)
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

'''
seed = 23
key = jax.random.PRNGKey(seed)
"""for example, only one atom and six electrons."""
input = jnp.array([[0.1, 0.2, 0.1,], [0.2, 0.2, 0.2,],[0.3, 0.3, 0.3,], [0.4, 0.4, 0.4]])

params = init_ka_layer(key=key, n_in=3, n_out=3, g=3, k=3, add_residual=True, add_bias=True, external_weights=True)
output = forward_each_layer(x=input, n_in=3, n_out=3, g=3, k=3, grid_range=jnp.array([0, 1]),
                            c_basis = params['c_basis'], c_spl = params['c_spl'], bias = params['bias'], c_res = params['c_res'] )
'''