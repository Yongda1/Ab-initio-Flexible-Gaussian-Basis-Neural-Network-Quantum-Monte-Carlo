import jax.numpy as jnp
import jax
import chex
from typing import MutableMapping, Optional, Sequence, Tuple
"""to be continued...16.10.2025. This module is not necessary. we can make a simple one in kan_networks."""


"""we need make a small modification to KANet to construct orbitals."""
def init_ka_orbitals_layer(key: chex.PRNGKey,
                           n_in: int,
                           n_out: int,
                           g: int,
                           k: int,) -> MutableMapping[str, jnp.ndarray]:
    """Initialises parameters for a KA layer.
    g: the number of grid.
    k: the order of spline."""
    key_basis, key_residual, key_external_weights, key_bias = jax.random.split(key, 4)
    c_basis = jax.random.normal(key_basis, shape=(n_in * n_out, g + k))
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
    batch = x.shape[0]
    jax.debug.print("batch:{}", batch)
    x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(n_out, )).reshape((batch, n_in * n_out))
    x_ext = jnp.transpose(x_ext, (1, 0))
    grid = jnp.expand_dims(grid, axis=2)
    x = jnp.expand_dims(x_ext, axis=1)
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
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
                       c_basis: jnp.ndarray,):
    """we need make a large modification to single layer of KANets.16.10.2025."""
    batch = x.shape[0]
    grid = init_grid(n_in = n_in, n_out = n_out, g = g, k = k, grid_range = grid_range)
    Bi = spline_each_layer(x, n_in, n_out, k, grid)
    """the Bi is the value of the spline functions on the edge. The shape is the (n_in * n_out, batch, G+K) """
    #Bi = jnp.reshape(Bi, (-1))
    #c_basis = jnp.reshape(c_basis, (-1))
    Bi = jnp.reshape(Bi, (k+g, -1)).transpose()
    #jax.debug.print("Bi:{}", Bi)
    #jax.debug.print("c_basis:{}", c_basis)
    value = jnp.sum(Bi * c_basis, axis = -1)
    #jax.debug.print("value:{}", value)
    return value

'''
seed = 23
key = jax.random.PRNGKey(seed)
"""for example, only one atom and six electrons."""
input = jnp.array([[0.17320508],
                   [0.17320508],
                   [0.17320508]])

params = init_ka_orbitals_layer(key=key, n_in=1, n_out=1, g=3, k=3)
output = forward_each_layer(x=input, n_in=1, n_out=1, g=3, k=3, grid_range=jnp.array([0, 1]), c_basis = params['c_basis'])
jax.debug.print("output:{}", output)'''