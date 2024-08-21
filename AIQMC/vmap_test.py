import jax
import jax.numpy as jnp

def linear_layer(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Evaluates a linear layer, xw+b. Here."""
    y = jnp.dot(x, w) + b
    return y

def linear_layer_no_b(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Evaluates a linear layer, xw. Here."""
    y = jnp.dot(x, w)
    return y

vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)
vmap_linearlayer_no_b = jax.vmap(linear_layer_no_b, in_axes=(0, None), out_axes=0)