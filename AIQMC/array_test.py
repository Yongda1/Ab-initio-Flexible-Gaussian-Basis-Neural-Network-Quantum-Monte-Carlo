import jax
import jax.numpy as jnp

vmap_h_in = jnp.array([[[[[1, 2, 3], [4, 5, 6]]]]])
b = jnp.array([[1, 1, 1]])
w = jnp.array([[[1, 1, 1], [2, 3, 2], [2, 2, 4]]])
print(jnp.dot(vmap_h_in, w))
output = jnp.dot(vmap_h_in, w) + b
print(output)