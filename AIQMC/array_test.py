import jax
import jax.numpy as jnp
import numpy as np

vmap_h_in = jnp.array([[[[[1, 2, 3], [4, 5, 6]]]]])
b = jnp.array([[1, 1, 1]])
w = jnp.array([[[1, 1, 1], [2, 3, 2], [2, 2, 4]]])

output = jnp.dot(vmap_h_in, w) + b

a = jnp.arange(0, 16.0, 5.0)
print(a)
gauss = np.random.normal(scale=np.sqrt(1), size=(2, 3))
#print(gauss)
forward = np.sum(gauss**2, axis=1)
#print(forward)


