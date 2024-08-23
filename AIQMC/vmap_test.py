import jax
import jax.numpy as jnp

xs =jnp.arange(5.)
z = jnp.arange(5.)
def f(x, z):
    print('x', x)
    jax.debug.print("vmap_x: {}", x)
    jax.debug.print("z: {}", z)
    y = jnp.sin(x) + jnp.sin(z)
    jax.debug.print("y: {}", y)
    return y
jax.vmap(f)(xs, z)
# Prints: x: 0.0
#         x: 1.0
#         x: 2.0
#         y: 0.0
#         y: 0.841471
#         y: 0.9092974