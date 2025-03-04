"""create the function for two dimensional ewald summation."""
import jax
import jax.numpy as jnp
from AIQMCrelease3.Energy import pphamiltonian

atoms = jnp.array([[0.0, 0.0, 0.0], [2/3, 1/3, 0.0]])
charges = jnp.array([4.0, 4.0])
spins = jnp.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
lattice = jnp.array([[0.5 * jnp.sqrt(3), 0.5, 0],
                     [0.5 * jnp.sqrt(3), -0.5, 0],
                     [0, 0, 10]])
jax.debug.print("lattice:{}", jnp.linalg.norm(lattice, axis=-1))
