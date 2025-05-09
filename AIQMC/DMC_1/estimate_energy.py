import jax.numpy as jnp


def estimate_energy(energy: jnp.array, weights: jnp.array):
    return jnp.average(a=energy, weights=weights)