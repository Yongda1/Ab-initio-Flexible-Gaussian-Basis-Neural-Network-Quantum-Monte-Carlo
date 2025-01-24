import jax
import numpy as np
import jax.numpy as jnp

tstep = 0.1
nconf = 4
gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
print("gauss", gauss)


def Gaussian(x: float, mu: float, sigma: float):
    return 1/jnp.sqrt(2 * jnp. pi * (sigma**2)) * jnp.exp(-1 * ((x-mu)**2 / (2 * sigma**2)))


mu = 0
sigma = jnp.sqrt(tstep)

