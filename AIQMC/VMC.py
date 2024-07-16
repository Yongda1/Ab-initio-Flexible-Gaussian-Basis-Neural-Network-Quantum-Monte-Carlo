"""In this module, we implement VMC algorithm for AIQMC. We only use one-electron move.
Because one-electron move is more stable than all-electrons move.
Here, we are still not sure which structure should be used for wave function.
No matter what we use as wave function, the shape of output layer should be m*N*3.
m is the number of walkers/configurations. N is the number of electrons. 3 is the dimension of coordinates.
"""
import chex
import jax
import jax.numpy as jnp
from AIQMC import nn

def mh_update(params: nn.ParamTree, f: nn.AINetData, data: nn.AINetData, key: chex.PRNGKey,
              atoms: jnp.ndarray, ndim=3, blocks=1):
    """params: Wavefunction parameters.
            f: Callable with signature f(params, x) which returns the log of wavefunction.
         data: Initial MCMC configurations (batched).
        Returns: x: updated MCMC configurations.
        To implement the one electron move, we need the gradient of wave function with respect to electron position. """
    key, subkey = jax.random.split(key)
    x1 = data.positions
    


def make_mcmc_step(batch_network, batch_per_device, steps_per_block=10, atoms=None, ndim=3, blocks=1):
    """Creates the MCMC step function."""
    inner_fun = mh_update