"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
from AIQMC import nn
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np

"""Tomorrow, we are going to finish the walkers moving part. But differently from FermiNet, we will use the traditional moving strategy.
19.08.2024. no worry, everything will fine."""
def mh_block_update(params: nn.ParamTree, f: nn.LogAINetLike, data: nn.AINetData, key: chex.PRNGKey, atoms: jnp.ndarray, ndim=3, blocks=2, batch_size=4, nelectrons=4):
    key, subkey = jax.random.split(key)



def make_mc_step(batch_network, batch_per_device, steps=10, atoms: jnp.ndarray, ndim=3, blocks=2):
    inner_fun = mh_block_update