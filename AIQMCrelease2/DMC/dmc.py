from AIQMCrelease2.wavefunction import nn
import chex
import jax.numpy as jnp

def dmc_propagate(params: nn.ParamTree,
                  key: chex.PRNGKey,
                  data: nn.AINetData,
                  batch_size: int,
                  weights: jnp.array,
                  tstep: float,
                  branchcut_start: jnp.array,
                  e_trial: jnp.array,
                  e_est: jnp.array,
                  nsteps: int,):
    """we start to constuct the one loop dmc progagation process."""