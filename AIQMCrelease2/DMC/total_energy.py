from typing import Tuple
import chex
from AIQMCrelease2 import constants
from AIQMCrelease2.Energy import pphamiltonian
from AIQMCrelease2.wavefunction import nn
import jax
import jax.numpy as jnp


def calculate_total_energy(local_energy: pphamiltonian.LocalEnergy,):
    """Creates the loss function, including custom gradients."""
    batch_local_energy = jax.vmap(
        local_energy,
        in_axes=(
            None,
            0,
            nn.AINetData(positions=0, spins=0, atoms=0, charges=0),
        ),
        out_axes=(0, 0)
    )

    def total_energy(
            params: nn.ParamTree,
            key: chex.PRNGKey,
            data: nn.AINetData,):
        """Evaluates the total energy of the network for a batch of configurations."""
        keys = jax.random.split(key, num=data.positions.shape[0])
        e_l, e_l_mat = batch_local_energy(params, keys, data)
        loss = constants.pmean(jnp.mean(e_l))
        loss_diff = e_l - loss
        variance = constants.pmean(jnp.mean(loss_diff * jnp.conj(loss_diff)))
        #jax.debug.print("loss:{}", loss)
        return e_l, variance
    return total_energy