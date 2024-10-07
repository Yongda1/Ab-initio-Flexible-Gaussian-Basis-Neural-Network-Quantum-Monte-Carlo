from typing import Tuple
import chex
#from AIQMC import constants
from hamiltonian import local_energy
from AIQMC import nn
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol
from AIQMC import main

signednetwork, data, batchparams, batchphase, batchnetwork, mc_step, localenergy = main.main()
print("data.positions", data.positions)
print("params", batchparams)
jax.debug.print("mc_step:{}", mc_step)
jax.debug.print("local_energy:{}", localenergy)
key = jax.random.PRNGKey(seed=1)

"""Before we go into the loss function, we have to finish the hamiltonian first."""
"""we already finsihed the hamiltonian module. today, we try to do this."""

@chex.dataclass
class AuxiliaryLossData:
    variance: jax.Array
    local_energy: jax.Array
    grad_local_energy: jax.Array | None

class LossAINet(Protocol):

    def __call__(self, params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) -> Tuple[jnp.array, AuxiliaryLossData]:
        """Evaluagtes the total energy of the network for a batch of configurations."""


def make_loss(network: nn.AINetLike, local_energy, data: nn.AINetData, complex_output: bool = False) -> LossAINet:
    """our local_energy function from hamiltonian module is already batched version.
    We dont need rewrite it here."""
    batch_local_energy = local_energy(f=signednetwork, complex_number=True)
    batch_signed_network = jax.vmap(network, in_axes=(None, 1, 1, 1), out_axes=0)

    @jax.custom_jvp
    def total_energy(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData):
        e_l = batch_local_energy(batch_size=4, ndim=3, batchparams=batchparams, key=key, data=data)
        print('-----loss-----')
        jax.debug.print("e_l:{}", e_l)
        loss = jnp.mean(jnp.mean(e_l, axis=-1), axis=-1)
        jax.debug.print("loss:{}", loss)
        loss_diff = e_l - loss
        variance = jnp.mean(jnp.mean(loss_diff*jnp.conj(loss_diff), axis=-1), axis=-1)
        jax.debug.print("variance:{}", variance)
        return loss, AuxiliaryLossData(variance=variance, local_energy=e_l, grad_local_energy=None)



    """here, we probably need redefine the differentiation rules for the mean value of total energy."""
    @total_energy.defjvp()
    def total_energy_jvp(primals, tangents):
        params, key, data = primals
        loss, aux_data = total_energy(params=batchparams, key=key, data=data)
        diff = aux_data.local_energy - loss
        data = primals[2]
        data_tangents = tangents[2]
        primals = (primals[0], data.positions, data.atoms, data.charges)
        tangents = (tangents[0], data_tangents.positions, data_tangents.atoms, data_tangents.charges)
        psi_primal, psi_tangent = jax.jvp(batch_signed_network, primals, tangents)
        clipped_el = diff + aux_data.local_energy
        term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) + jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
        term2 = jnp.sum(aux_data.local_energy * psi_tangent.real)
        kfac_jax.register_normal_predictive_distribution(psi_primal.real[:, None])
        primals_out = loss.real, aux_data
        """the following codes need to be rewrite 7/10/2024."""
        device_batch_size = jnp.shape(aux_data.local_energy)
        tangents_out = ((term1 - 2 * term2).real / device_batch_size)
        """to be continued..."""
        return primals_out, tangents_out

    return total_energy




output = make_loss(signednetwork, local_energy=local_energy, data=data, complex_output=True)
