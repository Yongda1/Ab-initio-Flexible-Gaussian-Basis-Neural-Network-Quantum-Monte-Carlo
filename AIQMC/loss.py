from typing import Tuple
import chex
#from AIQMC import constants
#from hamiltonian import local_energy
from AIQMC import nn
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol
#from AIQMC import main
#from jax import jvp, grad
from AIQMC.utils import utils

'''
signednetwork, data, batchparams, batchphase, batchnetwork, mc_step, local_energy = main.main()
print("data.positions", data.positions)
print("data.atoms", data.atoms)
print("data.charges", data.charges)
print("params", batchparams)
jax.debug.print("mc_step:{}", mc_step)
jax.debug.print("local_energy:{}", local_energy)
key = jax.random.PRNGKey(seed=1)
'''
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


def make_loss(network: nn.AINetLike, local_energy, params: nn.ParamTree, data: nn.AINetData, complex_output: bool = False) -> LossAINet:
    """our local_energy function from hamiltonian module is already batched version.
    We dont need rewrite it here."""
    batch_local_energy = local_energy(f=network, complex_number=True)
    logabs_f = utils.select_output(network, 1)
    angle_f = utils.select_output(network, 2)
    batch_signed_network_logabs = jax.vmap(logabs_f, in_axes=(None, 1, 1, 1), out_axes=0)
    batch_signed_network_angle = jax.vmap(angle_f, in_axes=(None, 1, 1, 1), out_axes=0)



    @jax.custom_jvp
    def total_energy(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        jax.debug.print("params:{}", params)
        jax.debug.print("data:{}", data)
        e_l = batch_local_energy(batch_size=4, ndim=3, params=params, key=key, data=data)
        e_l = jnp.mean(e_l, axis=-1)
        loss = jnp.mean(e_l, axis=-1)
        loss_diff = e_l - loss
        variance = jnp.mean(loss_diff*jnp.conj(loss_diff), axis=-1)
        return loss, AuxiliaryLossData(variance=variance, local_energy=e_l, grad_local_energy=None)



    """here, we probably need redefine the differentiation rules for the mean value of total energy."""
    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, key, data = primals
        loss, aux_data = total_energy(params=params, key=key, data=data)
        #jax.debug.print("loss:{}", loss)
        diff = aux_data.local_energy - loss
        #jax.debug.print("diff:{}", diff)
        data = primals[2]
        data_tangents = tangents[2]
        primals = (primals[0], data.positions, data.atoms, data.charges)
        """actually, I dont understand this line."""
        tangents = (tangents[0], data_tangents.positions, data_tangents.atoms, data_tangents.charges)
        """herwe, be aware of the output of batch_signed_network"""
        psi_primal_logabs, psi_tangent_logabs = jax.jvp(batch_signed_network_logabs, primals, tangents)
        psi_primal_angle, psi_tangent_angle = jax.jvp(batch_signed_network_angle, primals, tangents)
        psi_primal = psi_primal_logabs + 1.j * psi_primal_angle
        psi_tangent = psi_tangent_logabs + 1.j * psi_tangent_angle
        clipped_el = diff + aux_data.local_energy
        term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) + jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
        term2 = jnp.sum(aux_data.local_energy * psi_tangent.real)
        kfac_jax.register_normal_predictive_distribution(psi_primal.real[:, None])
        primals_out = loss.real, aux_data
        """the following codes need to be rewrite 7/10/2024."""
        device_batch_size = 1
        tangents_out = ((term1 - 2 * term2).real / device_batch_size, aux_data)
        #jax.debug.print("tangents_out:{}", tangents_out)
        """to be continued...
        we already checked the formula of real version and complex version. 08.10.2024.
        we can finish this module after the conference."""
        return primals_out, tangents_out

    return total_energy




#total_energy_test = make_loss(signednetwork, local_energy=local_energy, params=batchparams, data=data, complex_output=True)
#output = total_energy_test(batchparams, key, data)
#print(jvp(total_energy_test)(batchparams, key, data))
