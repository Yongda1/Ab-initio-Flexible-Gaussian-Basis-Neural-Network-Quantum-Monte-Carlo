from typing import Tuple
import chex
from AIQMCbatch import constants
from AIQMCbatch import hamiltonianA
from AIQMCbatch import nn
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol
from AIQMCbatch import main_kfac
from AIQMCbatch.utils import utils

signed_network, data, batch_params, batch_network, batch_phase_network = main_kfac.main()
#jax.debug.print("local_energy:{}", local_energy)
key = jax.random.PRNGKey(seed=1)

"""Before we go into the loss function, we have to finish the hamiltonian first."""
"""we already finsihed the hamiltonian module. today, we try to do this.

We have a large problem here. Because the kfac does not accept the pmap in the loss function,
the hamiltonian and loss module have to be non_batch version. But with vmap 1.11.2024. Then,we turn to the hamiltonian module.
"""

@chex.dataclass
class AuxiliaryLossData:
    variance: jax.Array
    local_energy: jax.Array
    grad_local_energy: jax.Array | None


class LossAINet(Protocol):

    def __call__(self, params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) \
            -> Tuple[jnp.array, AuxiliaryLossData]:
        """Evaluagtes the total energy of the network for a batch of configurations."""


def make_loss(network: nn.AINetLike, local_energy: hamiltonianA.LocalEnergy) -> LossAINet:
    """our local_energy function from hamiltonian module is already batched version.
    We dont need rewrite it here."""

    batch_local_energy = jax.pmap(jax.vmap(local_energy(network), in_axes=(None, None, 0, 0, 0), out_axes=0))
    logabs_f = utils.select_output(network, 1)
    angle_f = utils.select_output(network, 2)
    batch_signed_network_logabs = jax.pmap(jax.vmap(logabs_f, in_axes=(None, 0, 0, 0), out_axes=0),
                                           in_axes=0, out_axes=0)
    batch_signed_network_angle = jax.pmap(jax.vmap(angle_f, in_axes=(None, 0, 0, 0), out_axes=0),
                                          in_axes=0, out_axes=0)


    @jax.custom_jvp
    def total_energy(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        keys = jax.random.split(key, num=data.positions.shape[0])
        e_l = batch_local_energy(params, keys, data.positions, data.atoms, data.charges)
        jax.debug.print("e_l:{}", e_l)
        loss = constants.pmean(jnp.mean(e_l))
        jax.debug.print("loss:{}", loss)
        loss_diff = e_l - loss
        variance = jnp.mean(loss_diff*jnp.conj(loss_diff), axis=-1)
        jax.debug.print("variance:{}", variance)
        return loss, AuxiliaryLossData(variance=variance, local_energy=e_l, grad_local_energy=None)


    """here, we probably need redefine the differentiation rules for the mean value of total energy."""
    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, key, data = primals
        loss, aux_data = total_energy(params, key, data)
        data = primals[2]
        data_tangents = tangents[2]
        primals = (primals[0], data.positions, data.atoms, data.charges)
        tangents = (
            tangents[0],
            data_tangents.positions,
            data_tangents.atoms,
            data_tangents.charges,
        )
        psi_primal_logabs, psi_tangent_logabs = jax.jvp(batch_signed_network_logabs, primals, tangents)
        psi_primal_angle, psi_tangent_angle = jax.jvp(batch_signed_network_angle, primals, tangents)
        psi_primal = psi_primal_logabs + 1.j * psi_primal_angle
        psi_tangent = psi_tangent_logabs + 1.j * psi_tangent_angle
        #clipped_el = diff + aux_data.local_energy
        jax.debug.print("psi_tanget:{}", jnp.shape(psi_tangent))
        jax.debug.print("aux_data.local_energy:{}", jnp.shape(aux_data.local_energy))
        term1 = jnp.sum(aux_data.local_energy * jnp.conjugate(psi_tangent_logabs)) + jnp.sum(jnp.conjugate(aux_data.local_energy)*psi_tangent_logabs)
        term2 = jnp.sum(aux_data.local_energy * psi_tangent_logabs.real)
        kfac_jax.register_normal_predictive_distribution(psi_primal.real[:, None])
        primals_out = loss.real, aux_data
        """the following codes need to be rewrite 7/10/2024."""
        device_batch_size = 4
        tangents_out = ((term1 - 2 * term2).real / device_batch_size, aux_data)
        #jax.debug.print("tangents_out:{}", tangents_out)
        return primals_out, tangents_out

    return total_energy




total_energy_test = make_loss(signed_network, hamiltonianA.local_energy)
loss, aux_data = total_energy_test(batch_params, key, data)
val_and_grad = jax.value_and_grad(total_energy_test, argnums=0, has_aux=True)
#jax.debug.print("batch_params:{}", batch_params)
#jax.debug.print("data:{}", data)
output2 = val_and_grad(batch_params, key, data)
jax.debug.print("aux_data:{}", aux_data.local_energy)
logabs_f = utils.select_output(signed_network, 1)
angle_f = utils.select_output(signed_network, 2)
batch_signed_network_logabs = jax.pmap(jax.vmap(logabs_f, in_axes=(None, 0, 0, 0), out_axes=0),
                                           in_axes=0, out_axes=0)
batch_signed_network_angle = jax.pmap(jax.vmap(angle_f, in_axes=(None, 0, 0, 0), out_axes=0),
                                          in_axes=0, out_axes=0)
primals = (batch_params, data.positions, data.atoms, data.charges)
tangents = (batch_params, data.positions, data.atoms, data.charges)
psi_primal_angle, psi_tangent_angle = jax.jvp(batch_signed_network_logabs, primals, tangents)
#print(jvp(total_energy_test)(batchparams, key, data))
psi_primal_logabs, psi_tangent_logabs = jax.jvp(batch_signed_network_logabs, primals, tangents)
jax.debug.print("psi_tangent_angle:{}", psi_tangent_angle)
jax.debug.print("psi_primal_angle:{}", psi_primal_angle)
jax.debug.print("psi_primal_logabs:{}", psi_primal_logabs)
jax.debug.print("psi_tangent_logabs:{}", psi_tangent_logabs)
psi_primal = psi_primal_logabs + 1.j * psi_primal_angle
psi_tangent = psi_tangent_logabs + 1.j * psi_tangent_angle
jax.debug.print("psi_primal:{}", psi_primal)
jax.debug.print("psi_tangent:{}", psi_tangent)
test = jnp.conjugate(psi_tangent)
jax.debug.print("test:{}", test)
#test2 = jnp.dot(aux_data.local_energy, jnp.conjugate(psi_tangent))
term1 = jnp.sum(aux_data.local_energy * jnp.conjugate(psi_tangent_logabs)) + jnp.sum(jnp.conjugate(aux_data.local_energy)*psi_tangent_logabs)
jax.debug.print("aux_data:{}", aux_data.local_energy)
jax.debug.print("test2:{}", aux_data.local_energy * jnp.conjugate(psi_tangent))
term2 = jnp.sum(aux_data.local_energy * psi_tangent_logabs.real)
tangents_out = ((term1 - 2 * term2).real / 4, aux_data)
jax.debug.print("tangents_out:{}", tangents_out)
primals_out = loss.real, aux_data
jax.debug.print("primals_out:{}", primals_out)