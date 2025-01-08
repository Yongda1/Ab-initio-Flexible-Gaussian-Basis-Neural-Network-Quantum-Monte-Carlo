
from typing import Tuple
import chex
from AIQMCrelease1 import constants
from AIQMCrelease1.Energy import hamiltonian
from AIQMCrelease1.wavefunction import nn
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
    variance: jax.Array
    local_energy: jax.Array
    grad_local_energy: jax.Array | None
    local_energy_mat: jax.Array | None


class LossAINet(Protocol):

    def __call__(self, params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) \
            -> Tuple[jnp.array, AuxiliaryLossData]:
        """Evaluagtes the total energy of the network for a batch of configurations."""


def make_loss(network: nn.LogAINetLike, local_energy: hamiltonian.LocalEnergy) -> LossAINet:
    """our local_energy function from hamiltonian module is already batched version.
    We dont need rewrite it here."""
    batch_local_energy = jax.vmap(local_energy, in_axes=(None, None, nn.AINetData(positions=0, atoms=0, charges=0),
                                                         ), out_axes=(0, 0))
    batch_signed_network_log = jax.vmap(network, in_axes=(None, 0, 0, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) -> Tuple[jnp.array, AuxiliaryLossData]:
        #keys = jax.random.split(key, num=data.positions.shape[0])
        e_l, e_l_mat = batch_local_energy(params, key, data)
        jax.debug.print("e_l:{}", e_l)
        loss = constants.pmean(jnp.mean(e_l))
        loss_diff = e_l - loss
        variance = constants.pmean(jnp.mean(loss_diff*jnp.conj(loss_diff)))
        jax.debug.print("loss:{}", loss)
        jax.debug.print("variance:{}", variance)
        return loss, AuxiliaryLossData(variance=variance.real, local_energy=e_l, grad_local_energy=None, local_energy_mat=None)

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
        psi_primal, psi_tangent = jax.jvp(batch_signed_network_log, primals, tangents)
        term1 = (jnp.sum(aux_data.local_energy*jnp.conjugate(psi_tangent)) +
                 jnp.sum(jnp.conjugate(aux_data.local_energy)*psi_tangent))
        term2 = jnp.sum(aux_data.local_energy * psi_tangent.real)
        kfac_jax.register_normal_predictive_distribution(psi_primal.real[:, None])
        primals_out = loss.real, aux_data
        #tangents_out = jnp.sum(psi_tangent).real, aux_data
        device_batch_size = jnp.shape(aux_data.local_energy)[0]
        tangents_out = ((term1 - 2 * term2).real / device_batch_size, aux_data)
        return primals_out, tangents_out

    return total_energy



'''
signed_network, data, batch_params, log_network = main_kfac.main()
key = jax.random.PRNGKey(1)
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
localenergy = hamiltonian.local_energy(f=signed_network)
total_energy_test = make_loss(log_network, local_energy=localenergy)
total_energy_test_pmap = jax.pmap(total_energy_test, in_axes=(0, 0, nn.AINetData(positions=0, atoms=0, charges=0),), out_axes=(0, 0))
loss, aux_data = total_energy_test_pmap(batch_params, subkeys, data)
jax.debug.print("aux_data:{}", aux_data)
val_and_grad = jax.pmap(jax.value_and_grad(total_energy_test, argnums=0, has_aux=True))
output1, output2 = val_and_grad(batch_params, subkeys, data)
'''