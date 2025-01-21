
from typing import Tuple
import chex
from AIQMCrelease1 import constants
from AIQMCrelease1.Energy import hamiltonian_wrong
from AIQMCrelease1.wavefunction_debug import nn_wrong
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
    variance: jax.Array
    local_energy: jax.Array
    clipped_energy: jax.Array
    grad_local_energy: jax.Array | None
    local_energy_mat: jax.Array | None


class LossAINet(Protocol):

    def __call__(self, params: nn_wrong.ParamTree, key: chex.PRNGKey, data: nn_wrong.AINetData) \
            -> Tuple[jnp.array, AuxiliaryLossData]:
        """Evaluagtes the total energy of the network for a batch of configurations."""

def clip_local_values(
    local_values: jnp.ndarray,
    mean_local_values: jnp.ndarray,
    clip_scale: float,
    clip_from_median: bool,
    center_at_clipped_value: bool,
    complex_output: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

  batch_mean = lambda values: constants.pmean(jnp.mean(values))

  def clip_at_total_variation(values, center, scale):
    tv = batch_mean(jnp.abs(values- center))
    return jnp.clip(values, center - scale * tv, center + scale * tv)

  if clip_from_median:
    # More natural place to center the clipping, but expensive due to both
    # the median and all_gather (at least on multihost)
    clip_center = jnp.median(constants.all_gather(local_values).real)
  else:
    clip_center = mean_local_values
  # roughly, the total variation of the local energies
  if complex_output:
    clipped_local_values = (
        clip_at_total_variation(
            local_values.real, clip_center.real, clip_scale) +
        1.j * clip_at_total_variation(
            local_values.imag, clip_center.imag, clip_scale)
    )
  else:
    clipped_local_values = clip_at_total_variation(
        local_values, clip_center, clip_scale)
  if center_at_clipped_value:
    diff_center = batch_mean(clipped_local_values)
  else:
    diff_center = mean_local_values
  diff = clipped_local_values - diff_center
  return diff_center, diff


def make_loss(network: nn.AINetLike,
              local_energy: hamiltonian_wrong.LocalEnergy,
              clip_local_energy: float = 0.0,
              clip_from_median: bool = True,
              center_at_clipped_energy: bool = True,
              complex_output: bool = False) -> LossAINet:
    """our local_energy function from hamiltonian module is already batched version.
    We dont need rewrite it here."""
    batch_local_energy = jax.vmap(local_energy, in_axes=(None, None, nn_wrong.AINetData(positions=0, spins=0, atoms=0, charges=0),
                                                         ), out_axes=(0, 0))
    batch_signed_network_log = jax.vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) -> Tuple[jnp.array, AuxiliaryLossData]:
        #keys = jax.random.split(key, num=data.positions.shape[0])
        e_l, e_l_mat = batch_local_energy(params, key, data)
        jax.debug.print("e_l:{}", e_l)
        loss = constants.pmean(jnp.mean(e_l))
        loss_diff = e_l - loss
        variance = constants.pmean(jnp.mean(loss_diff*jnp.conj(loss_diff)))
        jax.debug.print("loss:{}", loss)
        #jax.debug.print("variance:{}", variance)
        return loss, AuxiliaryLossData(variance=variance.real, local_energy=e_l, clipped_energy=e_l, grad_local_energy=None, local_energy_mat=e_l_mat)

    """here, we probably need redefine the differentiation rules for the mean value of total energy."""
    '''
    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, key, data = primals
        loss, aux_data = total_energy(params, key, data)
        data = primals[2]
        data_tangents = tangents[2]
        primals = (primals[0], data.positions, data.spins, data.atoms, data.charges)
        tangents = (
            tangents[0],
            data_tangents.positions,
            data_tangents.spins,
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
    '''

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, key, data = primals
        loss, aux_data = total_energy(params, key, data)
        if clip_local_energy > 0.0:
            aux_data.clipped_energy, diff = clip_local_values(
                aux_data.local_energy,
                loss,
                clip_local_energy,
                clip_from_median,
                center_at_clipped_energy,
                complex_output)
        else:
            diff = aux_data.local_energy - loss
        data = primals[2]
        data_tangents = tangents[2]
        primals = (primals[0], data.positions, data.spins, data.atoms, data.charges)
        tangents = (
            tangents[0],
            data_tangents.positions,
            data_tangents.spins,
            data_tangents.atoms,
            data_tangents.charges,
        )
        psi_primal, psi_tangent = jax.jvp(batch_signed_network_log, primals, tangents)
        clipped_el = diff + aux_data.clipped_energy
        term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) +
                 jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
        term2 = jnp.sum(aux_data.clipped_energy * psi_tangent.real)
        kfac_jax.register_normal_predictive_distribution(psi_primal.real[:, None])
        primals_out = loss.real, aux_data
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