# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions to create the loss and custom gradient of the loss."""

from typing import Tuple
import chex
from AIQMCrelease3 import constants
from AIQMCrelease3.Energy import pphamiltonian
from AIQMCrelease3.wavefunction_Ynlm import nn
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
    """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
    clipped_energy: local energy after clipping has been applied
    grad_local_energy: gradient of the local energy.
    local_energy_mat: for excited states, the local energy matrix.
  """
    variance: jax.Array
    local_energy: jax.Array
    clipped_energy: jax.Array
    grad_local_energy: jax.Array | None
    local_energy_mat: jax.Array | None


class LossAINet(Protocol):

    def __call__(
            self,
            params: nn.ParamTree,
            key: chex.PRNGKey,
            data: nn.AINetData,
    ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched data elements to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """


def clip_local_values(
        local_values: jnp.ndarray,
        mean_local_values: jnp.ndarray,
        clip_scale: float,
        clip_from_median: bool,
        center_at_clipped_value: bool,
        complex_output: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Clips local operator estimates to remove outliers.

  Args:
    local_values: batch of local values,  Of/f, where f is the wavefunction and
      O is the operator of interest.
    mean_local_values: mean (over the global batch) of the local values.
    clip_scale: clip local quantities that are outside nD of the estimate of the
      expectation value of the operator, where n is this value and D the mean
      absolute deviation of the local quantities from the estimate of w, to the
      boundaries. The clipped local quantities should only be used to evaluate
      gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate/robust to outliers.
    center_at_clipped_value: If true, center the local energy differences passed
      back to the gradient around the clipped quantities, so the mean difference
      across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Tuple of the central value (estimate of the expectation value of the
    operator) and deviations from the central value for each element in the
    batch. If per_device_threshold is True, then the central value is per
    device.
  """

    batch_mean = lambda values: constants.pmean(jnp.mean(values))

    def clip_at_total_variation(values, center, scale):
        tv = batch_mean(jnp.abs(values - center))
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


def make_loss(network,
              local_energy: pphamiltonian.LocalEnergy,
              clip_local_energy: float = 0.0,
              clip_from_median: bool = True,
              center_at_clipped_energy: bool = True,
              complex_output: bool = True) -> LossAINet:
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
    batch_network = jax.vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(
            params: nn.ParamTree,
            key: chex.PRNGKey,
            data: nn.AINetData,
    ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
        """Evaluates the total energy of the network for a batch of configurations."""
        keys = jax.random.split(key, num=data.positions.shape[0])
        e_l, e_l_mat = batch_local_energy(params, keys, data)
        loss = constants.pmean(jnp.mean(e_l))
        loss_diff = e_l - loss
        variance = constants.pmean(jnp.mean(loss_diff * jnp.conj(loss_diff)))
        #jax.debug.print("loss:{}", loss)
        return loss, AuxiliaryLossData(
            variance=variance.real,
            local_energy=e_l,
            clipped_energy=e_l,
            grad_local_energy=None,
            local_energy_mat=e_l_mat,
        )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
        """Custom Jacobian-vector product for unbiased local energy gradients."""
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
        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)

        if complex_output:
            clipped_el = diff + aux_data.clipped_energy
            term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) +
                     jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
            term2 = jnp.sum(aux_data.clipped_energy * psi_tangent.real)
            kfac_jax.register_normal_predictive_distribution(
                psi_primal.real[:, None])
            primals_out = loss.real, aux_data
            device_batch_size = jnp.shape(aux_data.local_energy)[0]
            tangents_out = ((term1 - 2 * term2).real / device_batch_size, aux_data)
        else:
            kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
            primals_out = loss, aux_data
            device_batch_size = jnp.shape(aux_data.local_energy)[0]
            tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
        return primals_out, tangents_out

    return total_energy