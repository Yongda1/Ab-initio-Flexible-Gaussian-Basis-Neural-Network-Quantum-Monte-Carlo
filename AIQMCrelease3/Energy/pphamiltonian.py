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

"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union
import chex
from AIQMCrelease3.wavefunction_Ynlm import nn
from AIQMCrelease3.utils import utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol
from AIQMCrelease3.pseudopotential import pp_energy_test
from AIQMCrelease3.pseudopotential import pseudopotential

Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):

    def __call__(
            self,
            params: nn.ParamTree,
            key: chex.PRNGKey,
            data: nn.AINetData,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

    def __call__(
            self,
            f: nn.AINetLike,
            lognetwork: nn.LogAINetLike,
            charges: jnp.ndarray,
            nspins: Sequence[int],
            use_scan: bool = False,
            complex_output: bool = False,
            **kwargs: Any
    ) -> LocalEnergy:
        """Builds the LocalEnergy function."""


KineticEnergy = Callable[[nn.ParamTree, nn.AINetData], jnp.ndarray]


def local_kinetic_energy(
        f: nn.AINetLike,
        use_scan: bool = False,
        complex_output: bool = True,) -> KineticEnergy:
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    def _lapl_over_f(params, data):
        global phase_primal
        n = data.positions.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.grad(logabs_f, argnums=1)

        def grad_f_closure(x):
            return grad_f(params, x, data.spins, data.atoms, data.charges)

        primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)
        if complex_output:
            grad_phase = jax.grad(phase_f, argnums=1)

            def grad_phase_closure(x):
                return grad_phase(params, x, data.spins, data.atoms, data.charges)

            phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, data.positions)
            hessian_diagonal = (lambda i: dgrad_f(eye[i])[i] + 1.j * dgrad_phase(eye[i])[i])
        else:
            hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

        if use_scan:
            _, diagonal = lax.scan(
                lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
            result = -0.5 * jnp.sum(diagonal)
        else:
            result = -0.5 * lax.fori_loop(0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
        result -= 0.5 * jnp.sum(primal ** 2)
        if complex_output:
            result += 0.5 * jnp.sum(phase_primal ** 2)
            result -= 1.j * jnp.sum(primal * phase_primal)
        return result
    return _lapl_over_f


def potential_electron_electron(r_ee: Array) -> jnp.ndarray:
    r_ee = r_ee[jnp.triu_indices_from(r_ee[..., 0], 1)]
    return (1.0 / r_ee).sum()


def potential_electron_nuclear(charges: Array, r_ae: Array) -> jnp.ndarray:
    return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(charges: Array, atoms: Array) -> jnp.ndarray:
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    return jnp.sum(
        jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def potential_energy(r_ee: Array, atoms: Array,
                     charges: Array) -> jnp.ndarray:
    return (potential_electron_electron(r_ee) +
            potential_nuclear_nuclear(charges, atoms))


def local_energy(
        f: nn.AINetLike,
        lognetwork,
        charges: jnp.array,
        nspins: Sequence[int],
        rn_local: jnp.array,
        local_coes: jnp.array,
        local_exps: jnp.array,
        rn_non_local: jnp.array,
        non_local_coes: jnp.array,
        non_local_exps: jnp.array,
        natoms: int,
        nelectrons: int,
        ndim: int,
        list_l: int,
        use_scan: bool = False,
        complex_output: bool = False) -> LocalEnergy:
    del nspins
    """To be continued 22.1.2025."""
    ke = local_kinetic_energy(f,
                              use_scan=use_scan,
                              complex_output=complex_output,)

    get_local_part_energy_test = pseudopotential.local_pp_energy(nelectrons=nelectrons,
                                                                 natoms=natoms,
                                                                 ndim=ndim,
                                                                 rn_local=rn_local,
                                                                 local_coefficient=local_coes,
                                                                 local_exponent=local_exps)

    get_non_local_coe_test = pseudopotential.get_non_v_l(ndim=ndim,
                                                         nelectrons=nelectrons,
                                                         natoms=natoms,
                                                         rn_non_local=rn_non_local,
                                                         non_local_coefficient=non_local_coes,
                                                         non_local_exponent=non_local_exps)

    generate_points_information_test = pseudopotential.get_P_l(nelectrons=nelectrons,
                                                               natoms=natoms,
                                                               ndim=ndim,
                                                               log_network_inner=lognetwork)
    total_energy_function_test = pp_energy_test.total_energy_pseudopotential(
        get_local_pp_energy=get_local_part_energy_test,
        get_nonlocal_pp_coes=get_non_local_coe_test,
        get_P_l=generate_points_information_test,
        list_l=list_l)

    def _e_l(params: nn.ParamTree,
             key: chex.PRNGKey,
             data: nn.AINetData) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:

        ae, ee, r_ae, r_ee = nn.construct_input_features(data.positions, data.atoms)
        potential = (potential_energy(r_ee, data.atoms, charges))
        kinetic = ke(params, data)
        pp_energy_value = total_energy_function_test(params, key, data)
        total_energy = pp_energy_value + kinetic + potential
        energy_mat = None
        return total_energy, energy_mat

    return _e_l
