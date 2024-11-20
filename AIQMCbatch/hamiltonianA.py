"""Evaluates the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Sequence, Union

import chex

from AIQMCbatch import nn
# from AIQMC import pseudopotential as pp
from AIQMCbatch.utils import utils
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol
from AIQMCbatch import nn
#from AIQMCbatch import main_kfac

"""the hamilonianA module is only for one configuration. Because this function will be vmapped in loss.py, i.e. loss function.
 So, the input is just like this:.
"""
#pos = [0.04145381, -0.01200894, -0.1026478 , -0.13694875, -0.09510095, -0.01401753,  0.36738175,  0.13366607,
#                 0.24013737,  0.15820067,  0.15000297,  0.06570248]
#pos = jnp.array(pos)
#ee = jnp.reshape(pos, [1, -1, 3]) - jnp.reshape(pos, [-1, 1, 3])

#r_ee = jnp.linalg.norm(ee, axis=-1)
#signed_network, data, batch_params, batch_network, batch_phase_network = main_kfac.main()
#key = jax.random.PRNGKey(seed=1)
#jax.debug.print("data:{}", data)


Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):
    def __call__(self, params: nn.ParamTree, data: nn.AINetData, batch_size: int, ndim: int,) -> jnp.array:
        """Returns the local energy of a Hamiltonian at a configuration."""


class MakeLocalEnergy(Protocol):
    def __call__(self, f: nn.AINetLike, charges: jnp.ndarray, nspins: Sequence[int], use_scan: bool = False,
                 complex_output: bool = False, **kwargs: Any) -> LocalEnergy:
        """Builds the LocalEnergy function."""


KineticEnergy = Callable[[nn.ParamTree, nn.AINetData], jnp.ndarray]


def local_kinetic_energy(f: nn.AINetLike) -> KineticEnergy:
    """Create the function for the local kinetic energy, -1/2 \nabla^2 ln|f|.
    29.08.2024 here our codes will be completely different from other codes due to the introduction of angular functions.
    I need take some notes on my slides.
    29.08.2024 I dont understand angular functions, complex number? how to calculate kinetic energy by real number? Why is it a real number?"""
    logabs_f = utils.select_output(f, 1)
    angle_f = utils.select_output(f, 2)

    def _lapl_over_f(params, pos: jnp.array, atoms: jnp.array, charges: jnp.array):
        """29.08.2024 take care of the following function, we need write argnums=1 two times."""
        """here, we also need use both pmap and vmap.
        2.11.2024, this means the vmap should be done in loss.py.  """
        second_grad_value = jax.jacfwd(jax.jacrev(logabs_f, argnums=1), argnums=1)
        #jax.debug.print("pos:{}", pos)
        hessian_value_logabs = second_grad_value(params, pos, atoms, charges)
        angle_grad_hessian = jax.jacfwd(jax.jacrev(angle_f, argnums=1), argnums=1)
        hessian_value_angle_f = angle_grad_hessian(params, pos, atoms, charges)
        hessian_value_all = hessian_value_logabs + 1.j * hessian_value_angle_f
        #jax.debug.print("hessian_value_all:{}", hessian_value_all)
        #jax.debug.print("shape:{}", jnp.shape(hessian_value_all))
        #hessian_value_all = jnp.reshape(hessian_value_all, (4, 12, 12))
        hessian_diagonal = jnp.diagonal(hessian_value_all)
        #jax.debug.print("hessian_diagonal:{}", hessian_diagonal)
        angle_grad_value = jax.grad(angle_f, argnums=1)
        grad_value = jax.grad(logabs_f, argnums=1)
        first_derivative_angle = angle_grad_value(params, pos, atoms, charges)
        first_derivative_f = grad_value(params, pos, atoms, charges)
        #jax.debug.print("first_derivative_f:{}", first_derivative_f)
        #jax.debug.print("first_derivaative_angle:{}", first_derivative_angle)
        #first_derivative_f = jnp.reshape(first_derivative_f, (4, 12))  # 4 is batch size. 12 is the number of elements in data.positions.
        #first_derivative_angle = jnp.reshape(first_derivative_angle, (4, 12))
        third_term = jnp.sum(1.j * 2 * first_derivative_angle * first_derivative_f, axis=-1)
        #jax.debug.print("third_term:{}", third_term)
        second_term = jnp.sum(jnp.square(first_derivative_f), axis=-1) - jnp.sum(jnp.square(first_derivative_angle), axis=-1)
        #jax.debug.print("second_term:{}", second_term)
        first_term = jnp.sum(hessian_diagonal, axis=-1)
        #jax.debug.print("first_term:{}", first_term)
        kinetic_energy = first_term + second_term + third_term
        return kinetic_energy

    return _lapl_over_f


# lap_over_f = local_kinetic_energy(signednetwork)
# output = lap_over_f(batchparams, data)


"""all electron interaction calculation."""


def potential_electron_electron(r_ee: jnp.array) -> jnp.array:
    """electron-electron interaction energy.
    Good news, today we finished kinetic energy calculation and electron-electron interaction energy 02.09.2024.
    Tomorrow, we are going to finish the all-electrons potential energy calculation module."""
    """the size argument of jnp.nonzero must be statically specified. This must be solved later. 22.10.2024."""
    """be aware of this line, currently, we use one loop to solve the problem. Later, we have to improve it.23.10.2024.
    we rewirte it as single configuration version. 2.11.2024."""
    r_ee = jnp.array(r_ee)
    #jax.debug.print("r_ee:{}", r_ee)
    #jax.debug.print("r_ee:{}", r_ee[..., 0])
    r_ee = r_ee[jnp.triu_indices_from(r_ee, 1)]
    #jax.debug.print("r_ee:{}", r_ee)
    return (1.0 / r_ee).sum()

"""we test potential_electron_electron """
#output = potential_electron_electron(r_ee=r_ee)

def potential_electron_nuclear(r_ae: jnp.array, charges: jnp.array) -> jnp.array:
    """electron-atom interaction energy"""
    charges = jnp.reshape(jnp.repeat(charges, 4, axis=0), (4, 2)) #4 is the batch size. 2 is the number of atoms.
    #jax.debug.print("charges:{}", charges)
    Energy = charges / r_ae
    #jax.debug.print("Energy:{}", Energy)
    return jnp.sum(jnp.sum(Energy, axis=-1), axis=-1)


def potential_nuclear_nuclear(atoms: jnp.array, charges: jnp.array) -> jnp.array:
    """atom-atom interaction energy
    to be continued 3.11.2024."""
    jax.debug.print("atoms:{}", atoms)
    jax.debug.print("charges:{}", charges)
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    jax.debug.print("r_aa:{}", r_aa)
    jax.debug.print("charges:{}", charges[..., None])
    return jnp.sum(jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def local_energy(f: nn.AINetLike) -> LocalEnergy:
    """create the function to evaluate the local energy.
    default is complex number.
    f: signednetwork
    we leave the interface for pseudopotential calculation to be continued.
    28.10.2024, we still keep batch_atoms and batch_charges as the inputs.
    Because the optimizer do not accept the extra input, i.e. batch_atoms and batch_charges. We have to remove them.
    """
    lap_over_f = local_kinetic_energy(f)
    def _e_l(params: nn.ParamTree, key: chex.PRNGKey, pos: jnp.array, atoms: jnp.array, charges: jnp.array) -> jnp.array:
        """after we change the parallel, we also need rewrite this part. we will solve this later, 31.10.2024.
        This is the version for test. Be careful of the input. we need change the last three variabls to data.3.11.2024.
        We leave it to monday."""
        """we already created the correct batch version. We do not need reshape the array, data.atoms and data.charges."""
        ndim = 3
        """here, we have some problems about the construction of input layers.02.09.2024."""
        ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
        ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ee = jnp.linalg.norm(ee, axis=-1)
        jax.debug.print("r_ae:{}", r_ae)
        jax.debug.print("r_ee:{}", r_ee)
        potential_E_ee = potential_electron_electron(r_ee)
        jax.debug.print("charges:{}", charges)
        potential_E_ae = potential_electron_nuclear(r_ae, charges)
        potential_E_aa = potential_nuclear_nuclear(atoms, charges)
        kinetic = lap_over_f(params, pos, atoms, charges)
        total_energy = kinetic + potential_E_ee + potential_E_ae + potential_E_aa
        jax.debug.print("total_energy:{}", total_energy)
        return total_energy

    return _e_l


#localenergy = local_energy(f=signed_network)
#batch_local_energy = jax.pmap(jax.vmap(localenergy, in_axes=(None, 0, 0, 0), out_axes=0))
#output = batch_local_energy(batch_params, data.positions, data.atoms, data.charges)