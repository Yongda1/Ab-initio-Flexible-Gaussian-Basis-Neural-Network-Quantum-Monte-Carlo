"""Evaluates the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union
import chex
from AIQMC import nn
from AIQMC import pseudopotential as pp
from AIQMC.utils import utils
#import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol
from AIQMC import main


signednetwork, data, batchparams, batchphase, batchnetwork = main.main()
print("data.positions", data.positions)
#print("params", batchparams)
key = jax.random.PRNGKey(seed=1)


Array = Union[jnp.ndarray, np.ndarray]

class LocalEnergy(Protocol):
    def __call__(self, params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) \
            -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Returns the local energy of a Hamiltonian at a configuration."""


class MakeLocalEnergy(Protocol):
    def __call__(self, f: nn.AINetLike, charges: jnp.ndarray, nspins: Sequence[int], use_scan: bool = False,
                 complex_output: bool = False, **kwargs: Any) -> LocalEnergy:
        """Builds the LocalEnergy function."""


KineticEnergy = Callable[[nn.ParamTree, nn.AINetData], jnp.ndarray]


def local_kinetic_energy(f: nn.AINetLike, use_scan: bool = False, complex_output: bool = False, laplacian_method: str = 'default') -> KineticEnergy:
    """Create the function for the local kinetic energy, -1/2 \nabla^2 ln|f|.
    29.08.2024 here our codes will be completely different from other codes due to the introduction of angular functions.
    I need take some notes on my slides.
    29.08.2024 I dont understand angular functions, complex number? how to calculate kinetic energy by real number? Why is it a real number?"""
    #phase_f = utils.select_output(f, 0)
    #jax.debug.print("data.positions:{}", data.positions)
    logabs_f = utils.select_output(f, 1)
    angle_f = utils.select_output(f, 2)
    def _lapl_over_f(params, data):
        """29.08.2024 take care of the following function, we need write argnums=1 two times."""
        second_grad_value = jax.vmap(jax.jacfwd(jax.jacrev(logabs_f, argnums=1), argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)
        hessian_value_logabs = second_grad_value(params, data.positions, data.atoms, data.charges)
        #jax.debug.print("hessian_value_logabs:{}", hessian_value_logabs)
        angle_grad_hessian = jax.vmap(jax.jacfwd(jax.jacrev(angle_f, argnums=1), argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)
        hessian_value_angle_f = angle_grad_hessian(params, data.positions, data.atoms, data.charges)
        print("-------------------------------")
        #jax.debug.print("hessian_value_logabs:{}", hessian_value_logabs)
        print("-------------------------------")
        #ax.debug.print("hessian_value_angle:{}", hessian_value_angle_f)
        hessian_value_all = hessian_value_logabs + 1.j*hessian_value_angle_f
        #jax.debug.print("hessian_value_diagonal:{}", hessian_value_all)
        """here, notice the shape of the array hessian_value_all"""
        hessian_value_all = jnp.reshape(hessian_value_all, (4, 12, 12))
        print("-------------------------------")
        #jax.debug.print("hessian_value_all:{}", hessian_value_all)
        """here, notice the shape of the array hessian_value_all, also the axis"""
        hessian_diagonal = jnp.diagonal(hessian_value_all, axis1=1, axis2=2)
        #jax.debug.print("hessian_diagonal:{}", hessian_diagonal)
        angle_grad_value = jax.vmap(jax.grad(angle_f, argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)
        grad_value = jax.vmap(jax.grad(logabs_f, argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)
        first_derivative_angle = angle_grad_value(params, data.positions, data.atoms, data.charges)
        first_derivative_f = grad_value(params, data.positions, data.atoms, data.charges)
        #jax.debug.print("first_derivative_angle:{}", first_derivative_angle)
        #jax.debug.print("first_derivative_f:{}", first_derivative_f)
        first_derivative_f = jnp.reshape(first_derivative_f, (4, 12)) # 4 is batch size. 12 is the number of elements in data.positions.
        first_derivative_angle = jnp.reshape(first_derivative_angle, (4, 12))
        third_term = jnp.sum(1.j * 2 * first_derivative_angle * first_derivative_f, axis=1)
        jax.debug.print("third_term:{}", third_term)
        #jax.debug.print("first_derivative_angle:{}", first_derivative_angle)
        #jax.debug.print("first_derivative_f:{}", first_derivative_f)
        second_term = jnp.sum(jnp.square(first_derivative_f), axis=1) - jnp.sum(jnp.square(first_derivative_angle), axis=1)
        jax.debug.print("second_term:{}", second_term)
        first_term = jnp.sum(hessian_diagonal, axis=1)
        jax.debug.print("first_term:{}", first_term)
        #jax.debug.print("first_term:{}", jnp.shape(first_term))
        kinetic_energy = first_term + second_term + third_term
        return kinetic_energy
    return _lapl_over_f


#lap_over_f = local_kinetic_energy(signednetwork)
#output = lap_over_f(batchparams, data)


"""all electron interaction calculation."""
def potential_electron_electron(r_ee: jnp.array, batch_size: int) -> jnp.array:
    """electron-electron interaction energy.
    Good news, today we finished kinetic energy calculation and electron-electron interaction energy 02.09.2024.
    Tomorrow, we are going to finish the all-electrons potential energy calculation module."""
    #jax.debug.print("r_ee[0]:{}", r_ee[0])
    indices = jnp.nonzero(jnp.triu(r_ee))
    #jax.debug.print("indices:{}", indices)
    r_ee = r_ee[indices]
    r_ee = jnp.reshape(r_ee, (batch_size, -1))
    #jax.debug.print("r_ee:{}", r_ee)
    #r_ee = r_ee[jnp.triu_indices_from(r_ee[0], k=1)]
    #jax.debug.print("r_ee_triu:{}", r_ee)
    #jax.debug.print("energy_ee:{}", jnp.sum(1.0 / r_ee, axis=-1))
    #jax.debug.print("energy:{}", jnp.sum(1.0 / r_ee, axis=1))
    return jnp.sum(1.0 / r_ee, axis=1)

def potential_electron_nuclear(ee: jnp.array) -> jnp.array:
    """electron-atom interaction energy"""

def potential_nuclear_nuclear(ee: jnp.array) -> jnp.array:
    """atom-atom interaction energy"""


def local_energy(f: nn.AINetLike, complex_number: bool = True) -> LocalEnergy:
    """create the function to evaluate the local energy.
    default is complex number.
    we leave the interface for pseudopotential calculation to be continued."""
    ke = local_kinetic_energy(f)
    def _e_l(batch_size: int, ndim: int, batchparams: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData):
        jax.debug.print("data.positions:{}", data.positions)
        jax.debug.print("data.atoms:{}", data.atoms)
        electron_pos_temp = jnp.reshape(data.positions, (batch_size, 12))#4 is batch_size, 12 is the size of input parameters.
        atoms_position_temp = jnp.reshape(data.atoms, (batch_size, 2, ndim))#2 is the number of atoms.
        jax.debug.print("electron_pos_temp:{}", electron_pos_temp)
        jax.debug.print("atoms_position_temp:{}", atoms_position_temp)
        """here, we have some problems about the construction of input layers.02.09.2024."""
        #ae, ee = jax.vmap(nn.construct_input_features(electron_pos_temp, atoms_position_temp), in_axes=(1, 1), out_axes=(0, 0))
        ee = jnp.reshape(electron_pos_temp, [batch_size, 1, -1, ndim]) - jnp.reshape(electron_pos_temp, [batch_size, -1, 1, ndim])
        ae = jnp.reshape(electron_pos_temp, [batch_size, -1, 1, ndim]) - atoms_position_temp[batch_size, None, ...]
        print("-----------------------------")
        jax.debug.print("ee:{}", ee)
        print("-----------------------------")
        jax.debug.print("ae:{}", ae)
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ee = jnp.linalg.norm(ee, axis=-1)
        jax.debug.print("r_ae:{}", r_ae)
        jax.debug.print("r_ee:{}", r_ee)
        potential = potential_electron_electron(r_ee, batch_size=batch_size)
        kinetic = ke(batchparams, data)
    return _e_l


_e_l = local_energy(f=signednetwork, complex_number=True)
output = _e_l(batch_size=4, ndim=3, batchparams=batchparams, key=key, data=data)


















