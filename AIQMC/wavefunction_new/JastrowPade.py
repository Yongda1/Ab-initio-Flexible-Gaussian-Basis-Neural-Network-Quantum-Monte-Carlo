"""we implement pade functions as Jastrow factors."""
import enum
from typing import Any, Callable, Iterable, Mapping, Union, Tuple
import jax.numpy as jnp
import jax


ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
    """Available Jastrow factors."""
    Pade = enum.auto()


def r_ees_parallel_spins(parallel_indices_inner: jnp.array, rees_inner: jnp.array):
    return rees_inner[parallel_indices_inner[0]][parallel_indices_inner[1]]

"""we should try to avoid this line. 11.5.2025."""
r_ees_parallel_spins_parallel = jax.vmap(r_ees_parallel_spins, in_axes=(1, None), out_axes=0)


def _jastrow_ee(r_ees: jnp.array, params: ParamTree, parallel_indices: jnp.array, antiparallel_indices: jnp.array,
                jastrow_fun: Callable[[jnp.array, float, jnp.array], jnp.array]) -> jnp.array:
    """create the electron-electron jastrow factors.
        ee: ee vectors.
        params: parameters of the neural network.
        nelectrons: the number of electron.
        spins: spin configuration.
        now the problem is how to make the system identify the spin configurations automatically?
    """
    #jax.debug.print("ee:{}", ee)
    #r_ees = jnp.linalg.norm(ee, axis=-1)
    jax.debug.print("r_ees:{}", r_ees)
    r_ees_parallel = r_ees_parallel_spins_parallel(parallel_indices, r_ees)
    jax.debug.print("r_ees_parallel:{}", r_ees_parallel)
    r_ees_antiparallel = r_ees_parallel_spins_parallel(antiparallel_indices, r_ees)
    jax.debug.print("r_ees_antiparallel:{}", r_ees_antiparallel)
    jax.debug.print("params['ee_par']:{}", params['ee_par'])
    jax.debug.print("params['eePanti']:{}", params['ee_anti'])
    jastrow_ee_par = jnp.sum(jax.vmap(jastrow_fun, in_axes=(0, None, 0))(r_ees_parallel, 0.25, params['ee_par']))
    jastrow_ee_anti = jnp.sum(jax.vmap(jastrow_fun, in_axes=(0, None, 0))(r_ees_antiparallel, 0.5, params['ee_anti']))
    return jastrow_ee_anti + jastrow_ee_par


def make_pade_ee_jastrow() -> ...:
    """Create a simple Pade Jastrow factor for electron-electron cusps.
    This cusp means 0.25 or 0.5.
    The number of variational parameter is just one. Later, we could think how to add radius dependence into Jastrow parameters.
    think more !!! 06.08.2024
    something is wrong. I dont unerstand. 8.4.2025"""

    def pade_ee_cusp_fun(r_ee: jnp.array, cusp: float, alpha: jnp.array) -> jnp.array:
        #jax.debug.print("r_ee:{}", r_ee)
        #jax.debug.print("alpha:{}", alpha)
        return (r_ee * cusp) / (1.0 + alpha * r_ee)

    def init(n_parallel: int, n_antiparallel: int) -> Mapping[str, jnp.array]:
        params = {}
        params['ee_par'] = jnp.ones(shape=n_parallel)
        params['ee_anti'] = jnp.ones(shape=n_antiparallel)
        return params

    def apply(r_ee: jnp.array, params: ParamTree, parallel_indices: jnp.array, antiparallel_indices: jnp.array,) -> jnp.array:
        return _jastrow_ee(r_ee, params, parallel_indices, antiparallel_indices, jastrow_fun=pade_ee_cusp_fun)

    return init, apply



'''
init, apply = make_pade_ee_jastrow()
params1 = init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
Jastrow_ee = apply(ee, params=params1, parallel_indices=parallel_indices, antiparallel_indices=antiparallel_indices)
'''


def _jastrow_ae(r_ae: jnp.ndarray, params: ParamTree,
                jastrow_fun: Callable[[jnp.array, jnp.array], jnp.array]) -> jnp.array:
    """we also need the format of r_ae and charges to do the summation. To be continued."""
    jastrow_ae = jnp.sum(jastrow_fun(r_ae, params['ae']))
    return jastrow_ae


def make_pade_ae_jastrow(charges: jnp.array) -> ...:
    """create a Jastrow factor for atom-electron cusp condition. Currently, we only consider homogeneous term."""

    def pade_ae_cusp_fun(r_ae: jnp.array, beta: jnp.array, ) -> jnp.array:
        #jax.debug.print("r_ae:{}", r_ae)
        #jax.debug.print("beta:{}", beta)

        def multiply(r_ae_inner: jnp.array, beta_inner: jnp.array, charges: jnp.array):
            return -1 * jnp.float_power(2.0 * charges, 3.0/4.0) * (1.0 - jnp.exp(-1.0 * jnp.float_power((2.0 * charges), 1.0 / 4.0) * r_ae_inner * beta_inner)) / (2.0 * beta_inner)

        multiply_parallel = jax.vmap(multiply, in_axes=(0, 0, None))
        jastrow_ae_value = multiply_parallel(r_ae, beta, charges)
        return jnp.sum(jastrow_ae_value)

    def init(nelectrons: int, natoms: int) -> Mapping[str, jnp.array]:
        params = {}
        params['ae'] = jnp.ones(shape=(nelectrons, natoms))
        return params

    def apply(r_ae: jnp.ndarray, params: ParamTree) -> jnp.array:
        return _jastrow_ae(r_ae, params, jastrow_fun=pade_ae_cusp_fun)

    return init, apply

'''
#atom_indices = jnp.array([0, 1, 1, 2, 2, 2])
#charges_needed = jnp.array([1, 2, 2, 3, 3, 3])
#init, apply = make_pade_ae_jastrow(atom_indices=atom_indices, charges_needed=charges_needed)
charges_jastrow = np.array([1, 2, 3])
charges_indices_jastrow = np.arange(3)
atom_jastrow_indices = []
charged_jastrow_needed = []
for i in range(len(charges_indices_jastrow)):
    temp = np.repeat(charges_indices_jastrow[i], charges_jastrow[i])
    temp1 = np.repeat(charges_jastrow[i], charges_jastrow[i])
    jax.debug.print("temp:{}", temp)
    atom_jastrow_indices.append(temp)
    charged_jastrow_needed.append(temp1)

atom_jastrow_indices = np.hstack(np.array(atom_jastrow_indices))
charged_jastrow_needed = np.hstack(np.array(charged_jastrow_needed))
jax.debug.print("atom_indices:{}", atom_jastrow_indices)
jax.debug.print("charged_needed:{}", charged_jastrow_needed)
#charges_needed_1 = jnp.repeat(charges_indices, charges, axis=-1)
#jax.debug.print("charges_needed_1:{}", charges_needed_1)
#params2 = init(nelectron=6)
#print('params2', params2)
#Jastrow_ae = apply(ae, params=params2)
#jax.debug.print("Jastrow_ae:{}", Jastrow_ae)
'''

def get_jastrow(charges: jnp.array) -> ...:
    jastrow_ee_init, jastrow_ee_apply = make_pade_ee_jastrow()
    jastrow_ae_init, jastrow_ae_apply = make_pade_ae_jastrow(charges)
    return jastrow_ee_init, jastrow_ee_apply, jastrow_ae_init, jastrow_ae_apply