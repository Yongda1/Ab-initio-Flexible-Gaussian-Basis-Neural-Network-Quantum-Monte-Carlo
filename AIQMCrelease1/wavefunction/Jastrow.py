"""we implement pade functions as Jastrow factors."""
import enum
from typing import Any, Callable, Iterable, Mapping, Union, Tuple
import jax.numpy as jnp
import jax
#from nn import construct_input_features

'''
pos = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
atoms = jnp.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
ae, ee = construct_input_features(pos=pos, atoms=atoms)
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
temp = jnp.reshape(spins, (6, 1))
spins = jnp.reshape(spins, (1, 6))
spins_total = spins * temp
spins_total_uptriangle = jnp.triu(spins_total, k=1)
sample = jnp.zeros_like(a=spins_total_uptriangle)
parallel = jnp.where(spins_total_uptriangle > sample, spins_total_uptriangle, sample)
antiparallel = jnp.where(spins_total_uptriangle < sample, spins_total_uptriangle, sample)
parallel_indices = jnp.nonzero(parallel)
antiparallel_indices = jnp.nonzero(antiparallel)
parallel_indices = jnp.array(parallel_indices)
antiparallel_indices = jnp.array(antiparallel_indices)
n_parallel = len(parallel_indices[0])
n_antiparallel = len(antiparallel_indices[0])
'''


ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
    """Available Jastrow factors."""
    Pade = enum.auto()


def _jastrow_ee(ee: jnp.ndarray, params: ParamTree, parallel_indices: jnp.array, antiparallel_indices: jnp.array,
                jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """create the electron-electron jastrow factors.
        ee: ee vectors.
        params: parameters of the neural network.
        nelectrons: the number of electron.
        spins: spin configuration.
        now the problem is how to make the system identify the spin configurations automatically?
    """
    r_ees = jnp.linalg.norm(ee, axis=-1)

    def r_ees_parallel_spins(parallel_indices: jnp.array, rees: jnp.array):
        return rees[parallel_indices[0]][parallel_indices[1]]

    r_ees_parallel_spins_parallel = jax.vmap(r_ees_parallel_spins, in_axes=(1, None), out_axes=0)
    r_ees_parallel = r_ees_parallel_spins_parallel(parallel_indices, r_ees)
    r_ees_antiparallel = r_ees_parallel_spins_parallel(antiparallel_indices, r_ees)
    jastrow_ee_par = jnp.sum(jastrow_fun(r_ees_parallel, 0.25, params['ee_par']))
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees_antiparallel, 0.5, params['ee_anti']))
    return jastrow_ee_anti + jastrow_ee_par


def make_pade_ee_jastrow() -> ...:
    """Create a simple Pade Jastrow factor for electron-electron cusps.
    This cusp means 0.25 or 0.5.
    The number of variational parameter is just one. Later, we could think how to add radius dependence into Jastrow parameters.
    think more !!! 06.08.2024"""

    def pade_ee_cusp_fun(r_ee: jnp.ndarray, cusp: float, alpha: jnp.ndarray) -> jnp.ndarray:
        return r_ee / cusp * (1.0 / (jnp.abs(1.0 + alpha * r_ee)))

    def init(n_parallel: int, n_antiparallel: int) -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ee_par'] = jnp.ones(shape=n_parallel)
        params['ee_anti'] = jnp.ones(shape=n_antiparallel)
        return params

    def apply(ee: jnp.ndarray, params: ParamTree, parallel_indices: jnp.array, antiparallel_indices: jnp.array,) -> jnp.ndarray:
        return _jastrow_ee(ee, params, parallel_indices, antiparallel_indices, jastrow_fun=pade_ee_cusp_fun)

    return init, apply



'''
init, apply = make_pade_ee_jastrow()
params1 = init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
Jastrow_ee = apply(ee, params=params1, parallel_indices=parallel_indices, antiparallel_indices=antiparallel_indices)
'''

def _jastrow_ae(ae: jnp.ndarray, nelectron: int, natoms: int, charges: jnp.array, params: ParamTree,
                jastrow_fun: Callable[[jnp.array, int, jnp.array, jnp.array], jnp.array]) -> jnp.array:
    """we also need the format of r_ae and charges to do the summation. To be continued."""
    jastrow_ae = jnp.sum(jastrow_fun(ae, nelectron, natoms, charges, params['ae']))
    return jastrow_ae


def make_pade_ae_jastrow() -> ...:
    """create a Jastrow factor for atom-electron cusp condition. Currently, we only consider homogeneous term.
    In this case, we have two atoms. Each atom has two electrons."""

    def pade_ae_cusp_fun(ae: jnp.ndarray, nelectron: int, natoms: int, charges: jnp.array, beta: jnp.array) -> jnp.array:
        r_ae = jnp.linalg.norm(ae, axis=-1)
        charges = jnp.reshape(jnp.repeat(charges, nelectron), (-1, natoms))
        beta = jnp.reshape(beta, (nelectron, natoms))
        return -1.0 * jnp.float_power((2.0 * charges), (3.0 / 4.0)) * (
                    1.0 - jnp.exp(-1.0 * jnp.float_power((2.0 * charges), 1.0 / 4.0) * r_ae * beta)) / (2.0 * beta)

    def init(nelectron: int, charges: jnp.ndarray) -> Mapping[str, jnp.array]:
        params = {}
        params['ae'] = jnp.ones(shape=nelectron * len(charges))
        return params

    def apply(ae: jnp.ndarray, nelectron: int, natoms: int, charges: jnp.ndarray, params: ParamTree) -> jnp.array:
        return _jastrow_ae(ae, nelectron, natoms, charges, params, jastrow_fun=pade_ae_cusp_fun)

    return init, apply


'''
init, apply = make_pade_ae_jastrow()
params2 = init(nelectron=4, charges=charges)
print('params2', params2)
Jastrow_ae = apply(ae, nelectron=4, charges=charges, params=params2)
'''


def get_jastrow(jastrow: JastrowType) -> ...:
    jastrow_ee_init, jastrow_ee_apply = make_pade_ee_jastrow()
    jastrow_ae_init, jastrow_ae_apply = make_pade_ae_jastrow()
    return jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply
