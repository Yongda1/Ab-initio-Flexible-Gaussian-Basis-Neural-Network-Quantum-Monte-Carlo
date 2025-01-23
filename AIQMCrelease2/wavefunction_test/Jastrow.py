"""we implement pade functions as Jastrow factors."""
import enum
from typing import Any, Callable, Iterable, Mapping, Union, Tuple
import jax.numpy as jnp
import jax
#from nn import construct_input_features
#import numpy as np

'''
pos = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
atoms = jnp.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
ae, ee, r_ae, r_ee = construct_input_features(pos=pos, atoms=atoms)

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


def r_ees_parallel_spins(parallel_indices_inner: jnp.array, rees_inner: jnp.array):
    return rees_inner[parallel_indices_inner[0]][parallel_indices_inner[1]]

r_ees_parallel_spins_parallel = jax.vmap(r_ees_parallel_spins, in_axes=(1, None), out_axes=0)


def _jastrow_ee(ee: jnp.ndarray, params: ParamTree, parallel_indices: jnp.array, antiparallel_indices: jnp.array,
                jastrow_fun: Callable[[jnp.array, float, jnp.array], jnp.array]) -> jnp.array:
    """create the electron-electron jastrow factors.
        ee: ee vectors.
        params: parameters of the neural network.
        nelectrons: the number of electron.
        spins: spin configuration.
        now the problem is how to make the system identify the spin configurations automatically?
    """
    r_ees = jnp.linalg.norm(ee, axis=-1)
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

    def pade_ee_cusp_fun(r_ee: jnp.array, cusp: float, alpha: jnp.array) -> jnp.array:
        return (r_ee / cusp) * (1.0 / (1.0 + alpha * r_ee))

    def init(n_parallel: int, n_antiparallel: int) -> Mapping[str, jnp.array]:
        params = {}
        params['ee_par'] = jnp.ones(shape=n_parallel)
        params['ee_anti'] = jnp.ones(shape=n_antiparallel)
        return params

    def apply(ee: jnp.array, params: ParamTree, parallel_indices: jnp.array, antiparallel_indices: jnp.array,) -> jnp.array:
        return _jastrow_ee(ee, params, parallel_indices, antiparallel_indices, jastrow_fun=pade_ee_cusp_fun)

    return init, apply



'''
init, apply = make_pade_ee_jastrow()
params1 = init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
Jastrow_ee = apply(ee, params=params1, parallel_indices=parallel_indices, antiparallel_indices=antiparallel_indices)
'''


def _jastrow_ae(ae: jnp.ndarray, params: ParamTree,
                jastrow_fun: Callable[[jnp.array, ], jnp.array]) -> jnp.array:
    """we also need the format of r_ae and charges to do the summation. To be continued."""
    jastrow_ae = jnp.sum(jastrow_fun(ae, params['ae']))
    return jastrow_ae


def make_pade_ae_jastrow(atom_indices: jnp.array, charges_needed: jnp.array) -> ...:
    """create a Jastrow factor for atom-electron cusp condition. Currently, we only consider homogeneous term.
    In this case, we test two H atoms, each one has one electron.
    here, we have one problem. The one-body jastrow is defined as the interaction between the electron between the atom.
    It is not many-electrons interaction.
    In this way, we need tell the system how many electrons each atom has
    to be continued...16.1.2025.
    """

    def pade_ae_cusp_fun(ae: jnp.array, beta: jnp.array, ) -> jnp.array:
        r_ae = jnp.linalg.norm(ae, axis=-1)

        def extract_indices(r_ae_inner: jnp.array, atom_indices: jnp.array):
            r_needed = r_ae_inner[atom_indices]
            return r_needed

        extract_indices_parallel = jax.vmap(extract_indices)
        r_needed = extract_indices_parallel(r_ae_inner=r_ae, atom_indices=atom_indices)
        return -1.0 * jnp.float_power((2.0 * charges_needed), (3.0 / 4.0)) \
               * (1.0 - jnp.exp(-1.0 * jnp.float_power((2.0 * charges_needed), 1.0 / 4.0) * r_needed * beta)) / (2.0 * beta)

    def init(nelectron: int) -> Mapping[str, jnp.array]:
        params = {}
        params['ae'] = jnp.ones(shape=nelectron)
        return params

    def apply(ae: jnp.ndarray, params: ParamTree) -> jnp.array:
        return _jastrow_ae(ae, params, jastrow_fun=pade_ae_cusp_fun)

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

def get_jastrow() -> ...:
    jastrow_ee_init, jastrow_ee_apply = make_pade_ee_jastrow()
    #jastrow_ae_init, jastrow_ae_apply = make_pade_ae_jastrow()
    return jastrow_ee_init, jastrow_ee_apply
