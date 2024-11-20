"""we implement pade functions as Jastrow factors."""
import enum
from typing import Any, Callable, Iterable, Mapping, Union, Tuple
import jax.numpy as jnp
#import jax
# from nn import construct_input_features

'''
def construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Construct inputs to AINet from raw electron and atomic positions.
    Here, we assume that the electron spin is up and down along the axis=0 in array pos.
    So, the pairwise distance ae also follows this order.
        pos: electron positions, Shape(nelectrons * dim)
        atoms: atom positions. Shape(natoms, ndim)
    """
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    return ae, ee


pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0], [0.2, 0.2, 0.2]])
ae, ee = construct_input_features(pos, atoms, ndim=3)
charges = jnp.array([2, 2])'''

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
    """Available Jastrow factors."""
    Pade = enum.auto()


def _jastrow_ee(ee: jnp.ndarray, params: ParamTree, nelectron: int,
                jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """we need develope the method to spit the spin configurations. 01.08.2024.
    We already found the method for spliting the spin configurations. 07.08.2024.
    we need change the way of spliting the spin configurations.
    Because we mentioned that the spin configurations should be input by hand.[[1, 0],[1, 0]] 22.08.2024."""
    #print("start Jastrow part----------------")
    # print('ee', ee)
    # jax.debug.print("vmap_ee:{}", ee)
    r_ees = jnp.linalg.norm(ee, axis=-1)
    # jax.debug.print("r_ees:{}", r_ees)
    n_spin_up = int(nelectron / 2)
    indices_r_ee_parallel = jnp.arange(start=1, stop=nelectron, step=2)
    indices_r_ee_anti_parallel = jnp.arange(start=2, stop=nelectron, step=2)
    # jax.debug.print("indices_r_ee_spin_parallel:{}", indices_r_ee_anti_parallel)
    iu = jnp.triu(r_ees)
    # jax.debug.print("iu:{}", iu)
    r_ees_parallel = jnp.concatenate([jnp.diagonal(iu, offset=2)])
    r_ees_anti_parallel = jnp.concatenate([jnp.diagonal(iu, offset=1), jnp.diagonal(iu, offset=3)])
    # jax.debug.print("r_ees_parallel:{}", r_ees_parallel)
    # jax.debug.print("r_ees_anti_paralle:{}", r_ees_anti_parallel)
    # print('spin_up', iu[:, :n_spin_up])
    # r_ees_parallel_up_up = jnp.ravel(iu[:, :n_spin_up])[jnp.nonzero(jnp.ravel(iu[:, :n_spin_up]))]
    # r_ees_parallel_down_down = jnp.ravel(iu[-n_spin_up:, :])[jnp.nonzero(iu[-n_spin_up, :])]
    # r_ees_parallel = jnp.concatenate([r_ees_parallel_up_up, r_ees_parallel_down_down])
    # r_ees_anti_parallel = jnp.ravel(iu[:n_spin_up, -n_spin_up:])[jnp.nonzero(jnp.ravel(iu[:n_spin_up, -n_spin_up:]))]
    jastrow_ee_par = jnp.sum(jastrow_fun(r_ees_parallel, 0.25, params['ee_par']))
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees_anti_parallel, 0.5, params['ee_anti']))
    #jax.debug.print('jastrow_ee_par:{}', jastrow_ee_par)
    #jax.debug.print('jastrow_ee_anti:{}', jastrow_ee_anti)

    return jastrow_ee_anti + jastrow_ee_par


def make_pade_ee_jastrow() -> ...:
    """Create a simple Pade Jastrow factor for electron-electron cusps.
    This cusp means 0.25 or 0.5.
    The number of variational parameter is just one. Later, we could think how to add radius dependence into Jastrow parameters.
    think more !!! 06.08.2024"""

    def pade_ee_cusp_fun(r_ee: jnp.ndarray, cusp: float, alpha: jnp.ndarray) -> jnp.ndarray:
        return r_ee / cusp * (1.0 / (jnp.abs(1.0 + alpha * r_ee)))

    def init() -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ee_par'] = jnp.ones(shape=1)
        params['ee_anti'] = jnp.ones(shape=1)
        return params

    def apply(ee: jnp.ndarray, nelectron: int, params: ParamTree) -> jnp.ndarray:
        return _jastrow_ee(ee, params, nelectron, jastrow_fun=pade_ee_cusp_fun)

    return init, apply


'''
print('ee', ee)
init, apply = make_pade_ee_jastrow()
params1 = init()
print('params', params1)
Jastrow_ee = apply(ee, nelectron=4, params=params1)
'''


def _jastrow_ae(ae: jnp.ndarray, nelectron: int, charges: jnp.array, params: ParamTree,
                jastrow_fun: Callable[[jnp.ndarray, int, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """we also need the format of r_ae and charges to do the summation. To be continued."""
    jastrow_ae = jnp.sum(jastrow_fun(ae, nelectron, charges, params['ae']))
    # print('jastrow_ae', jastrow_ae)
    return jastrow_ae


def make_pade_ae_jastrow() -> ...:
    """create a Jastrow factor for atom-electron cusp condition. Currently, we only consider homogeneous term.
    In this case, we have two atoms. Each atom has two electrons."""

    def pade_ae_cusp_fun(ae: jnp.ndarray, nelectron: int, charges: jnp.array, beta: jnp.ndarray) -> jnp.ndarray:
        # print('r_ae', ae)
        r_ae = jnp.linalg.norm(ae, axis=-1)
        # print('r_ae', r_ae)
        natoms = len(charges)
        charges = jnp.reshape(jnp.repeat(charges, nelectron), (-1, natoms))
        # print('charges', charges)
        'now, we need replicate the charge array.'
        beta = jnp.reshape(beta, (nelectron, natoms))
        # print('beta', beta)
        # print('value_Jastrow_ae', -1 * jnp.float_power((2 * charges), (3/4)) * (1 - jnp.exp(-1 * jnp.float_power((2 * charges), 1/4) * r_ae * beta))/(2 * beta))
        return -1.0 * jnp.float_power((2.0 * charges), (3.0 / 4.0)) * (
                    1.0 - jnp.exp(-1.0 * jnp.float_power((2.0 * charges), 1.0 / 4.0) * r_ae * beta)) / (2.0 * beta)

    def init(nelectron: int, charges: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ae'] = jnp.ones(shape=nelectron * len(charges))
        return params

    def apply(ae: jnp.ndarray, nelectron: int, charges: jnp.ndarray, params: ParamTree) -> jnp.ndarray:
        return _jastrow_ae(ae, nelectron, charges, params, jastrow_fun=pade_ae_cusp_fun)

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
