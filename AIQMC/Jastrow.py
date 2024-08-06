"""we implement pade functions as Jastrow factors."""
import enum
from typing import Any, Callable, Iterable, Mapping, Union, Tuple
import jax.numpy as jnp
#from nn import construct_input_features
import numpy as np


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
charges = jnp.array([2, 2])
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available Jastrow factors."""
  Pade = enum.auto()


def _jastrow_ee(ee: jnp.ndarray, params: ParamTree, nelectron: int, jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """we need develope the method to spit the spin configurations. 01.08.2024"""
    r_ees = jnp.linalg.norm(ee, axis=-1)
    #print('r_ee', r_ees)
    n_spin_up = int(nelectron/2)
    #print(n_spin_up)
    iu = jnp.triu(r_ees)
    #print('iu', iu)
    #print('spin_up', iu[:, :n_spin_up])
    r_ees_parallel_up_up = jnp.ravel(iu[:, :n_spin_up])[jnp.nonzero(jnp.ravel(iu[:, :n_spin_up]))]
    #print('r_ees_parallel_up_up', r_ees_parallel_up_up)
    r_ees_parallel_down_down = jnp.ravel(iu[-n_spin_up:, :])[jnp.nonzero(iu[-n_spin_up, :])]
    #print('r_ees_parallel_down_down', r_ees_parallel_down_down)
    r_ees_parallel = jnp.concatenate([r_ees_parallel_up_up, r_ees_parallel_down_down])
    #print("r_ees_paralles_reset", r_ees_parallel_reset)
    #print('anti_parallel', iu[:n_spin_up, -n_spin_up:])
    r_ees_anti_parallel = jnp.ravel(iu[:n_spin_up, -n_spin_up:])[jnp.nonzero(jnp.ravel(iu[:n_spin_up, -n_spin_up:]))]
    jastrow_ee_par = jnp.sum(jastrow_fun(r_ees_parallel, 0.25, params['ee_par']))
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees_anti_parallel, 0.5, params['ee_anti']))
    print('jastrow_ee_par', jastrow_ee_par)
    print('jastrow_ee_anti', jastrow_ee_anti)

    return jastrow_ee_anti + jastrow_ee_par

def make_pade_ee_jastrow() -> ...:
    """Create a simple Pade Jastrow factor for electron-electron cusps.
    This cusp means 0.25 or 0.5.
    The number of variational parameter is just one. Later, we could think how to add radius dependence into Jastrow parameters.
    think more !!! 06.08.2024"""

    def pade_ee_cusp_fun(r_ee: jnp.ndarray, cusp: float, alpha: jnp.ndarray) -> jnp.ndarray:
        return r_ee/cusp * (1/(jnp.abs(1 + alpha * r_ee)))

    def init() -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ee_par'] = jnp.ones(shape=1)
        params['ee_anti'] = jnp.ones(shape=1)
        return params

    def apply(ee: jnp.ndarray, nelectron: int, params: ParamTree) -> jnp.ndarray:
        return _jastrow_ee(ee, params, nelectron, jastrow_fun=pade_ee_cusp_fun)

    return init, apply


print('ee', ee)
init, apply = make_pade_ee_jastrow()
params1 = init()
print('params', params1)
Jastrow_ee = apply(ee, nelectron=4, params=params1)


"""to be continued 17/7/2024"""
def _jastrow_ae(r_ae: jnp.ndarray, nelectron: int, charges: jnp.array, params: ParamTree, jastrow_fun: Callable[[jnp.ndarray, int, jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """we also need the format of r_ae and charges to do the summation. To be continued."""
    jastrow_ae = jnp.sum(jastrow_fun(r_ae, nelectron, charges, params['ae']))
    return jastrow_ae


def make_pade_ae_jastrow() -> ...:
    """create a Jastrow factor for atom-electron cusp condition. Currently, we only consider homogeneous term.
    In this case, we have two atoms. Each atom has two electrons."""

    def pade_ae_cusp_fun(ae: jnp.ndarray, nelectron: int, charges: jnp.array, beta: jnp.ndarray) -> jnp.ndarray:
        print('r_ae', ae)
        r_ae = jnp.linalg.norm(ae, axis=-1)
        print('r_ae', r_ae)
        natoms = len(charges)
        charges = jnp.reshape(jnp.repeat(charges, nelectron), (-1, natoms))
        print('charges', charges)
        'now, we need replicate the charge array.'
        beta = jnp.reshape(beta, (nelectron, natoms))
        return -1 * jnp.float_power((2 * charges), (3/4)) * (1 - jnp.exp(-1 * jnp.float_power((-2 * charges), 1/4) * r_ae * beta))/(2 * beta)

    def init(nelectron: int, charges: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ae'] = jnp.ones(shape=nelectron*len(charges))
        return params

    def apply(ae: jnp.ndarray, nelectron: int, charges: jnp.ndarray, params: ParamTree) -> jnp.ndarray:
        return _jastrow_ae(ae, nelectron, charges, params, jastrow_fun=pade_ae_cusp_fun)
    
    return init, apply


init, apply = make_pade_ae_jastrow()
params2 = init(nelectron=4, charges=charges)
print('params2', params2)
Jastrow_ae = apply(ae, nelectron=4, charges=charges, params=params2)


def get_jastrow(jastrow: JastrowType) -> ...:
    jastrow_ee_init, jastrow_ee_apply = make_pade_ee_jastrow()
    jastrow_ae_init, jastrow_ae_apply = make_pade_ae_jastrow()
    return jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply
