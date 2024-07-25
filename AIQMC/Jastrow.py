"""we implement pade functions as Jastrow factors."""
import enum
from typing import Any, Callable, Iterable, Mapping, Union
import jax.numpy as jnp
#from nn import construct_input_features
import numpy as np

#pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
#atoms = jnp.array([[0, 0, 0], [1, 1, 1]])
#ae, ee = construct_input_features(pos, atoms, ndim=3)
"""this charge format is just for matching the format of jastrow factors. For convenice, we use it for now."""
#charges = jnp.array([[[2], [2]], [[2], [2]], [[2], [2]], [[2], [2]]])
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
#print("ae", ae)


class JastrowType(enum.Enum):
  """Available Jastrow factors."""
  Pade = enum.auto()


def _jastrow_ee(ee: jnp.ndarray, params: ParamTree, jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Currently, we are sure that we will use the shape of ee array is nelectrons * nelectrons * 3.
     We take the 2 atoms, 4 electrons as the example i.e. spin configurations: up, up, down, down.
    Due to that the jnp.ndarray cannot take agruments as element.
    So we need think how to get parallel spin and anti_parallel spin.Meanwhile, we need be careful about batch version.
    17/07/2024.
    The following part is not an universial method to split the spin configurations. I could leave this problem to strudents."""
    r_ees = jnp.linalg.norm(ee, axis=-1)
    #print("r_ees", r_ees)
    #print(r_ees[0][1], r_ees[2][3])
    """Currently, """
    r_ees_parallel = jnp.ones(2)
    r_ees_parallel_reset = r_ees_parallel.at[0].set(r_ees[0][1])
    r_ees_parallel_reset = r_ees_parallel_reset.at[1].set(r_ees[2][3])
    #print("r_ees_paralles_reset", r_ees_parallel_reset)
    r_ees_anti_parallel = jnp.ones(4)
    r_ees_anti_parallel_set = r_ees_anti_parallel.at[0].set(r_ees[0][2])
    r_ees_anti_parallel_set = r_ees_anti_parallel_set.at[1].set(r_ees[0][3])
    r_ees_anti_parallel_set = r_ees_anti_parallel_set.at[2].set(r_ees[1][2])
    r_ees_anti_parallel_set = r_ees_anti_parallel_set.at[3].set(r_ees[1][3])
    #print("r_ees_anti_parallel_set", r_ees_anti_parallel_set)
    #Jastrow_temp = jastrow_fun(r_ees_parallel_reset, 0.25, params['ee_par'])
    #print("Jastrow_temp", Jastrow_temp)
    jastrow_ee_par = jnp.sum(jastrow_fun(r_ees_parallel_reset, 0.25, params['ee_par']))
    #print("jastrow_ee_par", jastrow_ee_par)
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees_anti_parallel_set, 0.5, params['ee_anti']))

    return jastrow_ee_anti + jastrow_ee_par

def make_pade_ee_jastrow() -> ...:
    """Create a Jastrow factor for electron-electron cusps.
    This cusp means 0.25 or 0.5."""

    def pade_ee_cusp_fun(r_ee: jnp.ndarray, cusp: float, alpha: jnp.ndarray) -> jnp.ndarray:
        #print("r_ee", r_ee)
        #print("alpha", alpha)
        #temp1 = r_ee/cusp
        #print("temp1", temp1)
        #temp2 = 1 + alpha * r_ee
        #print("temp2", temp2)
        return r_ee/cusp * (1/(jnp.abs(1 + alpha * r_ee)))

    def init() -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ee_par'] = jnp.ones(shape=2)
        params['ee_anti'] = jnp.ones(shape=4)
        return params

    def apply(ee: jnp.ndarray, params: ParamTree) -> jnp.ndarray:
        return _jastrow_ee(ee, params, jastrow_fun=pade_ee_cusp_fun)

    return init, apply


#init, apply = make_pade_ee_jastrow()
#params1 = init()
#Jastrow_ee = apply(ee, params1)

"""to be continued 17/7/2024"""
def _jastrow_ae(r_ae: jnp.ndarray, charges: jnp.array, params: ParamTree, jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """we also need the format of r_ae and charges to do the summation. To be continued."""
    jastrow_ae = jnp.sum(jastrow_fun(r_ae, charges, params['ae']))
    return jastrow_ae


def make_pade_ae_jastrow() -> ...:
    """create a Jastrow factor for atom-electron cusp condition. Currently, we only consider homogeneous term.
    In this case, we have two atoms. Each atom has two electrons."""

    def pade_ae_cusp_fun(r: jnp.ndarray, charges: jnp.array, beta: jnp.ndarray, ) -> jnp.ndarray:
        #print("ae", r)
        #print("charges", charges)
        #print("beta", beta)
        #temp2 = charges * 2
        beta = jnp.reshape(beta, (4, 2, 1))
        #print("beta", beta)
        #print(temp2)
        #temp1 = jnp.float_power((2 * charges), 1/4)
        #print(temp1)
        #print("charges", charges)
        #temp3 = temp1 * r * beta
        #print(temp3)
        return -1 * jnp.float_power((2 * charges), (3/4)) * (1 - jnp.exp(-1 * jnp.float_power((-2 * charges), 1/4) * r * beta))/(2 * beta)

    def init() -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ae'] = jnp.ones(shape=8)
        return params

    def apply(r_ae: jnp.ndarray, charges: jnp.ndarray, params: ParamTree) -> jnp.ndarray:
        return _jastrow_ae(r_ae, charges, params, jastrow_fun=pade_ae_cusp_fun)
    
    return init, apply


#init, apply = make_pade_ae_jastrow()
#params2 = init()
#Jastrow_ae = apply(ae, charges, params2)


def get_jastrow(jastrow: JastrowType) -> ...:
    jastrow_ee_init, jastrow_ee_apply = make_pade_ee_jastrow()
    jastrow_ae_init, jastrow_ae_apply = make_pade_ae_jastrow()
    return jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply
