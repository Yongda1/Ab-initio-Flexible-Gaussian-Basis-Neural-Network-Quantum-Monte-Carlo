"""we implement pade functions as Jastrow factors."""
import enum
from typing import Any, Callable, Iterable, Mapping, Union
import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]

class JastrowType(enum.Enum):
    PADE_EE_AE = enum.auto()


def _jastrow_ee(r_ee: jnp.ndarray, params: ParamTree, nspins: tuple[int, int], jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Here, we still are not sure which type of vector can be used in the neural network.
    So, we only keep an interface here."""
    jastrow_ee_par = jnp.sum(jastrow_fun(r_ees_parallel, 0.25, params['ee_par']))
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees_anti_parallel, 0.5, params['ee_anti']))

    return jastrow_ee_anti + jastrow_ee_par

def make_pade_ee_jastrow() -> ...:
    """Create a Jastrow factor for electron-electron cusps.
    This cusp means 0.25 or 0.5."""

    def pade_ee_cusp_fun(r: jnp.ndarray, cusp: float, alpha: jnp.ndarray) -> jnp.ndarray:
        return r/cusp * (jnp.abs(1 + alpha * r)) ^ (-1)

    def init() -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ee_par'] = jnp.ones(shape=1,)
        params['ee_anti'] = jnp.ones(shape=1)
        return params

    def apply(r_ee: jnp.ndarray, params: ParamTree, nspins: tuple[int, int]) -> jnp.ndarray:
        return _jastrow_ee(r_ee, params, nspins, jastrow_fun=pade_ee_cusp_fun)

    return init, apply


def _jastrow_ae(r_ae: jnp.ndarray, params: ParamTree, jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """we also need the format of r_ae and charges to do the summation. To be continued."""
    jastrow_ae = jnp.sum()


def make_pade_ae_jastrow() -> ...:
    """create a Jastrow factor for atom-electron cusp condition."""

    def pade_ae_cusp_fun(r: jnp.ndarray, beta: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
        return -1 * (2 * charges) ^ (3/4) * (1 - jnp.exp(-(2 * charges) ^ (1/4) * r * beta))/(2 * beta)

    def init() -> Mapping[str, jnp.ndarray]:
        params = {}
        params['ae'] = jnp.ones(shape=1)
        return params

    def apply(r_ae: jnp.ndarray, charges: jnp.ndarray, params: ParamTree, nspins: tuple[int, int], ) -> jnp.ndarray:
        return _jastrow_ae(r_ae, params, jastrow_fun=pade_ae_cusp_fun)
    
    return init, apply

def get_jastrow(jastrow: JastrowType):
        jastrow_ee_init, jastrow_ee_apply = make_pade_ee_jastrow()
        jastrow_ae_init, jastrow_ae_apply = make_pade_ae_jastrow()
        return jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply
