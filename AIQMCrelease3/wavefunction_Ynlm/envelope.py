import enum
from typing import Any, Mapping, Sequence, Union
import jax
import jax.numpy as jnp
from typing_extensions import Protocol


def make_pp_like_envelope():
  """Creates an isotropic exponentially decaying multiplicative envelope."""

  def init(natom: int, nelectrons: int, ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    params = []
    for electron in range(nelectrons):
      params.append({
        'pi': jnp.ones(shape=(natom, ndim)),
        'sigma': jnp.ones(shape=(natom, ndim)),
        'alpha': jnp.ones(shape=1),
        'beta': jnp.ones(shape=natom),
        'xi': jnp.ones(shape=1),
        'eplion': jnp.ones(shape=(natom, ndim)),
        'mu': jnp.ones(shape=natom),
        'nu': jnp.ones(shape=natom)
      })

    return params

  def apply(orbitals: jnp.array, r_ae: jnp.ndarray, ae: jnp.array, charges: jnp.array, params_envelope) \
          -> jnp.ndarray:
    r_ae = jnp.reshape(r_ae, (-1))
    #jax.debug.print("orbitals:{}", orbitals)
    #jax.debug.print("r_ae:{}", r_ae)
    #jax.debug.print("ae:{}", ae)
    #jax.debug.print("charges:{}", charges)
    #jax.debug.print("params_envelope:{}", params_envelope)
    return (jnp.sum(jnp.exp(-params_envelope['beta'] * r_ae**2) * params_envelope['alpha'] ) + \
          jnp.sum(jnp.exp(-ae * params_envelope['pi']) * params_envelope['sigma'] * params_envelope['xi'])) * orbitals
    #return (jnp.sum(jnp.exp(-params_envelope['beta'] * r_ae**2) * params_envelope['alpha'] * jnp.power(r_ae, params_envelope['mu'] * charges)) + \
          #jnp.sum(jnp.exp(-ae * params_envelope['pi']) * params_envelope['sigma'] * params_envelope['xi'] * jnp.power(ae, jnp.reshape( params_envelope['nu'] * charges, (-1, 1))))) * orbitals

  return init, apply