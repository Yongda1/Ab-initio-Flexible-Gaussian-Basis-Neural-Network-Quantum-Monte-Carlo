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
        'nu': jnp.ones(shape=natom)})

    return params

  def apply(orbitals: jnp.array, r_ae: jnp.ndarray, ae: jnp.array, charges: jnp.array, params_envelope) \
          -> jnp.ndarray:
    r_ae = jnp.reshape(r_ae, (-1))
    return (jnp.sum(jnp.exp(-params_envelope['beta'] * r_ae**2) * params_envelope['alpha']) +
            jnp.sum(jnp.exp(-ae * params_envelope['pi']) * params_envelope['sigma'] * params_envelope['xi'])) * orbitals

  return init, apply