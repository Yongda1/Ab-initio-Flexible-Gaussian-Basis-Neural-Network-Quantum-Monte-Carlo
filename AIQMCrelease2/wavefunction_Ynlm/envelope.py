import enum
from typing import Any, Mapping, Sequence, Union

import attr
from ferminet import curvature_tags_and_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol


def make_isotropic_envelope():
  """Creates an isotropic exponentially decaying multiplicative envelope."""

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del ndim  # unused
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.ones(shape=(natom, output_dim))
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes an isotropic exponentially-decaying multiplicative envelope."""
    del ae, r_ee  # unused
    return jnp.sum(jnp.exp(-r_ae * sigma) * pi, axis=1)

  return init, apply