"""construct Gaussian basis envelopes for the single orbitals."""
from typing_extensions import Protocol
import jax
import jax.numpy as jnp
import attr
from typing import Any, Mapping, Sequence, Union


class EnvelopeInit(Protocol):
    def __call__(self, natom: int, output_dims: Union[int, Sequence[int]], ndim: int) \
            -> Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
        """Returns the evelope parameters"""


class EnvelopeApply(Protocol):
    def __call__(self, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray, **kwargs: jnp.ndarray) -> jnp.ndarray:
        """Returns a multiplicative envelope to ensure boundary conditions are met."""


@attr.s(auto_attribs=True)
class Envelope:
    init: EnvelopeInit
    apply: EnvelopeApply

def make_GTO_envelope() -> Envelope:
    """Create a Slater-type orbital envelop: pi * r^l * exp(-sigma * r**2) * Y_{lm}(\theta, \phi)"""
    def init(natom: int, output_dims: int, ndim: int = 3) -> Mapping[str, jnp.ndarray]:
        pi = jnp.zeros(shape=(natom, output_dims))
        sigma = jnp.tile(jnp.eye(ndim)[..., None, None], [1 , 1, natom, output_dims])
        return {'pi': pi, 'sigma': sigma}

    def apply(ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray, pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:

