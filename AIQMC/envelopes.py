"""construct Gaussian basis envelopes for the single orbitals.
Actually, this part is extremely important. Because our orbitals constructed from ccecp basis, our initialization of electrons
 must have order. For example, C atom has 4 effective electrons, i.e. 1 s orbital, 3 p orbitals. The ae array and ee array must
 corresponds to this order. This means that initio positions of electrons must distribute according to this orbitals order.
 The second problem is about the output from neural network. We are not only using the modified r. i.e. electrons positions, also
 modified coefficients, coe(r), modified exponents, xi(r). So these two variables should be done in nn moudle, but not in envelope.
  We have to wait for the output from neural network, then finish this module."""
import numpy as np
from typing_extensions import Protocol
import jax
import jax.numpy as jnp
import attr
from typing import Any, Mapping, Sequence, Union, Tuple


def construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct inputs to AINet from raw electron and atomic positions."""
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


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


pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0]])
print(atoms.shape[1])
ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
print("ae", ae)
print("ee", ee)
print("r_ae", r_ae)
print("r_ee", r_ee)

'''
def make_GTO_envelope() -> Envelope:
    """Create a Slater-type orbital envelop:(1 + alpha (r + Z_eff o g(x(r, R)))*exp(xi)) * pi * r^l * exp(-sigma * r**2) * Y_{lm}(\theta, \phi)"""
    def init(natom: int, output_dims: int, ndim: int = 3) -> Mapping[str, jnp.ndarray]:
        pi = jnp.zeros(shape=(natom, output_dims))
        sigma = jnp.tile(jnp.eye(ndim)[..., None, None], [1 , 1, natom, output_dims])
        return {'pi': pi, 'sigma': sigma}

    def apply(ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray, pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        '''