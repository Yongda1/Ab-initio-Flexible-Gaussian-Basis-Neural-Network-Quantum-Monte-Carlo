"""construct Gaussian basis envelopes for the single orbitals.
Actually, this part is extremely important. Because our orbitals constructed from ccecp basis, our initialization of electrons
 must have order. For example, C atom has 4 effective electrons, i.e. 1 s orbital, 3 p orbitals. The ae array and ee array must
 corresponds to this order. This means that initio positions of electrons must distribute according to this orbitals order.
 The second problem is about the output from neural network. We are not only using the modified r. i.e. electrons positions, also
 modified coefficients, coe(r), modified exponents, xi(r). So these two variables should be done in nn moudle, but not in envelope.
  We have to wait for the output from neural network, then finish this module.
  Currently, we already confirm the shape of output array, it should be h[numer_one_features + number_two_features].
  We can get r easily by multiplying a vector w. 18/07/2024.
  So, we can begin to deal with angular momentum functions in the envelope function."""
import enum
from typing_extensions import Protocol
import jax
import jax.numpy as jnp
import attr
from typing import Any, Mapping, Sequence, Union, Tuple
#from nn import construct_input_features
from jax.scipy.special import sph_harm


'''
pos = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
atoms = jnp.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
ae, ee = construct_input_features(pos=pos, atoms=atoms)
jax.debug.print("ae:{}", ae)
'''

class EnvelopType(enum.Enum):
    PRE_Orbital = enum.auto


class EnvelopeInit(Protocol):
    def __call__(self, natom: int, nelectrons: int) -> Sequence[Mapping[str, jnp.ndarray]]:
        """Returns the envelope parameters"""


class EnvelopeApply(Protocol):
    def __call__(self, ae: jnp.ndarray, xi: jnp.ndarray, natoms: int, nelectrons: int) -> jnp.ndarray:
        """Returns a multiplicative envelope to ensure boundary conditions are met."""


@attr.s(auto_attribs=True)
class Envelope:
    init: EnvelopeInit
    apply: EnvelopeApply


def make_GTO_envelope():
    def init(natoms: int, nelectrons: int,) -> Mapping[str, jnp.ndarray]:
        """first we need construct the angular momentum vector by specifying the order of polarization.
        0 means only s orbitals, i.e. the number of orbitals is 1.
        1 means p orbitals, i.e. the number of orbitals is 3.
        2 means d orbitals, i.e. the number of orbitals is 5.
        3 means f orbitals, i.e. the number of orbitals is 7.
        """
        num_Y_lm = 1 + 3
        xi = jnp.ones(shape=(nelectrons, natoms, num_Y_lm))
        return {'xi': xi}

    def apply(ae: jnp.array, r_ae: jnp.array, xi: jnp.array, natoms: int, nelectrons: int) -> jnp.ndarray:
        """the input for the apply function must be the r. It should be scalar.
        we need assign the electrons to atoms. Now, we have 4 electrons, each atom has two electrons.
        The number one and number three electrons belong to first_atom. The spin configuration is up up down down.
        """
        """we need convert the coordinates to spherical coordinates."""
        temp = ae / r_ae
        Y_0 = 1 / 2 * jnp.sqrt(1 / jnp.pi)
        Y_0 = jnp.repeat(Y_0, natoms * nelectrons)
        Y_0 = jnp.reshape(Y_0, (nelectrons, natoms, -1))
        #jax.debug.print("Y_0:{}", Y_0)
        Y_1 = jnp.sqrt((3 / (4 * jnp.pi))) * temp
        #jax.debug.print("Y_1:{}", Y_1)
        #jax.debug.print("xi:{}", xi)
        Y_total = jnp.concatenate([Y_0, Y_1], axis=-1)
        #jax.debug.print("Y_total:{}", Y_total)
        angular = jnp.sum(jnp.sum(xi * Y_total, axis=-1), axis=-1, keepdims=True)
        #jax.debug.print("angular:{}", angular)
        return angular
        #test = 1 / 2 * jnp.sqrt(1 / jnp.pi), jnp.sqrt(3 / (4 * jnp.pi) * ae / r_ae)


    return Envelope(init=init, apply=apply)

'''
envelope = make_GTO_envelope()
xi = envelope.init(natoms=3, nelectrons=6)
print("xi", xi)
output = envelope.apply(ae, xi['xi'], natoms=3, nelectrons=6)
'''