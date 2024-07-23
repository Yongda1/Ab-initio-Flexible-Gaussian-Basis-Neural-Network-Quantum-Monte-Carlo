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

import numpy as np
from typing_extensions import Protocol
import jax
import jax.numpy as jnp
import attr
from typing import Any, Mapping, Sequence, Union, Tuple
from nn import construct_input_features
from jax.scipy.special import sph_harm

'''
class EnvelopType(enum.Enum):
    Expand_high_angular_momentum_functions = enum.auto


class EnvelopeLabel(enum.Enum):
    Gaussian = enum.auto()


class EnvelopeInit(Protocol):
    def __call__(self, natom: int, output_dims: Union[int, Sequence[int]], ndim: int, p_order: int) \
            -> Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
        """Returns the envelope parameters"""


class EnvelopeApply(Protocol):
    def __call__(self, ae: jnp.ndarray, ee: jnp.ndarray, **kwargs: jnp.ndarray) -> jnp.ndarray:
        """Returns a multiplicative envelope to ensure boundary conditions are met."""


@attr.s(auto_attribs=True)
class Envelope:
    init: EnvelopeInit
    apply: EnvelopeApply

'''
pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0], [0.2, 0.2, 0.2]])
ae, ee = construct_input_features(pos, atoms, ndim=3)
print("ae", ae)
print("ee", ee)
m = jnp.arange(-1*2, 2+1)
print("m", m)


def make_GTO_envelope():
    """Create a Slater-type orbital envelop as we show in the slides."""
    def init(natom: int, nelectrons: int, ndim: int, order: int,) -> Mapping[str, jnp.ndarray]:
        """first we need construct the angular momentum vector by specifying the order of polarization.
        0 means only s orbitals, i.e. the number of orbitals is 1.
        1 means p orbitals, i.e. the number of orbitals is 3.
        2 means d orbitals, i.e. the number of orbitals is 5.
        3 means f orbitals, i.e. the number of orbitals is 7.
        we usually apply one order higher angular momentum function."""
        n = jnp.arange(order+1)
        print("n", n)
        m = []
        for i in n:
            m.append(jnp.arange(-1*i, i+1))
        print("m", m)
        """Here, we need confirm the size of the angular momentum vector by checking the order.
        we use the orbitals up to f. So, we need 1 + 3 + 5 + 7.
        And this number must be same with the number of the parameters."""
        num_Y_lm = 1 + 3 + 5 + 7
        xi = jnp.ones(shape=(nelectrons, num_Y_lm))
        #print("xi", xi)
        return {"xi": xi}

    def apply(ae: jnp.ndarray, atoms: jnp.ndarray, xi: jnp.ndarray, natoms: int, nelectrons: int) -> jnp.ndarray:
        """the input for the apply function must be the r_effective. It should be scalar.
        we need assign the electrons to atoms. Now, we have 4 electrons, each atom has two electrons.
        The number one and number three electrons belong to first_atom. The spin configuration is up up down down.
        """
        electron_coordinates = ae[0][0] + atoms[0]
        r_ae = jnp.linalg.norm(ae, axis=-1)
        print("r_ae", r_ae)
        print("ae", ae)
        print("xi", xi)
        """we need convert the coordinates to spherical coordinates."""
        def to_spherical(x: jnp.ndarray):
            """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
            radius = jnp.linalg.norm(x)
            phi_ = jnp.arccos(x[2]/radius)
            theta_ = jnp.arctan(x[1]/x[0])
            return jnp.array([radius, theta_, phi_])

        ae = jnp.reshape(ae, (natoms * nelectrons, 3))
        print("ae", ae)
        """here, we used vmap to vectorize the function."""
        to_spherical1 = jax.vmap(to_spherical, axis_size=1, out_axes=0)
        spherical_coordinates = to_spherical1(ae)
        print("spherical", spherical_coordinates)
        theta_coordinates = spherical_coordinates[:, 1]
        phi_coordinates = spherical_coordinates[:, 2]
        print("theta_coordinates", theta_coordinates)
        print("phi_coordinates", phi_coordinates)
        """here, n is L, i.e. the angular momentum quantum number."""
        print(spherical_coordinates[0][1], spherical_coordinates[0][2])
        angular_value_00 = sph_harm(n=jnp.array([0]), m=jnp.array([0]), theta=theta_coordinates, phi=phi_coordinates)
        angular_value_1_1 = sph_harm(n=jnp.array([1]), m=jnp.array([-1]), theta=theta_coordinates, phi=phi_coordinates)
        angular_value_10 = sph_harm(n=jnp.array([1]), m=jnp.array([0]), theta=theta_coordinates, phi=phi_coordinates)
        angular_value_11 = sph_harm(n=jnp.array([1]), m=jnp.array([1]), theta=theta_coordinates, phi=phi_coordinates)
        angular_value = jnp.array([angular_value_00, angular_value_1_1, angular_value_10, angular_value_11])
        #print("angular_value", angular_value)
        angular_value = jnp.transpose(angular_value)
        print("-----------------------------")
        print("angular_value", angular_value)

        return angular_value

    return init, apply


init, apply = make_GTO_envelope()
xi = init(natom=2, nelectrons=4, ndim=3, order=3)
print("xi", xi)
output = apply(ae, atoms, xi, natoms=2, nelectrons=4, )