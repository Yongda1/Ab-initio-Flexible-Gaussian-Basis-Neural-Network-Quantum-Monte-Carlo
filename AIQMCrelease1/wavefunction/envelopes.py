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
from nn import construct_input_features
from jax.scipy.special import sph_harm



pos = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
atoms = jnp.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
ae, ee = construct_input_features(pos=pos, atoms=atoms)
jax.debug.print("ae:{}", ae)


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
    """Create a Slater-type orbital envelop as we show in the slides."""
    def init(natoms: int, nelectrons: int,) -> Sequence[Mapping[str, jnp.ndarray]]:
        """first we need construct the angular momentum vector by specifying the order of polarization.
        0 means only s orbitals, i.e. the number of orbitals is 1.
        1 means p orbitals, i.e. the number of orbitals is 3.
        2 means d orbitals, i.e. the number of orbitals is 5.
        3 means f orbitals, i.e. the number of orbitals is 7.
        """
        num_Y_lm = 1 + 3 + 5
        xi = jnp.ones(shape=(nelectrons, num_Y_lm, natoms))
        return {'xi': xi}

    def apply(ae: jnp.array, xi: jnp.array, natoms: int, nelectrons: int) -> jnp.ndarray:
        """the input for the apply function must be the r. It should be scalar.
        we need assign the electrons to atoms. Now, we have 4 electrons, each atom has two electrons.
        The number one and number three electrons belong to first_atom. The spin configuration is up up down down.
        """
        """we need convert the coordinates to spherical coordinates."""
        def to_spherical(x: jnp.ndarray):
            """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
            radius = jnp.linalg.norm(x)
            phi_ = jnp.arccos(x[2]/radius)
            theta_ = jnp.arctan(x[1]/x[0])
            return jnp.array([radius, theta_, phi_])

        ae = jnp.reshape(ae, (natoms * nelectrons, 3))
        """here, we used vmap to vectorize the function."""
        to_spherical1 = jax.vmap(to_spherical, axis_size=1, out_axes=0)
        spherical_coordinates = to_spherical1(ae)
        theta_coordinates = spherical_coordinates[:, 1]
        phi_coordinates = spherical_coordinates[:, 2]
        """here, n is L, i.e. the angular momentum quantum number."""
        """Assume that all orbitals have the same max angular value l."""
        l_1 = jnp.array([[0]])
        m_1 = jnp.array([[0]])
        l_2 = jnp.array([[1]])
        m_2 = jnp.array([[-1], [0], [1]])
        l_3 = jnp.array([[2]])
        m_3 = jnp.array([[-2], [-1], [0], [1], [2]])

        def angular_function(l: jnp.array, m: jnp.array, theta: jnp.array, phi: jnp.array):
            value = sph_harm(m=m, n=l, theta=theta, phi=phi, n_max=2)
            return value

        angular_function_parallel = jax.vmap(jax.vmap(angular_function, in_axes=(0, None, None, None), out_axes=0),
                                             in_axes=(None, 0, None, None), out_axes=0)
        output1 = angular_function_parallel(l_1, m_1, theta_coordinates, phi_coordinates)
        output2 = angular_function_parallel(l_2, m_2, theta_coordinates, phi_coordinates)
        output3 = angular_function_parallel(l_3, m_3, theta_coordinates, phi_coordinates)
        angular_total = jnp.concatenate([output1, output2, output3], axis=0)
        angular_total = jnp.reshape(angular_total, (9, natoms * nelectrons))
        angular_total = jnp.reshape(angular_total, (9, nelectrons, natoms))
        jax.debug.print("xi_shape:{}", xi.shape)
        angular_total = jnp.transpose(angular_total, (1, 0, 2))

        def multiply(xi: jnp.array, angular_total: jnp.array):
            return xi * angular_total

        jax.debug.print("angular_total_shape:{}", angular_total.shape)
        multiply_parallel = jax.vmap(jax.vmap(multiply, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=0)
        angular_part = multiply_parallel(xi, angular_total)
        angular_part = jnp.sum(angular_part, axis=[2, 3])
        return angular_part

    return Envelope(init=init, apply=apply)


envelope = make_GTO_envelope()
xi = envelope.init(natoms=3, nelectrons=6)
print("xi", xi)
output = envelope.apply(ae, xi['xi'], natoms=3, nelectrons=6)