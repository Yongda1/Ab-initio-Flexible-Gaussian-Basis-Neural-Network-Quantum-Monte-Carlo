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

'''
def construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Construct inputs to AINet from raw electron and atomic positions.
    Here, we assume that the electron spin is up and down along the axis=0 in array pos.
    So, the pairwise distance ae also follows this order.
        pos: electron positions, Shape(nelectrons * dim)
        atoms: atom positions. Shape(natoms, ndim)
    """
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    return ae, ee
pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0], [0.2, 0.2, 0.2]])
ae, ee = construct_input_features(pos, atoms, ndim=3)
print("ae", ae)
print("ee", ee)
m = jnp.arange(-1*2, 2+1)
print("m", m)
'''

def make_GTO_envelope():
    """Create a Slater-type orbital envelop as we show in the slides."""
    def init(natoms: int, nelectrons: int,) -> Sequence[Mapping[str, jnp.ndarray]]:
        """first we need construct the angular momentum vector by specifying the order of polarization.
        0 means only s orbitals, i.e. the number of orbitals is 1.
        1 means p orbitals, i.e. the number of orbitals is 3.
        2 means d orbitals, i.e. the number of orbitals is 5.
        3 means f orbitals, i.e. the number of orbitals is 7.
        we usually apply one order higher angular momentum function.
        This part is wrong. We need fix it."""
        """Here, we need confirm the size of the angular momentum vector by checking the order.
        we use the orbitals up to p. So, we need 1 + 3 .
        And this number must be same with the number of the parameters."""
        num_Y_lm = 1 + 3
        xi = jnp.ones(shape=(num_Y_lm * natoms, nelectrons))
        params = []
        for _ in jnp.arange(nelectrons):
            params.append({'xi': xi})
        #print("xi", xi)
        #print('shape of xi', jnp.shape(xi))
        #print('params', params)
        return params

    def apply(ae: jnp.ndarray, params, natoms: int, nelectrons: int) -> jnp.ndarray:
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
        #print("ae", ae)
        """here, we used vmap to vectorize the function."""
        to_spherical1 = jax.vmap(to_spherical, axis_size=1, out_axes=0)
        spherical_coordinates = to_spherical1(ae)
        #print("spherical", spherical_coordinates)
        theta_coordinates = spherical_coordinates[:, 1]
        phi_coordinates = spherical_coordinates[:, 2]
        #print("theta_coordinates", theta_coordinates)
        #print("phi_coordinates", phi_coordinates)
        """here, n is L, i.e. the angular momentum quantum number."""
        #print(spherical_coordinates[0][1], spherical_coordinates[0][2])
        angular_value_00 = sph_harm(n=jnp.array([0]), m=jnp.array([0]), theta=theta_coordinates, phi=phi_coordinates, n_max=2)
        angular_value_1_1 = sph_harm(n=jnp.array([1]), m=jnp.array([-1]), theta=theta_coordinates, phi=phi_coordinates, n_max=2)
        angular_value_10 = sph_harm(n=jnp.array([1]), m=jnp.array([0]), theta=theta_coordinates, phi=phi_coordinates, n_max=2)
        angular_value_11 = sph_harm(n=jnp.array([1]), m=jnp.array([1]), theta=theta_coordinates, phi=phi_coordinates, n_max=2)
        angular_value = jnp.array([angular_value_00, angular_value_1_1, angular_value_10, angular_value_11])
        #print("angular_value", angular_value)
        angular_value = jnp.transpose(angular_value)
        #print("-----------------------------")
        #print("angular_value", angular_value)
        #print("angular_value", angular_value)
        num_Y_lm = 1 + 3
        angular_value = jnp.reshape(angular_value, (nelectrons, natoms, num_Y_lm))
        #print("angular_value", angular_value)
        #temp = angular_value * xi
        #print("temp", temp)
        #print("shape of angular_value", jnp.shape(angular_value))
        #print('shape of xi', jnp.shape(xi))
        #print('type of xi', type(xi))
        #print(xi[0])
        l_com_ang = [jnp.dot(jnp.reshape(ang, (1, -1)), jnp.array(x['xi'])) for ang, x in zip(angular_value, params)]
        #l_com_ang = jnp.sum(jnp.sum(angular_value * xi, axis=-1), axis=-1)
        #print("l_com_ang", l_com_ang)

        return l_com_ang

    return Envelope(init=init, apply=apply)

'''
envelope = make_GTO_envelope()
xi = envelope.init(natoms=2, nelectrons=4)
print("xi", xi)
output = envelope.apply(ae, xi, natoms=2, nelectrons=4)
'''