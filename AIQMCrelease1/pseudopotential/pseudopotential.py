"""Evaluates the pseudopotential Hamiltonian on a wavefunction. 04.09.2024."""

from typing import Sequence, Callable

import chex
import jax
import jax.numpy as jnp
import kfac_jax
#from AIQMCrelease1.main import main_adam
from jax import Array
from typing import Tuple
from AIQMCrelease1.wavefunction import nn
from typing_extensions import Protocol

from AIQMCrelease1.wavefunction.nn import AINetData

"""for the implementation in the codes, we need consider the full situations with more atoms. 
However, light atoms basically dont have l=2 in the pseudopotential. 
Then, for convenience, we only take the CO2 molecular as the example to test this module."""

#signed_network, data, params, log_network = main_adam.main()
#jax.debug.print("data:{}", data)
#ndim = 3
#ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
#jax.debug.print("ae:{}", ae)

'''
"""the pseudopotential arrays for molecular CO2"""
symbol = ['C', 'O', 'O']
rn_local_general = jnp.array([[1, 3, 2], [1, 3, 2], [1, 3, 2]])
rn_non_local_general = jnp.array([[2], [2], [2]])
local_coefficient_general = jnp.array([[4.00000, 57.74008, -25.81955], [6.000000, 73.85984, -47.87600], [6.000000, 73.85984, -47.87600]])
nonlocal_coefficient_general = jnp.array([[52.13345], [85.86406], [85.86406]])
local_exponent_general = jnp.array([[14.43502, 8.39889, 7.38188], [12.30997, 14.76962, 13.71419], [12.30997, 14.76962, 13.71419]])
nonlocal_exponent_general = jnp.array([[7.76079], [13.65512], [13.65512]])
'''

"""The pseduopotential arrays for Ge, Si, O. we try these three atoms. 
However, we don't use the corresponding number of electrons in the pseudopotential file. 
we enlarge the shape of Rn_local to the largest element in the array. 0 means that it doesn't exist.
For convenience, we also enlarge the corresponding coefficients and exponents. This parallel way is working, but I am not sure
if the way can work efficiently.
The above method could be applied in the nonlcoal part. However, with higher angular momentum functions in the pp file, 
we have to change the parallel mechanism.
now, we need think about how to implement the P_l functions efficiently.
"""

'''
symbol = ['Ge', 'Si', 'O']
Rn_local = jnp.array([[1.0, 3.0, 2.0, 2.0], [1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0]])
Rn_non_local = jnp.array([[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                          [[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                          [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
Local_coes = jnp.array([[4.0,      5.9158506497680, -12.033712959815, 1.283543489065],
                        [4.000000, 20.673264,       -14.818174,       0],
                        [6.000000, 73.85984,        -47.87600,        0]])
Local_exps = jnp.array([[1.478962662442, 3.188905647765, 1.927438978253, 1.545539235916],
                        [5.168316,       8.861690,       3.933474,       0],
                        [12.30997,       14.76962,       13.71419,       0]])
Non_local_coes = jnp.array([[[43.265429324814, -1.909339873965], [35.263014141211, 0.963439928853], [2.339019442484, 0.541380654081]],
                            [[14.832760,       26.349664],       [7.621400,        10.331583],      [0,              0]],
                            [[85.86406,         0],               [0,               0],              [0,              0]]])
Non_local_exps = jnp.array([[[2.894473589836, 1.550339816290], [2.986528872039, 1.283381203893], [1.043001142249, 0.554562729807]],
                            [[9.447023,       2.553812],       [3.660001,       1.903653],       [0,              0]],
                            [[13.65512,        0],             [0,              0],              [0,              0]]])
'''


class LocalPPEnergy(Protocol):
    def __call__(self, data: nn.AINetData) -> jnp.array:
        """Returns the local pp energy of a Hamiltonian at a configuration."""


class NonlocalPPcoes(Protocol):
    def __call__(self, data: nn.AINetData) -> jnp.array:
        """Returns the nonlocal coes of a hamiltonian at a configuration."""


class NonlocalPPPoints(Protocol):
    def __call__(self, data: nn.AINetData, params: nn.ParamTree, Points: jnp.array, weights: float) \
            -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """Return the spherical integration points information"""


def local_pp_energy(nelectrons: int,
                    natoms: int,
                    ndim: int,
                    rn_local: jnp.array,
                    local_coefficient: jnp.array,
                    local_exponent: jnp.array,) -> LocalPPEnergy:
    """calculate the local part of pseudopotential energy.
    we need make the method be general. It means that we could have many atoms which may enlarge the dimension of the coefficient and exponent array.
    to be continued 26.12.2024."""
    rn_local = rn_local - 2

    def exp_single(r_ae_inner: jnp.array,
                   local_exponent_inner: jnp.array,
                   rn_local_inner: jnp.array,
                   local_coefficient_inner: jnp.array):
        return local_coefficient_inner * r_ae_inner ** rn_local_inner * jnp.exp(
            -local_exponent_inner * jnp.square(r_ae_inner))

    local_part2_parallel = jax.vmap(exp_single, in_axes=(0, None, None, None), out_axes=0)

    def pp_local_part_energy(data: nn.AINetData):
        ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
        r_ae = jnp.linalg.norm(ae, axis=-1)
        """the following line is the math formula -1 * Z_eff/r_ae"""
        local_part1 = -1 * data.charges / r_ae
        r_ae = jnp.reshape(r_ae, (nelectrons, natoms, 1))
        local_energy_part2 = local_part2_parallel(r_ae, local_exponent, rn_local, local_coefficient)
        local_energy_part2 = jnp.sum(local_energy_part2, axis=-1)
        total_local_energy = local_part1 + local_energy_part2
        return total_local_energy
    return pp_local_part_energy
        

'''
"""to be continued 26.12.2024."""
get_local_part_energy_test = local_pp_energy(nelectrons=16,
                                          natoms=3,
                                          ndim=3,
                                          rn_local=Rn_local,
                                          local_coefficient=Local_coes,
                                          local_exponent=Local_exps)

get_local_part_energy_test_parallel = jax.vmap(get_local_part_energy_test,
                                               in_axes=(nn.AINetData(positions=0, atoms=0, charges=0)), out_axes=0)
'''


def get_non_v_l(ndim: int,
                nelectrons: int,
                natoms: int,
                rn_non_local: jnp.array,
                non_local_coefficient: jnp.array,
                non_local_exponent: jnp.array) -> NonlocalPPcoes:
    """This function is working. Because the nonlocal part has only one parameter."""

    def exp_non_single(r_ae_inner: jnp.array,
                       rn_non_local_inner: jnp.array,
                       non_local_coefficient_inner: jnp.array,
                       non_local_exponent_inner: jnp.array):
        return non_local_coefficient_inner * (r_ae_inner ** rn_non_local_inner) * jnp.exp(-non_local_exponent_inner * jnp.square(r_ae_inner))

    non_local_parallel = jax.vmap(exp_non_single, in_axes=(0, None, None, None), out_axes=0)

    def get_non_local_coe(data: nn.AINetData):
        ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ae = jnp.reshape(r_ae, (nelectrons, natoms, 1))
        non_local_output = non_local_parallel(r_ae, non_local_exponent, rn_non_local, non_local_coefficient)
        non_local_output = jnp.sum(non_local_output, axis=-1)
        return non_local_output
    """i dont know here is an warning. 27.12.2024."""
    return get_non_local_coe


'''
get_non_local_coe_test = get_non_v_l(ndim=3,
                                     nelectrons=16,
                                     natoms=3,
                                     rn_non_local=Rn_non_local,
                                     non_local_coefficient=Non_local_coes,
                                     non_local_exponent=Non_local_exps)

get_non_local_coe_test_parallel = jax.vmap(get_non_local_coe_test,
                                           in_axes=(nn.AINetData(positions=0, atoms=0, charges=0)), out_axes=0)
'''


def generate_quadrature_grids():
    """generate quadrature grids from Mitas, Shirley, and Ceperley."""
    """Generate in Cartesian grids for octahedral symmetry.
    We are not going to give more options for users, so just default 50 integration points."""
    octpts = jnp.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T
    nonzero_count = jnp.count_nonzero(octpts, axis=1)
    OA = octpts[nonzero_count == 1]
    OB = octpts[nonzero_count == 2] / jnp.sqrt(2)
    OC = octpts[nonzero_count == 3] / jnp.sqrt(3)
    d1 = OC * jnp.sqrt(3 / 11)
    OD1 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0], (1, -1)), jnp.reshape(d1[:, 1], (1, -1)), jnp.reshape(d1[:, 2] * 3, (1, -1))), axis=0))
    OD2 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0], (1, -1)), jnp.reshape(d1[:, 1] * 3, (1, -1)), jnp.reshape(d1[:, 2], (1, -1))), axis=0))
    OD3 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0] * 3, (1, -1)), jnp.reshape(d1[:, 1], (1, -1)), jnp.reshape(d1[:, 2], (1, -1))), axis=0))
    OD = jnp.concatenate((OD1, OD2, OD3), axis=0)
    weights = jnp.array([[4/315], [64/2835], [27/1280], [14641/725760]])
    return OA, OB, OC, OD, weights


#output2 = generate_quadrature_grids()
#jax.debug.print("output2:{}", output2)

def get_rot(batch_size: int, key: chex.PRNGKey):
    key, subkey = jax.random.split(key)
    rot = jax.random.orthogonal(key=key, n=3, shape=(batch_size,))
    OA, OB, OC, OD, weights = generate_quadrature_grids()
    Points_OA = jnp.einsum('jkl,ik->jil', rot, OA,)
    Points_OB = jnp.einsum('jkl,ik->jil', rot, OB,)
    Points_OC = jnp.einsum('jkl,ik->jil', rot, OC,)
    Points_OD = jnp.einsum('jkl,ik->jil', rot, OD,)
    return Points_OA, Points_OB, Points_OC, Points_OD, weights


def P_l(x, list_l: float):
    """
    create the legendre polynomials functions
    :param x: cos(theta)
    :param list_l: the angular momentum functions used in the calculation.
    :return:
    """
    if list_l == 0:
        return jnp.ones(x.shape)
    if list_l == 1:
        return jnp.ones(x.shape), x
    if list_l == 2:
        return jnp.ones(x.shape), x, 0.5 * (3 * x * x - 1)
    if list_l == 3:
        return jnp.ones(x.shape), x, 0.5 * (3 * x * x - 1), 0.5 * (5 * x * x * x - 3 * x)


def get_P_l(nelectrons: int, natoms: int, ndim: int, log_network_inner: nn.AINetLike) -> NonlocalPPPoints:
    """currently, we apply CO2 molecular into the codes. So, we need debug this part again."""

    def rot_coords_single(r_ae_inner: jnp.array, Points_inner: jnp.array):
        #jax.debug.print("Points_inner_shape:{}", Points_inner.shape)
        #jax.debug.print("r_ae_inner:{}", r_ae_inner)
        return r_ae_inner * Points_inner

    rot_coords_parallel = jax.vmap(jax.vmap(rot_coords_single, in_axes=(0, None), out_axes=0), in_axes=(0, None),
                                   out_axes=0)

    def calculate_cos_theta_single(ae_inner_1: jnp.array, roted_coords_inner_1: jnp.array):
        return jnp.sum(ae_inner_1 * roted_coords_inner_1, axis=-1) / \
               (jnp.linalg.norm(ae_inner_1) * jnp.linalg.norm(roted_coords_inner_1))


    calculate_cos_theta_parallel = jax.vmap(jax.vmap(calculate_cos_theta_single, in_axes=(0, 0), out_axes=0),
                                            in_axes=(0, 0))


    def return_arrays(x2: jnp.array, roted_coords: jnp.array, order1: jnp.array):
        temp = x2.at[order1].set(roted_coords)
        temp = jnp.reshape(temp, (-1))
        return temp

    return_arrays_parallel = jax.vmap(jax.vmap(jax.vmap(return_arrays,
                                                        in_axes=(None, 0, None), out_axes=0),
                                               in_axes=(None, 0, None), out_axes=0),
                                      in_axes=(None, 0, 0))

    batch_lognetwork = jax.vmap(jax.vmap(jax.vmap(log_network_inner,
                                                  in_axes=(None, 0, None, None), out_axes=0),
                                         in_axes=(None, 0, None, None), out_axes=0),
                                in_axes=(None, 0, None, None), out_axes=0)

    def generate_points_information(data: nn.AINetData, params: nn.ParamTree, Points: jnp.array, weights: float):
        ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ae = jnp.reshape(r_ae, (nelectrons, natoms, 1))
        jax.debug.print("r_ae:{}", r_ae)
        denominator = log_network_inner(params, data.positions, data.atoms, data.charges)
        roted_coords = rot_coords_parallel(r_ae, Points)
        jax.debug.print("roted_coords:{}", roted_coords.shape)
        cos_theta = calculate_cos_theta_parallel(ae, roted_coords)
        order = jnp.arange(0, nelectrons, step=1)
        x1 = data.positions
        x2 = jnp.reshape(x1, (nelectrons, ndim))
        #jax.debug.print("x2:{}", x2)
        roted_configurations = return_arrays_parallel(x2, roted_coords, order)
        roted_wavefunciton_value = batch_lognetwork(params, roted_configurations, data.atoms, data.charges)
        ratios = roted_wavefunciton_value / denominator * weights
        return cos_theta, ratios, roted_configurations, weights, roted_coords
    return generate_points_information


'''
generate_points_information_test = get_P_l(nelectrons=16, natoms=3, ndim=3, log_network_inner=log_network)
generate_points_information_test_parallel = jax.pmap(jax.vmap(generate_points_information_test,
                                    in_axes=(nn.AINetData(positions=0, atoms=0, charges=0), None, 0,  None)),
                           in_axes=(0, 0, None, None,))
'''
#output2 = get_P_l_parallel(data, params, Points_OA, weights[0], l_list)

