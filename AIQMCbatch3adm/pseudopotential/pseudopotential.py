"""Evaluates the pseudopotential Hamiltonian on a wavefunction. 04.09.2024."""

from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import kfac_jax
from AIQMCbatch3adm import main_adam
from AIQMCbatch3adm import nn

"""for the implementation in the codes, we need consider the full situations with more atoms. 
However, light atoms basically dont have l=2 in the pseudopotential. 
Then, for convenience, we only take the CO2 molecular as the example to test this module."""

signed_network, data, params, log_network = main_adam.main()
#jax.debug.print("data:{}", data)
ndim = 3
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

def get_v_l(data: nn.AINetData, rn_local: jnp.array, local_coefficient: jnp.array, local_exponent: jnp.array,):
    """calculate the local part of pseudopotential energy.
    we need make the method be general. It means that we could have many atoms which may enlarge the dimension of the coefficient and exponent array."""
    nelectron = 16
    natoms = 3
    rn_local = rn_local - 2
    ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1)
    """the following line is the math formula -1 * Z_eff/r_ae"""
    local_part1 = -1 * data.charges/r_ae
    #jax.debug.print("local_part1:{}", local_part1)
    r_ae = jnp.reshape(r_ae, (nelectron, natoms, 1))


    def exp_single(r_ae: jnp.array, local_exponent: jnp.array, rn_local: jnp.array, local_coefficient: jnp.array):
        #jax.debug.print("r_ae:{}", r_ae)
        #jax.debug.print("local_exponent:{}", local_exponent)
        #jax.debug.print("rn_local:{}", rn_local)
        #jax.debug.print("local_coefficient:{}", local_coefficient)
        return local_coefficient * r_ae**rn_local * jnp.exp(-local_exponent * jnp.square(r_ae))

    local_part2_parallel = jax.vmap(exp_single, in_axes=(0, None, None, None), out_axes=0)
    local_energy_part2 = local_part2_parallel(r_ae, local_exponent, rn_local, local_coefficient)
    #jax.debug.print("local_energy_part2:{}", local_energy_part2)
    local_energy_part2 = jnp.sum(local_energy_part2, axis=-1)
    total_local_energy = local_part1 + local_energy_part2
    #jax.debug.print("total_local_energy:{}", total_local_energy)

    return total_local_energy


get_v_l_parallel = jax.pmap(jax.vmap(get_v_l,
                                    in_axes=(nn.AINetData(positions=0, atoms=0, charges=0), None, None, None,)),
                           in_axes=(0, None, None, None,))
"""if we have more hosts, we can duplicate the arrays to multi devices."""
#output = get_v_l_parallel(data, Rn_local, Local_coes, Local_exps)


def get_non_v_l(data: nn.AINetData, rn_non_local: jnp.array, non_local_coefficient: jnp.array, non_local_exponent: jnp.array):
    """This function is working. Because the nonlocal part has only one parameter."""
    nelectrons = 16
    natoms = 3
    ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1)
    r_ae = jnp.reshape(r_ae, (nelectrons, natoms, 1))
    #jax.debug.print("r_ae:{}", r_ae)

    def exp_non_single(r_ae: jnp.array, rn_non_local: jnp.array, non_local_coefficient: jnp.array, non_local_exponent: jnp.array):
        #jax.debug.print("r_ae:{}", r_ae)
        #jax.debug.print("rn_non_local:{}", rn_non_local)
        return non_local_coefficient * (r_ae ** rn_non_local) * jnp.exp(-non_local_exponent * jnp.square(r_ae))

    non_local_parallel = jax.vmap(exp_non_single, in_axes=(0, None, None, None), out_axes=0)
    non_local_output = non_local_parallel(r_ae,  non_local_exponent, rn_non_local, non_local_coefficient)
    non_local_output = jnp.sum(non_local_output, axis=-1)
    #jax.debug.print("non_local_output:{}", non_local_output)
    return non_local_output



get_non_v_l_parallel = jax.pmap(jax.vmap(get_non_v_l,
                                    in_axes=(nn.AINetData(positions=0, atoms=0, charges=0), None, None, None,)),
                           in_axes=(0, None, None, None,))
#output1 = get_non_v_l_parallel(data, Rn_non_local, Non_local_coes, Non_local_exps)


def generate_quadrature_grids():
    """generate quadrature grids from Mitas, Shirley, and Ceperley."""
    """Generate in Cartesian grids for octahedral symmetry.
    We are not going to give more options for users, so just default 50 integration points."""
    octpts = jnp.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T
    #jax.debug.print("octpts:{}", octpts)
    nonzero_count = jnp.count_nonzero(octpts, axis=1)
    #jax.debug.print("nonzero_count:{}", nonzero_count)
    OA = octpts[nonzero_count == 1]
    OB = octpts[nonzero_count == 2] / jnp.sqrt(2)
    OC = octpts[nonzero_count == 3] / jnp.sqrt(3)
    #jax.debug.print("OA:{}", OA)
    #jax.debug.print("OB:{}", OB)
    #jax.debug.print("OC:{}", OC)
    d1 = OC * jnp.sqrt(3 / 11)
    #jax.debug.print("d1:{}", d1)
    OD1 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0], (1, -1)), jnp.reshape(d1[:, 1], (1, -1)), jnp.reshape(d1[:, 2] * 3, (1, -1))), axis=0))
    OD2 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0], (1, -1)), jnp.reshape(d1[:, 1] * 3, (1, -1)), jnp.reshape(d1[:, 2], (1, -1))), axis=0))
    OD3 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0] * 3, (1, -1)), jnp.reshape(d1[:, 1], (1, -1)), jnp.reshape(d1[:, 2], (1, -1))), axis=0))
    #jax.debug.print("OD1:{}", OD1)
    #jax.debug.print("OD2:{}", OD2)
    #jax.debug.print("OD3:{}", OD3)
    OD = jnp.concatenate((OD1, OD2, OD3), axis=0)
    #jax.debug.print("OD:{}", OD)
    #coordinates = jnp.stack((OA, OB, OC, OD), axis=1)
    weights = jnp.array([[4/315], [64/2835], [27/1280], [14641/725760]])
    return OA, OB, OC, OD, weights


#output2 = generate_quadrature_grids()
#jax.debug.print("output2:{}", output2)

def get_rot(batch_size: int, key: chex.PRNGKey):
    """actually, here, we generate the normal rotation matrix to """
    key, subkey = jax.random.split(key)
    """here, we dont use random.Rotation. Because this function is not working currently."""
    rot = jax.random.orthogonal(key=key, n=3, shape=(batch_size,))
    #jax.debug.print("rot:{}", rot)
    OA, OB, OC, OD, weights = generate_quadrature_grids()
    #jax.debug.print("OA:{}", OA)
    """actually, I dont understand how to use jnp.einsum, but currently it is working."""
    Points_OA = jnp.einsum('jkl,ik->jil', rot, OA,)
    Points_OB = jnp.einsum('jkl,ik->jil', rot, OB,)
    Points_OC = jnp.einsum('jkl,ik->jil', rot, OC,)
    Points_OD = jnp.einsum('jkl,ik->jil', rot, OD,)
    #jax.debug.print("Points_OD:{}", Points_OD)
    return Points_OA, Points_OB, Points_OC, Points_OD, weights


def P_l(x, list_l: float):
    """
    create the legendre polynomials functions
    :param x: cos(theta)
    :param list_l: the angular momentum functions used in the calculation. For example, list_l = [1, 1, 1, 0, ] means s, p, d, no f
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


def get_P_l(data: nn.AINetData, params: nn.ParamTree, Points: jnp.array, weights: float):
    """currently, we apply CO2 molecular into the codes. So, we need debug this part again."""
    nelectrons = 16
    natoms = 3
    ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1)
    #jax.debug.print("data:{}", data)
    r_ae = jnp.reshape(r_ae, (nelectrons, natoms, 1))

    denominator = log_network(params, data.positions, data.atoms, data.charges)
    #jax.debug.print("denominator:{}", denominator)
    #jax.debug.print("Points:{}", Points)

    def rot_coords_single(r_ae: jnp.array, Points: jnp.array):
        #jax.debug.print("r_ae:{}", r_ae)
        return r_ae * Points

    rot_coords_parallel = jax.vmap(jax.vmap(rot_coords_single, in_axes=(0, None), out_axes=0), in_axes=(0, None), out_axes=0)
    roted_coords = rot_coords_parallel(r_ae, Points)

    def calculate_cos_theta_single(ae: jnp.array, roted_coords: jnp.array):
        return jnp.sum(ae * roted_coords, axis=-1)/(jnp.linalg.norm(ae) * jnp.linalg.norm(roted_coords))

    calculate_cos_theta_parallel = jax.vmap(jax.vmap(calculate_cos_theta_single, in_axes=(0, 0), out_axes=0), in_axes=(0, 0))
    cos_theta = calculate_cos_theta_parallel(ae, roted_coords)

    """then we need calculate the value of the wavefunction."""
    order = jnp.arange(0, nelectrons, step=1)
    x1 = data.positions
    x2 = jnp.reshape(x1, (nelectrons, ndim))

    def return_arrays(x2: jnp.array, roted_coords: jnp.array, order1: jnp.array):
        temp = x2.at[order1].set(roted_coords)
        temp = jnp.reshape(temp, (-1))
        return temp

    return_arrays_parallel = jax.vmap(jax.vmap(jax.vmap(return_arrays,
                                                        in_axes=(None, 0, None), out_axes=0),
                                               in_axes=(None, 0, None), out_axes=0),
                                      in_axes=(None, 0, 0))

    roted_configurations = return_arrays_parallel(x2, roted_coords, order)
    #jax.debug.print("roted_configurations:{}", roted_configurations)
    batch_lognetwork = jax.vmap(jax.vmap(jax.vmap(log_network,
                                                  in_axes=(None, 0, None, None), out_axes=0),
                                in_axes=(None, 0, None, None), out_axes=0),
                                in_axes=(None, 0, None, None), out_axes=0)
    roted_wavefunciton_value = batch_lognetwork(params, roted_configurations, data.atoms, data.charges)
    #jax.debug.print("roted_wavefunction_value:{}", roted_wavefunciton_value)
    ratios = roted_wavefunciton_value/denominator * weights
    #jax.debug.print("ratios:{}", ratios)
    #jax.debug.print("cos_theta:{}", cos_theta)
    """the following part is not general. We need think about the situation like CO2 or SiO2. 2.12.2024."""
    return cos_theta, ratios, roted_configurations, weights





get_P_l_parallel = jax.pmap(jax.vmap(get_P_l,
                                    in_axes=(nn.AINetData(positions=0, atoms=0, charges=0), None, 0,  None)),
                           in_axes=(0, 0, None, None,))
#output2 = get_P_l_parallel(data, params, Points_OA, weights[0], l_list)

def total_energy_pseudopotential(data: nn.AINetData, params: nn.ParamTree, rn_local_general: jnp.array, rn_non_local_general: jnp.array,
                                 local_coefficient_general: jnp.array, nonlocal_coefficient_general: jnp.array,
                                 local_exponent_general: jnp.array, nonlocal_exponent_general: jnp.array, nelectrons: int, natoms: int, list_l: int, batch_size: int):
    """This function caluclates the energy of pseudopotential.
    For the pp of C and O, only l=0 contributes to the nonlocal part.
    we have more problems here. If all atoms have the same shape of the pp parameters, it is ok.
    But if one of the atoms has higher angular momentum functions, I don't know how to do it efficiently."""
    local_part_energy = get_v_l_parallel(data, rn_local_general, local_coefficient_general, local_exponent_general)
    #jax.debug.print("local_part_energy:{}", local_part_energy)
    local_part_energy = jnp.sum(jnp.sum(local_part_energy, axis=-1), axis=-1)
    #jax.debug.print("local_part_energy:{}", local_part_energy)
    nonlocal_parameters = get_non_v_l_parallel(data, rn_non_local_general, nonlocal_coefficient_general, nonlocal_exponent_general)
    #jax.debug.print("nonlocal_parameters:{}", nonlocal_parameters)
    '''here, we only have s angular momentum function.'''
    key = jax.random.PRNGKey(1)
    # sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    # sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    Points_OA, Points_OB, Points_OC, Points_OD, weights = get_rot(batch_size, key)

    cos_theta_OA, ratios_OA, roted_configurations_OA, weights_OA = get_P_l_parallel(data, params, Points_OA, weights[0])
    cos_theta_OB, ratios_OB, roted_configurations_OB, weights_OB = get_P_l_parallel(data, params, Points_OB, weights[1])
    cos_theta_OC, ratios_OC, roted_configurations_OC, weights_OC = get_P_l_parallel(data, params, Points_OC, weights[2])
    cos_theta_OD, ratios_OD, roted_configurations_OD, weights_OD = get_P_l_parallel(data, params, Points_OD, weights[3])
    output_OA = jnp.sum(jnp.array(P_l(cos_theta_OA, list_l=list_l)) * ratios_OA, axis=-1)
    output_OB = jnp.sum(jnp.array(P_l(cos_theta_OB, list_l=list_l)) * ratios_OB, axis=-1)
    output_OC = jnp.sum(jnp.array(P_l(cos_theta_OC, list_l=list_l)) * ratios_OC, axis=-1)
    output_OD = jnp.sum(jnp.array(P_l(cos_theta_OD, list_l=list_l)) * ratios_OD, axis=-1)
    #jax.debug.print("output_OA:{}", output_OA)
    #jax.debug.print("output_OA_shape:{}", output_OA.shape)
    #jax.debug.print("nonlocal_parameters:{}", nonlocal_parameters)
    #jax.debug.print("nonlocal_parameters_shape:{}", nonlocal_parameters.shape)
    "now, the problem is the mismatch between two arrays. 6.12.2024."

    def multiply_test(a: jnp.array, b: jnp.array):
        return a * b

    """here, 4 is the number of points."""
    multiply_test_parallel = jax.vmap(multiply_test, in_axes=(0, 4), out_axes=0)
    OA_energy = multiply_test_parallel(output_OA, nonlocal_parameters)
    OB_energy = multiply_test_parallel(output_OB, nonlocal_parameters)
    OC_energy = multiply_test_parallel(output_OC, nonlocal_parameters)
    OD_energy = multiply_test_parallel(output_OD, nonlocal_parameters)
    """dimension of output: angular momentum, 1, batch_size, nelectrons, natoms"""
    #jax.debug.print("OA_energy:{}", OA_energy.shape)
    #jax.debug.print("OB_energy:{}", OB_energy.shape)
    #jax.debug.print("OC_energy:{}", OC_energy.shape)
    #jax.debug.print("OD_energy:{}", OD_energy.shape)
    nonlocal_energy = jnp.sum(jnp.sum(jnp.sum(OA_energy + OB_energy + OC_energy + OD_energy, axis=0), axis=-1), axis=-1)
    #jax.debug.print("nonlocal_energy_shape:{}", nonlocal_energy.shape)
    #jax.debug.print("local_part_energy_shape:{}", local_part_energy.shape)
    total_energy = local_part_energy + nonlocal_energy
    #jax.debug.print("total_energy:{}", total_energy)
    return total_energy, ratios_OA, ratios_OB, ratios_OC, ratios_OD, cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD, roted_configurations_OA, roted_configurations_OB, roted_configurations_OC, roted_configurations_OD, weights





'''
total_energy, ratios_OA, ratios_OB, ratios_OC, ratios_OD, cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD = \
    total_energy_pseudopotential(data=data, params=params,
                                 rn_local_general=Rn_local,
                                 rn_non_local_general=Rn_non_local,
                                 local_coefficient_general=Local_coes,
                                 nonlocal_coefficient_general=Non_local_coes,
                                 local_exponent_general=Local_exps,
                                 nonlocal_exponent_general=Non_local_exps, nelectrons=16, natoms=3, list_l=2, batch_size=4)
'''