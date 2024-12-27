import jax
import jax.numpy as jnp
from AIQMCrelease1.pseudopotential import pseudopotential
from AIQMCrelease1.wavefunction import nn
from AIQMCrelease1.main import main_adam

structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['C', 'O', 'O']
atoms = jnp.array([[1.33, 1.0, 1.0], [0.0, 1.0, 1.0], [2.66, 1.0, 1.0]])
charges = jnp.array([4.0, 6.0, 6.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
signed_network, data, batch_params, lognetwork = main_adam.main(atoms=atoms,
                                                                charges=charges,
                                                                spins=spins,
                                                                nelectrons=16,
                                                                natoms=3,
                                                                ndim=3,
                                                                batch_size=4,
                                                                iterations=1,
                                                                structure=structure)
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


def total_energy_pseudopotential(get_local_pp_energy: pseudopotential.LocalPPEnergy,
                                 get_nonlocal_pp_coes: pseudopotential.NonlocalPPcoes,
                                 nelectrons: int,
                                 natoms: int,
                                 ndim: int,
                                 list_l: int,
                                 batch_size: int):
    """This function caluclates the energy of pseudopotential.
    For the pp of C and O, only l=0 contributes to the nonlocal part.
    we have more problems here. If all atoms have the same shape of the pp parameters, it is ok.
    But if one of the atoms has higher angular momentum functions, I don't know how to do it efficiently."""

    def get_total_pp_energy(data: nn.AINetData):
        local_pp_energy = get_local_pp_energy(data)
        local_pp_energy = jnp.sum(jnp.sum(local_pp_energy, axis=-1), axis=-1)
        jax.debug.print("local_pp_energy:{}", local_pp_energy)
        nonlocal_parameters = get_nonlocal_pp_coes(data)
        jax.debug.print("nonlocal_paras:{}", nonlocal_parameters)
        return local_pp_energy

    return get_total_pp_energy


get_local_part_energy_test = pseudopotential.local_pp_energy(nelectrons=16,
                                                             natoms=3,
                                                             ndim=3,
                                                             rn_local=Rn_local,
                                                             local_coefficient=Local_coes,
                                                             local_exponent=Local_exps)

get_non_local_coe_test = pseudopotential.get_non_v_l(ndim=3,
                                                     nelectrons=16,
                                                     natoms=3,
                                                     rn_non_local=Rn_non_local,
                                                     non_local_coefficient=Non_local_coes,
                                                     non_local_exponent=Non_local_exps)

total_energy_function_test = total_energy_pseudopotential(get_local_pp_energy=get_local_part_energy_test,
                                                          get_nonlocal_pp_coes=get_non_local_coe_test,
                                                          nelectrons=16,
                                                          natoms=3,
                                                          ndim=3,
                                                          list_l=2,
                                                          batch_size=4)
total_energy_function_test_parallel = jax.pmap(jax.vmap(total_energy_function_test))
output = total_energy_function_test_parallel(data)
'''
get_local_part_energy_test = local_energy(nelectrons=16,
                                      natoms=3,
                                      ndim=3,
                                      rn_local=Rn_local,
                                      local_coefficient=Local_coes,
                                      local_exponent=Local_exps)

get_local_part_energy_test_parallel = jax.vmap(get_local_part_energy_test,
                                           in_axes=(nn.AINetData(positions=0, atoms=0, charges=0)), out_axes=0)

get_non_local_coe_test = get_non_v_l(ndim=3,
                                 nelectrons=16,
                                 natoms=3,
                                 rn_non_local=Rn_non_local,
                                 non_local_coefficient=Non_local_coes,
                                 non_local_exponent=Non_local_exps)

get_non_local_coe_test_parallel = jax.vmap(get_non_local_coe_test,
                                       in_axes=(nn.AINetData(positions=0, atoms=0, charges=0)), out_axes=0)


def total_energy_pp(data: nn.AINetData, key: chex.PRNGKey, params: nn.ParamTree):
local_part_energy = get_local_part_energy_test_parallel(data, rn_local_general, local_coefficient_general, local_exponent_general)
local_part_energy = jnp.sum(jnp.sum(local_part_energy, axis=-1), axis=-1)
nonlocal_parameters = get_non_local_coe_test_parallel(data, rn_non_local_general, nonlocal_coefficient_general, nonlocal_exponent_general)
keys = jax.random.PRNGKey(key)
Points_OA, Points_OB, Points_OC, Points_OD, weights = get_rot(batch_size, keys)
cos_theta_OA, ratios_OA, roted_configurations_OA, weights_OA, roted_coords_OA = generate_points_information_test_parallel(data, params, Points_OA, weights[0])
cos_theta_OB, ratios_OB, roted_configurations_OB, weights_OB, roted_coords_OB = generate_points_information_test_parallel(data, params, Points_OB, weights[1])
cos_theta_OC, ratios_OC, roted_configurations_OC, weights_OC, roted_coords_OC = generate_points_information_test_parallel(data, params, Points_OC, weights[2])
cos_theta_OD, ratios_OD, roted_configurations_OD, weights_OD, roted_coords_OD = generate_points_information_test_parallel(data, params, Points_OD, weights[3])
output_OA = jnp.sum(jnp.array(P_l(cos_theta_OA, list_l=list_l)) * ratios_OA, axis=-1)
output_OB = jnp.sum(jnp.array(P_l(cos_theta_OB, list_l=list_l)) * ratios_OB, axis=-1)
output_OC = jnp.sum(jnp.array(P_l(cos_theta_OC, list_l=list_l)) * ratios_OC, axis=-1)
output_OD = jnp.sum(jnp.array(P_l(cos_theta_OD, list_l=list_l)) * ratios_OD, axis=-1)

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
return total_energy, ratios_OA, ratios_OB, ratios_OC, ratios_OD, cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD, roted_configurations_OA, roted_configurations_OB, roted_configurations_OC, roted_configurations_OD, weights, roted_coords_OA, roted_coords_OB, roted_coords_OC, roted_coords_OD
'''




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