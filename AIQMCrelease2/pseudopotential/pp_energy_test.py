import chex
import jax
import jax.numpy as jnp
from AIQMCrelease2.pseudopotential import pseudopotential
from AIQMCrelease2.wavefunction import nn
#from AIQMCrelease1.main import main_adam
'''
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
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
'''

def total_energy_pseudopotential(get_local_pp_energy: pseudopotential.LocalPPEnergy,
                                 get_nonlocal_pp_coes: pseudopotential.NonlocalPPcoes,
                                 get_P_l,
                                 list_l: int):
    """This function caluclates the energy of pseudopotential.
    For the pp of C and O, only l=0 contributes to the nonlocal part.
    we have more problems here. If all atoms have the same shape of the pp parameters, it is ok.
    But if one of the atoms has higher angular momentum functions, I don't know how to do it efficiently.
    we need debug this function carefully. 7.1.2025."""

    def multiply_test(a: jnp.array, b: jnp.array):
        return a * b

    """here, 4 is the number of points."""
    #multiply_test_parallel = jax.vmap(multiply_test, in_axes=(0, 1), out_axes=0)
    multiply_test_parallel = jax.vmap(
        jax.vmap(multiply_test, in_axes=(0, None), out_axes=0),
        in_axes=(0, 2), out_axes=0)
    get_P_l_parallel = jax.vmap(get_P_l, in_axes=(None, None, 0, None))

    def multiply_P_l_ratios(a: jnp.array, b: jnp.array):
        return a * b

    multiply_P_l_ratios_parallel = jax.vmap(multiply_P_l_ratios, in_axes=(0, None))

    def get_total_pp_energy(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData,):
        local_pp_energy = get_local_pp_energy(data)
        local_pp_energy = jnp.sum(jnp.sum(local_pp_energy, axis=-1), axis=-1)
        nonlocal_parameters = get_nonlocal_pp_coes(data)
        """we have to set batch_size=1 here."""
        Points_OA, Points_OB, Points_OC, Points_OD, weights = pseudopotential.get_rot(batch_size=1, key=key)
        cos_theta_OA, ratios_OA, roted_configurations_OA, weights_OA, roted_coords_OA = get_P_l_parallel(
            data, params, Points_OA, weights[0])
        cos_theta_OB, ratios_OB, roted_configurations_OB, weights_OB, roted_coords_OB = get_P_l_parallel(
            data, params, Points_OB, weights[1])
        cos_theta_OC, ratios_OC, roted_configurations_OC, weights_OC, roted_coords_OC = get_P_l_parallel(
            data, params, Points_OC, weights[2])
        cos_theta_OD, ratios_OD, roted_configurations_OD, weights_OD, roted_coords_OD = get_P_l_parallel(
            data, params, Points_OD, weights[3])

        output_OA = jnp.sum(multiply_P_l_ratios_parallel(jnp.array(pseudopotential.P_l(cos_theta_OA, list_l=list_l)), ratios_OA), axis=-1)
        output_OB = jnp.sum(multiply_P_l_ratios_parallel(jnp.array(pseudopotential.P_l(cos_theta_OB, list_l=list_l)), ratios_OB), axis=-1)
        output_OC = jnp.sum(multiply_P_l_ratios_parallel(jnp.array(pseudopotential.P_l(cos_theta_OC, list_l=list_l)), ratios_OC), axis=-1)
        output_OD = jnp.sum(multiply_P_l_ratios_parallel(jnp.array(pseudopotential.P_l(cos_theta_OD, list_l=list_l)), ratios_OD), axis=-1)
        """output_OA shape should be l orbitals, 1, nelectrons, natoms."""
        """nonlocal_parameters shape should be nelectrons, natoms, l orbitals."""
        #jax.debug.print("output_OA_shape:{}", output_OA.shape)
        OA_energy = multiply_test_parallel(output_OA, nonlocal_parameters)
        #jax.debug.print("OA_energy:{}", OA_energy.shape)
        OB_energy = multiply_test_parallel(output_OB, nonlocal_parameters)
        OC_energy = multiply_test_parallel(output_OC, nonlocal_parameters)
        OD_energy = multiply_test_parallel(output_OD, nonlocal_parameters)
        nonlocal_energy = jnp.sum(jnp.sum(OA_energy + OB_energy + OC_energy + OD_energy, axis=0))
        #jax.debug.print("nonlocal_energy:{}", nonlocal_energy)
        total_energy = local_pp_energy + nonlocal_energy
        #jax.debug.print("total_energy:{}", total_energy)
        return total_energy

    return get_total_pp_energy

'''
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

generate_points_information_test = pseudopotential.get_P_l(nelectrons=16,
                                                           natoms=3,
                                                           ndim=3,
                                                           log_network_inner=lognetwork)


key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
#Points_OA, Points_OB, Points_OC, Points_OD, weights = pseudopotential.get_rot(batch_size=4, key=subkey)
#Points_OA = Points_OA[None, ...]
#Points_OB = Points_OB[None, ...]
#Points_OC = Points_OC[None, ...]
#Points_OD = Points_OD[None, ...]
#jax.debug.print("Points_OA_shape:{}", Points_OA.shape)
total_energy_function_test = total_energy_pseudopotential(get_local_pp_energy=get_local_part_energy_test,
                                                          get_nonlocal_pp_coes=get_non_local_coe_test,
                                                          get_P_l=generate_points_information_test,
                                                          log_network=lognetwork,
                                                          nelectrons=16,
                                                          natoms=3,
                                                          ndim=3,
                                                           list_l=2,
                                                          batch_size=4,
                                                          key=key)

total_energy_function_test_parallel = jax.pmap(jax.vmap(total_energy_function_test,
                                                        in_axes=(
                                                            nn.AINetData(positions=0, atoms=0, charges=0),
                                                            None,)))

output_OA, nonlocal_parameters = total_energy_function_test_parallel(data, batch_params)
'''