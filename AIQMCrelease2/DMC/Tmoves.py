"""This module proposes T-moves for walkers."""
import jax
import chex
import jax.numpy as jnp
from AIQMCrelease2.wavefunction import nn
from AIQMCrelease2.pseudopotential import pseudopotential
from AIQMCrelease2.pseudopotential.pseudopotential import get_non_v_l, get_P_l




def P_l_theta(x: jnp.array, list_l: float):
    """
    create the legendre polynomials functions
    :param x: cos(theta)
    :return: the largest l in the pp file.
    """
    if list_l == 0:
        return 1/(4 * jnp.pi) * jnp.ones(x.shape)
    if list_l == 1:
        return 1/(4 * jnp.pi) * jnp.ones(x.shape), \
               (2 + 1)/(4 * jnp.pi) * x
    if list_l == 2:
        return 1/(4 * jnp.pi) * jnp.ones(x.shape), \
               (2 + 1)/(4 * jnp.pi) * x, \
               (2 * 2 + 1)/(4 * jnp.pi) * 0.5 * (3 * x * x - 1)
    if list_l == 3:
        return 1/(4 * jnp.pi) * jnp.ones(x.shape), \
               (2 + 1)/(4 * jnp.pi) * x, \
               (2 * 2 + 1)/(4 * jnp.pi) * 0.5 * (3 * x * x - 1), \
               (3 * 2 + 1)/(4 * jnp.pi) * 0.5 * (5 * x * x * x - 3 * x)


def compute_tmoves(list_l: float,
                   tstep: float,
                   nelectrons: int,
                   natoms: int,
                   ndim: int,
                   lognetwork,
                   Rn_non_local: jnp.array,
                   Non_local_coes: jnp.array,
                   Non_local_exps: jnp.array
                   ):
    get_P_l = pseudopotential.get_P_l(nelectrons=nelectrons,
                                      natoms=natoms,
                                      ndim=ndim,
                                      log_network_inner=lognetwork)
    get_P_l_parallel = jax.vmap(get_P_l, in_axes=(None, None, 0, None))
    get_non_local_coe = pseudopotential.get_non_v_l(ndim=ndim,
                                                    nelectrons=nelectrons,
                                                    natoms=natoms,
                                                    rn_non_local=Rn_non_local,
                                                    non_local_coefficient=Non_local_coes,
                                                    non_local_exponent=Non_local_exps)

    def multiply_weights(v_r_non_local: jnp.array, output_P_l: jnp.array):
        return (jnp.exp(-1 * tstep * v_r_non_local) - 1) * output_P_l

    multiply_weights_parallel = jax.vmap( jax.vmap(jax.vmap( jax.vmap(multiply_weights, in_axes=(0, 0)), in_axes=(0, 0), out_axes=0),
                                                   in_axes=(None, 0), out_axes=0),
                                          in_axes=(2, 0), out_axes=0)

    def run_t_amplitudes(ratios: jnp.array, weights: jnp.array):
        return ratios * weights

    run_t_amplitudes_parallel = jax.vmap(run_t_amplitudes, in_axes=(0, 0), out_axes=0)


    """For a given electron, evaluate all possible t-moves."""
    def calculate_ratio_weight_tmoves(data: nn.AINetData, params: nn.ParamTree, key: chex.PRNGKey):
        """I forgot how to write this part....7.2.2025."""
        Points_OA, Points_OB, Points_OC, Points_OD, weights = pseudopotential.get_rot(batch_size=1,
                                                                                      key=key)
        #jax.debug.print("Points_OA:{}", Points_OA)
        #jax.debug.print("data:{}", data)
        cos_theta_OA, ratios_OA, roted_configurations_OA, weights_OA, roted_coords_OA = get_P_l_parallel(
            data, params, Points_OA, weights[0])
        cos_theta_OB, ratios_OB, roted_configurations_OB, weights_OB, roted_coords_OB = get_P_l_parallel(
            data, params, Points_OB, weights[1])
        cos_theta_OC, ratios_OC, roted_configurations_OC, weights_OC, roted_coords_OC = get_P_l_parallel(
            data, params, Points_OC, weights[2])
        cos_theta_OD, ratios_OD, roted_configurations_OD, weights_OD, roted_coords_OD = get_P_l_parallel(
            data, params, Points_OD, weights[3])
        #jax.debug.print("cos_theta_OA:{}", cos_theta_OA)

        def calculate_forward(cos_theta: jnp.array, ratios: jnp.array):
            output_P_l = jnp.array(P_l_theta(cos_theta, list_l=list_l))
            v_r_non_local = get_non_local_coe(data)
            """the shape of v_r_non_local is number of electrons, number of atoms, number of l orbitals. correct. 8.2.2025
            the shape of output_P_l_OA is the number of l orbitals, 1, the number of electrons, the number of atoms, the number of points."""
            weights = multiply_weights_parallel(v_r_non_local, output_P_l)
            """the shape of weights is number of l orbitals, number of electrons, number of atoms, number of points.
            The l orbitals dimension here is shrunk."""
            weights = jnp.sum(weights, axis=0)
            """do summation along the l orbitals dimension"""
            #jax.debug.print("weights:{}", weights.shape)
            #jax.debug.print("ratios:{}", ratios.shape)
            t_amplitudes = run_t_amplitudes_parallel(ratios, weights)
            forward_probability = jnp.zeros_like(t_amplitudes)
            """we need propose the move to the new configuration here. to be continued...10.12.2024."""
            forward_probability_output = jnp.where(t_amplitudes > forward_probability, t_amplitudes, forward_probability)
            #jax.debug.print("forward_probability_output:{}", forward_probability_output)
            return forward_probability_output

        forward_probability_output_OA = calculate_forward(cos_theta_OA, ratios_OA)
        forward_probability_output_OB = calculate_forward(cos_theta_OB, ratios_OB)
        forward_probability_output_OC = calculate_forward(cos_theta_OC, ratios_OC)
        forward_probability_output_OD = calculate_forward(cos_theta_OD, ratios_OD)

        norm = 1 + jnp.sum(weights_OA * forward_probability_output_OA) + jnp.sum(
            weights_OB * forward_probability_output_OB) + \
               jnp.sum(weights_OC * forward_probability_output_OC) + jnp.sum(weights_OD * forward_probability_output_OD)

        #jax.debug.print("forward_OA:{}", forward_probability_output_OA)
        #jax.debug.print("forward_OB:{}", forward_probability_output_OB)
        #jax.debug.print("forward_OC:{}", forward_probability_output_OC)
        #jax.debug.print("forward_OD:{}", forward_probability_output_OD)
        #jax.debug.print("forward_OA:{}", forward_probability_output_OA.shape)
        """to be continued... 8.2.2025."""
        forward_probability_output_OA = jnp.transpose(forward_probability_output_OA, (3, 0, 1, 2))
        #jax.debug.print("forward_OA:{}", forward_probability_output_OA.shape)
        forward_probability_output_OB = jnp.transpose(forward_probability_output_OB, (3, 0, 1, 2))
        forward_probability_output_OC = jnp.transpose(forward_probability_output_OC, (3, 0, 1, 2))
        forward_probability_output_OD = jnp.transpose(forward_probability_output_OD, (3, 0, 1, 2))
        forward_probability_output_total = jnp.concatenate([forward_probability_output_OA,
                                                            forward_probability_output_OB,
                                                            forward_probability_output_OC,
                                                            forward_probability_output_OD], axis=0)
        #jax.debug.print("forward_total:{}", forward_probability_output_total)
        #jax.debug.print("forward_total_shape:{}", forward_probability_output_total.shape)

        forward_probability_output_total_final = jnp.transpose(forward_probability_output_total, (1, 2, 3, 0))

        forward_probability_output_total_final = jnp.reshape(forward_probability_output_total_final, (1, nelectrons, -1))

        forward_probability_output_total_final = jnp.concatenate([jnp.ones((1, nelectrons, 1)), forward_probability_output_total_final], axis=-1)
        jax.debug.print("forward_probability_output_total_final_shape:{}", forward_probability_output_total_final.shape)
        cdf = jnp.cumsum(forward_probability_output_total_final / norm, axis=-1)
        #ax.debug.print("cdf:{}", cdf)
        jax.debug.print("cdf_shape:{}", cdf.shape)

        def select_walker(a: jnp.array):
            r = jax.random.uniform(key) + 1
            return jnp.searchsorted(a, r)

        selected_moves = jnp.apply_along_axis(func1d=select_walker, axis=-1, arr=cdf)
        #jax.debug.print("forward_probability_output_total_final:{}", forward_probability_output_total_final)
        move_selected = jnp.zeros_like(selected_moves)
        #jax.debug.print("selected_moves:{}", selected_moves)
        #jax.debug.print("move_selected:{}", move_selected)
        move_selected_final = jnp.where(selected_moves < forward_probability_output_total_final.shape[2], selected_moves, move_selected)
        jax.debug.print("move_selected_final:{}", move_selected_final)

        order = jnp.arange(0, nelectrons, step=1)
        pos_temp = data.positions
        pos_temp = jnp.reshape(pos_temp, (-1, 3))
        jax.debug.print("roted_coords_OA:{}", roted_coords_OA.shape)
        jax.debug.print("roted_coords_OB:{}", roted_coords_OB.shape)
        #jax.debug.print("roted_coords_OA:{}", roted_configurations_OA)

        roted_coords_total = jnp.concatenate([roted_coords_OA, roted_coords_OB, roted_coords_OC, roted_coords_OD],
                                             axis=3)
        jax.debug.print("roted_coords_total_shape:{}", roted_coords_total.shape)
        roted_coords_total = jnp.reshape(roted_coords_total, (nelectrons, -1, ndim))
        jax.debug.print("roted_coords_total_shape:{}", roted_coords_total.shape)

        ratios_total = jnp.concatenate([ratios_OA, ratios_OB, ratios_OC, ratios_OD], axis=-1)
        ratios_total = jnp.reshape(ratios_total, (nelectrons, -1))
        """3 is the dimension of the electrons."""
        pos_temp = jnp.reshape(pos_temp, (nelectrons, 1, ndim))
        total_configuration = jnp.concatenate([pos_temp, roted_coords_total], axis=1)
        ratio_ones = jnp.ones((nelectrons, 1))
        ratio_total_final = jnp.concatenate([ratio_ones, ratios_total], axis=1)
        jax.debug.print("move_selected_final:{}", move_selected_final)
        jax.debug.print("order:{}", order)
        jax.debug.print("ratio_total_final:{}", ratio_total_final.shape)
        jax.debug.print("roted_coords_total:{}", total_configuration.shape)
        jax.debug.print("forward_probability_output_total_final:{}", forward_probability_output_total_final.shape)

        def selected_configurations(move_selected_final: jnp.array, order: jnp.array,
                                   ratio_total: jnp.array, roted_coords_total: jnp.array, t_amp: jnp.array):
            temp = roted_coords_total[order]
            temp_ratio = ratio_total[order]
            reverse_ratio = 1 / temp_ratio[move_selected_final]
            back_amplitudes = t_amp[move_selected_final] * reverse_ratio
            return temp[move_selected_final], back_amplitudes

        selcted_configurations_parallel = jax.vmap(selected_configurations, in_axes=(0, 0, None, None, None))
        """to keep the same shape"""
        move_selected_final = jnp.reshape(move_selected_final, -1)
        jax.debug.print("move_selected_final:{}", move_selected_final)
        forward_probability_output_total_final = jnp.reshape(forward_probability_output_total_final, (nelectrons, -1))
        new_configuration, back_amplitudes = selcted_configurations_parallel(move_selected_final, order,
                                                                             ratio_total_final, total_configuration,
                                                                             forward_probability_output_total_final)
        jax.debug.print("new_configuration:{}", new_configuration)
        jax.debug.print("new_configuration:{}", new_configuration.shape)
        #jax.debug.print("back_amplitudes:{}", back_amplitudes)
        no_move_weights = jnp.array([[0.0]])
        weights_final = jnp.concatenate([no_move_weights, weights])
        jax.debug.print("weights_final:{}", weights_final)
        jax.debug.print("back_amplitudes:{}", back_amplitudes.shape)
        back_amplitudes_norm = 1.0 + jnp.sum(weights_final[0] * back_amplitudes[:, 0], axis=-1) + \
                               jnp.sum(weights_final[1] * back_amplitudes[:, 1:19], axis=-1) + \
                               jnp.sum(weights_final[2] * back_amplitudes[:, 19:55], axis=-1) + \
                               jnp.sum(weights_final[3] * back_amplitudes[:, 55:79], axis=-1) + \
                               jnp.sum(weights_final[4] * back_amplitudes[:, 79: 151], axis=-1)

        acceptance = norm / back_amplitudes_norm
        acceptance = jnp.reshape(acceptance.real, (-1, 1))
        jax.debug.print("acceptance:{}", acceptance.shape)
        jax.debug.print("acceptance:{}", acceptance)
        key, subkey = jax.random.split(key)
        rnd = jax.random.uniform(subkey, shape=acceptance.shape, minval=0, maxval=1.0)
        cond = acceptance > rnd
        x1 = data.positions
        x1 = jnp.reshape(x1, (-1, 3))
        final_configuration = jnp.where(cond, new_configuration, x1)
        jax.debug.print("final_configuration:{}", final_configuration)
        return final_configuration, acceptance
    return calculate_ratio_weight_tmoves


#ratio_weight = compute_tmoves(list_l=2, tstep=0.1)

'''
generate_points_information_test = pseudopotential.get_P_l(nelectrons=nelectrons,
                                                               natoms=natoms,
                                                               ndim=ndim,
                                                               log_network_inner=lognetwork)
def propose_t_moves(params: nn.ParamTree,
                    data: nn.AINetData,
                    lognetwork,
                    nelectrons: int,
                    natoms: int,
                    ndim: int,
                    Rn_non_local: jnp.array,
                    Non_local_coes: jnp.array,
                    Non_local_exps: jnp.array,
                    ratios_OA: jnp.array, ratios_OB: jnp.array, ratios_OC: jnp.array, ratios_OD: jnp.array,
                    cos_theta_OA: jnp.array, cos_theta_OB: jnp.array, cos_theta_OC: jnp.array, cos_theta_OD: jnp.array,
                    roted_coords_OA: jnp.array, roted_coords_OB: jnp.array, roted_coords_OC: jnp.array, roted_coords_OD: jnp.array,
                    weigths: jnp.array, key: chex.PRNGKey
                    ):

    def t_moves():

    nelectrons = 16
    forward_probability_output_OA = ratio_weight(params, data, cos_theta_OA, ratios_OA, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)
    forward_probability_output_OB = ratio_weight(params, data, cos_theta_OB, ratios_OB, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)
    forward_probability_output_OC = ratio_weight(params, data, cos_theta_OC, ratios_OC, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)
    forward_probability_output_OD = ratio_weight(params, data, cos_theta_OD, ratios_OD, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)


    norm = 1 + jnp.sum(weigths[0] * forward_probability_output_OA) + jnp.sum(weigths[1] * forward_probability_output_OB) + \
           jnp.sum(weigths[2] * forward_probability_output_OC) + jnp.sum(weigths[3] * forward_probability_output_OD)

    forward_probability_output_OA = jnp.transpose(forward_probability_output_OA, (1, 2, 0))
    forward_probability_output_OB = jnp.transpose(forward_probability_output_OB, (1, 2, 0))
    forward_probability_output_OC = jnp.transpose(forward_probability_output_OC, (1, 2, 0))
    forward_probability_output_OD = jnp.transpose(forward_probability_output_OD, (1, 2, 0))

    forward_probability_output_total = jnp.concatenate([forward_probability_output_OA,
                                                       forward_probability_output_OB,
                                                       forward_probability_output_OC,
                                                      forward_probability_output_OD], axis=2)

    forward_probability_output_total = jnp.reshape(forward_probability_output_total, (nelectrons, -1))
    forward_probability_output_total_final = jnp.concatenate([jnp.ones((nelectrons, 1)), forward_probability_output_total], axis=1)

    def select_walker(a: jnp.array):
        r = jax.random.uniform(key) + 1
        return jnp.searchsorted(a, r)

    cdf = jnp.cumsum(forward_probability_output_total_final / norm, axis=1)
    selected_moves = jnp.apply_along_axis(func1d=select_walker, axis=-1, arr=cdf)
    move_selected = jnp.zeros_like(selected_moves)
    move_selected_final = jnp.where(selected_moves < forward_probability_output_total_final.shape[1], selected_moves, move_selected)
    order = jnp.arange(0, nelectrons, step=1)
    pos_temp = data.positions
    pos_temp = jnp.reshape(pos_temp, (-1, 3))
    roted_coords_total = jnp.concatenate([roted_coords_OA, roted_coords_OB, roted_coords_OC, roted_coords_OD], axis=2)
    roted_coords_total = jnp.reshape(roted_coords_total, (nelectrons, -1, 3))
    ratios_total = jnp.concatenate([ratios_OA, ratios_OB, ratios_OC, ratios_OD], axis=-1)
    ratios_total = jnp.reshape(ratios_total, (nelectrons, -1))
    """3 is the dimension of the electrons."""
    pos_temp = jnp.reshape(pos_temp, (nelectrons, 1, 3))
    total_configuration = jnp.concatenate([pos_temp, roted_coords_total], axis=1)
    ratio_ones = jnp.ones((nelectrons, 1))
    ratio_total_final = jnp.concatenate([ratio_ones, ratios_total], axis=1)

    def selcted_configurations(move_selected_final: jnp.array, order: jnp.array,
                               ratio_total: jnp.array, roted_coords_total: jnp.array, t_amp: jnp.array):
        temp = roted_coords_total[order]
        temp_ratio = ratio_total[order]
        reverse_ratio = 1 / temp_ratio[move_selected_final]
        back_amplitudes = t_amp[move_selected_final] * reverse_ratio
        return temp[move_selected_final], back_amplitudes

    selcted_configurations_parallel = jax.vmap(selcted_configurations, in_axes=(0, 0, None, None, None))
    new_configuration, back_amplitudes = selcted_configurations_parallel(move_selected_final, order, ratio_total_final, total_configuration, forward_probability_output_total_final)
    no_move_weights = jnp.array([[0.0]])
    weights_final = jnp.concatenate([no_move_weights, weights])
    back_amplitudes_norm = 1.0 + jnp.sum(weights_final[0] * back_amplitudes[:, 0], axis=-1) + \
                            jnp.sum(weights_final[1] * back_amplitudes[:, 1:19], axis=-1) + \
                            jnp.sum(weights_final[2] * back_amplitudes[:, 19:55], axis=-1) + \
                            jnp.sum(weights_final[3] * back_amplitudes[:, 55:79], axis=-1) + \
                            jnp.sum(weights_final[4] * back_amplitudes[:, 79: 151], axis=-1)

    acceptance = norm / back_amplitudes_norm
    acceptance = jnp.reshape(acceptance.real, (-1, 1))
    rnd = jax.random.uniform(subkeys, shape=acceptance.shape, minval=0, maxval=1.0)
    cond = acceptance > rnd
    x1 = data.positions
    x1 = jnp.reshape(x1, (-1, 3))
    final_configuration = jnp.where(cond, new_configuration, x1)
    return final_configuration, acceptance
'''