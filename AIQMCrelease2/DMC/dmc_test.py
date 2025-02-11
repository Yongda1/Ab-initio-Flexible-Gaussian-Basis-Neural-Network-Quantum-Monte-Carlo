"""today, we start to do some dmc codes. Because of the problem of the parameter tree, we still have
one problem about the parallel calculation. But actually our calculation engine is working well."""
import chex
import jax
import jax.numpy as jnp
from jax import lax
from AIQMCbatch3adm import nn
from AIQMCbatch3adm import main_adam
from AIQMCbatch3adm.utils import utils
from AIQMCbatch3adm import hamiltonian
from AIQMCbatch3adm.loss import make_loss
from AIQMCbatch3adm.pseudopotential.pseudopotential import total_energy_pseudopotential, get_non_v_l
import kfac_jax
import numpy as np

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
                            [[14.832760,       26.349664],       [7.621400,        10.331583],      [0.0,              0.0]],
                            [[85.86406,         0.0],            [0.0,               0.0],          [0.0,              0.0]]])
Non_local_exps = jnp.array([[[2.894473589836, 1.550339816290], [2.986528872039, 1.283381203893], [1.043001142249, 0.554562729807]],
                            [[9.447023,       2.553812],       [3.660001,       1.903653],       [0.0,              0.0]],
                            [[13.65512,        0.0],           [0.0,              0.0],          [0.0,              0.0]]])


signed_network, data, params, log_network = main_adam.main()
key = jax.random.PRNGKey(1)
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
localenergy = hamiltonian.local_energy(f=signed_network, batch_size=4, natoms=3, nelectrons=16)
#localenergyDMC = hamiltonian.local_energy_dmc(f=signed_network)
total_energy_test = make_loss(log_network, local_energy=localenergy)
total_energy_test_pmap = jax.pmap(total_energy_test, in_axes=(0, 0, nn.AINetData(positions=0, atoms=0, charges=0),), out_axes=(0, 0))
loss, aux_data = total_energy_test_pmap(params, subkeys, data)
#jax.debug.print("loss:{}", loss)



total_energy, \
ratios_OA, ratios_OB, ratios_OC, ratios_OD, \
cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD, \
roted_configurations_OA, roted_configurations_OB, roted_configurations_OC, roted_configurations_OD, weights, \
roted_coords_OA, roted_coords_OB, roted_coords_OC, roted_coords_OD = \
    total_energy_pseudopotential(data=data, params=params,
                                 rn_local_general=Rn_local,
                                 rn_non_local_general=Rn_non_local,
                                 local_coefficient_general=Local_coes,
                                 nonlocal_coefficient_general=Non_local_coes,
                                 local_exponent_general=Local_exps,
                                 nonlocal_exponent_general=Non_local_exps, nelectrons=16, natoms=3, list_l=2, batch_size=4)


#jax.debug.print("ratios_OA:{}", ratios_OA)
#jax.debug.print("ratios_OA_shape:{}", ratios_OA.shape)
#jax.debug.print("cos_theta_OA:{}", cos_theta_OA)
#jax.debug.print("cos_theta_OA_shape:{}", cos_theta_OA.shape)
#jax.debug.print("roted_configurations_OA_shape:{}", roted_configurations_OA.shape)
#jax.debug.print("roted_coords_OA:{}", roted_coords_OA)
#jax.debug.print("roted_coords_OA_shape:{}", roted_coords_OA.shape)
#v_r_non_local = get_non_v_l_parallel(data, Rn_non_local, Non_local_coes, Non_local_exps)
#jax.debug.print("v_r_non_local:{}", v_r_non_local)
"""the last dimension is angular momentum function"""
#jax.debug.print("v_r_non_local_shape:{}", v_r_non_local.shape)


def P_l_theta(x: jnp.array, list_l: float):
    """
    create the legendre polynomials functions
    :param x: cos(theta)
    :param list_l: the angular momentum functions used in the calculation. For example, list_l = [1, 1, 1, 0, ] means s, p, d, no f
    :return:
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


def compute_tmoves(list_l: float, tstep: float):
    """For a given electron, evaluate all possible t-moves.
    Here, we need read the paper about T-moves.
    The implementation of T-moves is from the paper,
    'Nonlocal pseudopotentials and time-step errors in diffusion Monte Carlo' written by Anderson and Umrigar.
    We finished the pseudopotential part currently, now turn to the T-moves. 9.12.2024.
    we use the same strategy for this function. The input is just one configuration.
    """
    def calculate_ratio_weight_tmoves(data: nn.AINetData,
                                      costheta: jnp.array,
                                      ratios: jnp.array,
                                      Rn_non_local: jnp.array,
                                      Non_local_coes: jnp.array,
                                      Non_local_exps: jnp.array):

        output_P_l = jnp.array(P_l_theta(costheta, list_l=list_l))
        v_r_non_local = get_non_v_l(data, Rn_non_local, Non_local_coes, Non_local_exps)
        """we have some problems about the data type. It should be float32 but currently it is int32. Maybe it is a bug. We solve it later."""
        #jax.debug.print("v_r_non_local_shape:{}", v_r_non_local.shape)

        def multiply_weights(v_r_non_local: jnp.array, output_P_l: jnp.array):
            return (jnp.exp(-1 * tstep * v_r_non_local) - 1) * output_P_l

        multiply_weights_parallel = jax.vmap(jax.vmap(multiply_weights, in_axes=(2, 0), out_axes=0), in_axes=(None, 3), out_axes=0)
        """the shape of weights should be number of points, angular momentum functions, number of electrons, number of atoms"""
        weights = multiply_weights_parallel(v_r_non_local, output_P_l)
        weights = jnp.sum(weights, axis=1)

        def run_t_amplitudes(ratios: jnp.array, weights: jnp.array):
            return ratios * weights

        run_t_amplitudes_parallel = jax.vmap(run_t_amplitudes, in_axes=(2, 0), out_axes=0)
        t_amplitudes = run_t_amplitudes_parallel(ratios, weights)
        forward_probability = jnp.zeros_like(t_amplitudes)
        """we need propose the move to the new configuration here. to be continued...10.12.2024."""
        forward_probability_output = jnp.where(t_amplitudes > forward_probability, t_amplitudes, forward_probability)
        return forward_probability_output
    return calculate_ratio_weight_tmoves

ratio_weight = compute_tmoves(list_l=2, tstep=0.1)



def propose_t_moves(params: nn.ParamTree,
                    data: nn.AINetData,
                    Rn_non_local: jnp.array,
                    Non_local_coes: jnp.array,
                    Non_local_exps: jnp.array,
                    ratios_OA: jnp.array, ratios_OB: jnp.array, ratios_OC: jnp.array, ratios_OD: jnp.array,
                    cos_theta_OA: jnp.array, cos_theta_OB: jnp.array, cos_theta_OC: jnp.array, cos_theta_OD: jnp.array,
                    roted_coords_OA: jnp.array, roted_coords_OB: jnp.array, roted_coords_OC: jnp.array, roted_coords_OD: jnp.array,
                    weigths: jnp.array, key: chex.PRNGKey
                    ):
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


batch_lognetwork = jax.pmap(jax.vmap(log_network, in_axes=(None, 0, 0, 0), out_axes=0))


def comput_S(e_trial: float, e_est: float, branchcut: float, v2: jnp.array, tau: float, eloc: jnp.array, nelec: int):
    """here, we calculate the S. 24.11.2024."""
    v2 = jnp.sum(v2, axis=-1)
    eloc = jnp.real(eloc)
    e_est = jnp.real(e_est)
    e_trial = jnp.real(e_trial)
    e_cut = e_est-eloc
    e_cut = jnp.min(jnp.array([jnp.abs(e_cut[0]), branchcut]))*jnp.sign(e_cut)
    denominator = 1 + (v2 * tau/nelec) ** 2
    return e_trial - e_est + e_cut/denominator


def walkers_accept(x1, x2, ratio, key, nelectrons: int):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=ratio.shape, minval=0, maxval=1.0)
    cond = ratio > rnd
    #jax.debug.print("cond:{}", cond)
    cond = jnp.reshape(cond, (nelectrons, 1))
    x_new = jnp.where(cond, x2, x1)
    tdamp = jnp.sum(x_new) / jnp.sum(x2)
    return x_new, jnp.abs(tdamp)


def limdrift(g, tau, acyrus):
    v2 = jnp.sum(g**2)
    taueff = (jnp.sqrt(1 + 2 * tau * acyrus * v2) - 1)/ (acyrus * v2)
    return g * taueff


def propose_drift_diffusion(lognetwork: nn.LogAINetLike, tstep: float, nelectrons: int, dim: int, local_energy: hamiltonian.LocalEnergy):
    """this is just one step move."""
    logabs_f = utils.select_output(lognetwork, 1)

    def calculate_drift_diffusion(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData,
                                  etrial: float, e_est: float, branchcut_start: float, weights: float):
        x1 = data.positions
        grad_value = jax.value_and_grad(logabs_f, argnums=1)

        def grad_f_closure(x):
            return grad_value(params, x, data.atoms, data.charges)


        value, grad = grad_f_closure(x1)
        grad_eff = limdrift(jnp.real(grad), tstep, 0.25)

        grad_old_eff_s = grad_eff
        initial_configuration = jnp.reshape(x1, (nelectrons, dim))
        x1 = jnp.reshape(x1, (-1, dim))
        x1 = jnp.reshape(x1, (1, -1))
        x1 = jnp.repeat(x1, nelectrons, axis=0)
        x1 = jnp.reshape(x1, (nelectrons, nelectrons, dim))
        gauss = jax.random.normal(key=key, shape=(jnp.shape(grad_eff)))
        g = grad_eff + gauss
        g = jnp.reshape(g, (nelectrons, dim))
        order = jnp.arange(0, nelectrons, step=1)
        
        def change_configurations(order: jnp.array, g: jnp.array):
            z = jnp.zeros((nelectrons, dim))
            temp = z.at[order].add(g[order])
            return temp

        change_configurations_parallel = jax.vmap(change_configurations, in_axes=(0, None), out_axes=0)
        z = change_configurations_parallel(order, g)
        x2 = x1 + z
        changed_configuration = g + initial_configuration
        x2 = jnp.reshape(x2, (nelectrons, -1))
        value_new, grad_new = jax.vmap(grad_f_closure, in_axes=0, out_axes=0)(x2)
        grad_new_eff = limdrift(grad_new, tstep, 0.25)
        forward = gauss**2
        grad_eff = jnp.repeat(jnp.reshape(jnp.reshape(grad_eff, (-1, dim)), (1, -1)), nelectrons, axis=0)
        backward = (gauss + grad_eff + grad_new_eff)**2
        t_probability = jnp.exp(1/(2 * tstep) * (forward - backward))
        t_probability = jnp.reshape(t_probability, (nelectrons, nelectrons, dim))
        t_probability = jnp.sum(t_probability, axis=-1)
        t_probability = jnp.diagonal(t_probability)
        """now, we need calculate the wavefunction ratio. 22.11.2024."""
        x1 = jnp.reshape(x1, (nelectrons, -1))
        logabs_f_vmap = jax.vmap(logabs_f, in_axes=(None, 0, None, None,))
        wave_x2 = logabs_f_vmap(params, x2, data.atoms, data.charges)
        wave_x1 = logabs_f_vmap(params, x1, data.atoms, data.charges)
        ratios = jnp.abs(wave_x2/wave_x1) ** 2 * t_probability
        """to be continued...22.11.2024. we need do the acceptance judgement later."""
        ratios = ratios * jnp.sign(wave_x2/wave_x1)
        final_configuration, tdamp = walkers_accept(initial_configuration, changed_configuration, ratios, key, nelectrons)
        final_configuration = jnp.reshape(final_configuration, (-1))
        new_data = nn.AINetData(**(dict(data) | {'positions': final_configuration}))
        value_new_s, grad_new_s = grad_f_closure(new_data.positions)
        grad_new_eff_s = limdrift(grad_new_s, tstep, 0.25)
        eloc_old, e_l_mat_old = local_energy(params, key, data)
        eloc_new, e_l_mat_old = local_energy(params, key, new_data)
        S_old = comput_S(e_trial=etrial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_old_eff_s), tau=tstep,eloc=eloc_old, nelec=nelectrons)
        S_new = comput_S(e_trial=etrial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_new_eff_s), tau=tstep,eloc=eloc_new, nelec=nelectrons)
        #jax.debug.print("S_old:{}", S_old)
        #jax.debug.print("S_new:{}", S_new)
        wmult = jnp.exp(tstep * tdamp * (0.5 * S_new + 0.5 * S_old))
        weights *= wmult
        #jax.debug.print("weights:{}", weights)

        return None
    return calculate_drift_diffusion


propose_t_moves_parallel = jax.vmap(propose_t_moves,
                                        in_axes=(
                                        None, nn.AINetData(positions=0, atoms=0, charges=0), None, None, None, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
                                        out_axes=0)


def dmc(params: nn.ParamTree,
        key: chex.PRNGKey,
        data: nn.AINetData,
        Rn_non_local: jnp.array,
        Non_local_coes: jnp.array,
        Non_local_exps: jnp.array,
        ratios_OA: jnp.array, ratios_OB: jnp.array, ratios_OC: jnp.array, ratios_OD: jnp.array,
        cos_theta_OA: jnp.array, cos_theta_OB: jnp.array, cos_theta_OC: jnp.array, cos_theta_OD: jnp.array,
        roted_coords_OA: jnp.array, roted_coords_OB: jnp.array, roted_coords_OC: jnp.array, roted_coords_OD: jnp.array,
        weigths: jnp.array):
    """To make the different configuration have the different key, i.e. random process, we have to write the code in this way. 19.11.2024.
    Except that we need make the batched configurations, we also need batch electrons. 19.11.2024.
    we still need put a lot of effort on the construction of the parallel mechanisms. 16.12.2024.
    OK, we finished the main part of the codes. Next step is polishing. However, before next step, we still need go back
    to the pseudopotential part to rewrite the parallel part. Then decide how to improve the DMC part.
    """
    nelectrons = 16
    natoms = 3
    ndim = 3
    keys = jax.random.split(key, num=4)
    new_data, acceptance = propose_t_moves_parallel(params,
                                                    data,
                                                    Rn_non_local,
                                                    Non_local_coes,
                                                    Non_local_exps,
                                                    ratios_OA, ratios_OB, ratios_OC, ratios_OD,
                                                    cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD,
                                                    roted_coords_OA, roted_coords_OB, roted_coords_OC, roted_coords_OD,
                                                    weights, key)
    drift_diffusion = propose_drift_diffusion(lognetwork=signed_network, tstep=0.1, nelectrons=nelectrons, dim=ndim,
                                              local_energy=localenergy)
    drift_diffusion_parallel = jax.vmap(drift_diffusion,
                                        in_axes=(None, 0, nn.AINetData(positions=0, atoms=0, charges=0),
                                                 None, None, None, None), out_axes=0)

    data = drift_diffusion_parallel(params, keys, data, loss, loss, 10.0, 1.0)
    return None


dmc_parallel = jax.pmap(dmc, in_axes=(0, 0, nn.AINetData(positions=0, atoms=0, charges=0), None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))
output = dmc_parallel(params,
                      subkeys,
                      data,
                      Rn_non_local,
                      Non_local_coes,
                      Non_local_exps,
                      ratios_OA, ratios_OB, ratios_OC, ratios_OD,
                      cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD,
                      roted_coords_OA, roted_coords_OB, roted_coords_OC, roted_coords_OD,
                      weights,)
