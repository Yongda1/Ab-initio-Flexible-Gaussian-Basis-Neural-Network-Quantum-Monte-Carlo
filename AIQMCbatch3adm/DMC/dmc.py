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
jax.debug.print("loss:{}", loss)
#jax.debug.print("aux_data:{}", aux_data)



total_energy, \
ratios_OA, ratios_OB, ratios_OC, ratios_OD, \
cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD, \
roted_configurations_OA, roted_configurations_OB, roted_configurations_OC, roted_configurations_OD, weights = \
    total_energy_pseudopotential(data=data, params=params,
                                 rn_local_general=Rn_local,
                                 rn_non_local_general=Rn_non_local,
                                 local_coefficient_general=Local_coes,
                                 nonlocal_coefficient_general=Non_local_coes,
                                 local_exponent_general=Local_exps,
                                 nonlocal_exponent_general=Non_local_exps, nelectrons=16, natoms=3, list_l=2, batch_size=4)


#jax.debug.print("ratios_OA:{}", ratios_OA)
jax.debug.print("ratios_OA_shape:{}", ratios_OA.shape)
#jax.debug.print("cos_theta_OA:{}", cos_theta_OA)
jax.debug.print("cos_theta_OA_shape:{}", cos_theta_OA.shape)
jax.debug.print("roted_configurations_OA_shape:{}", roted_configurations_OA.shape)

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


def compute_tmoves(lognetwork: nn.LogAINetLike, list_l: float, tstep: float):
    """For a given electron, evaluate all possible t-moves.
    Here, we need read the paper about T-moves.
    The implementation of T-moves is from the paper,
    'Nonlocal pseudopotentials and time-step errors in diffusion Monte Carlo' written by Anderson and Umrigar.
    We finished the pseudopotential part currently, now turn to the T-moves. 9.12.2024.
    we use the same strategy for this function. The input is just one configuration.
    """
    def calculate_ratio_weight_tmoves(params: nn.ParamTree,
                                      data: nn.AINetData,
                                      costheta: jnp.array,
                                      ratios: jnp.array,
                                      Rn_non_local: jnp.array,
                                      Non_local_coes: jnp.array,
                                      Non_local_exps: jnp.array):
        #jax.debug.print("data.positions:{}", data.positions)
        #jax.debug.print("costheta:{}", costheta)
        #jax.debug.print("ratios:{}", ratios)
        output_P_l = jnp.array(P_l_theta(costheta, list_l=2.0))
        #jax.debug.print("output_P_l:{}", output_P_l)
        #jax.debug.print("output_P_l_shape:{}", output_P_l.shape)
        v_r_non_local = get_non_v_l(data, Rn_non_local, Non_local_coes, Non_local_exps)
        """we have some problems about the data type. It should be float32 but currently it is int32. Maybe it is a bug. We solve it later."""
        #jax.debug.print("v_r_non_local_shape:{}", v_r_non_local.shape)

        def multiply_weights(v_r_non_local: jnp.array, output_P_l: jnp.array):
            return (jnp.exp(-1 * tstep * v_r_non_local) - 1) * output_P_l

        multiply_weights_parallel = jax.vmap(jax.vmap(multiply_weights, in_axes=(2, 0), out_axes=0), in_axes=(None, 3), out_axes=0)
        """the shape of weights should be number of points, angular momentum functions, number of electrons, number of atoms"""
        weights = multiply_weights_parallel(v_r_non_local, output_P_l)
        #jax.debug.print("weights_shape:{}", weights.shape)
        weights = jnp.sum(weights, axis=1)
        #jax.debug.print("weights_shape:{}", weights.shape)
        #jax.debug.print("ratios_shape:{}", ratios.shape)

        def run_t_amplitudes(ratios: jnp.array, weights: jnp.array):
            return ratios * weights

        run_t_amplitudes_parallel = jax.vmap(run_t_amplitudes, in_axes=(2, 0), out_axes=0)
        t_amplitudes = run_t_amplitudes_parallel(ratios, weights)
        forward_probability = jnp.zeros_like(t_amplitudes)
        """we need propose the move to the new configuration here. to be continued...10.12.2024."""
        forward_probability_output = jnp.where(t_amplitudes > forward_probability, t_amplitudes, forward_probability)
        #jax.debug.print("t_amplitudes_shape:{}", forward_probability_output.shape)
        #jax.debug.print("t_amplitudes:{}", forward_probability_output)
        #norm = 1.0 + jnp.sum(forward_probability_output)
        #jax.debug.print("norm:{}", norm)
        #jax.debug.print("sum:{}", jnp.sum(forward_probability_output))



        return forward_probability_output

    return calculate_ratio_weight_tmoves
    

ratio_weight = compute_tmoves(lognetwork=log_network, list_l=2, tstep=0.1)
#run_ratio_weight = jax.pmap(jax.vmap(ratio_weight,
#                                     in_axes=(None, nn.AINetData(positions=0, atoms=0, charges=0), 0, 0, None, None, None), out_axes=0),
#                            in_axes=(0, nn.AINetData(positions=0, atoms=0, charges=0), 0, 0, None, None, None), out_axes=0)
#output = run_ratio_weight(params, data, cos_theta_OA, ratios_OA, Rn_non_local, Non_local_coes, Non_local_exps)



def propose_t_moves(params: nn.ParamTree,
                    data: nn.AINetData,
                    Rn_non_local: jnp.array,
                    Non_local_coes: jnp.array,
                    Non_local_exps: jnp.array,
                    ratios_OA: jnp.array, ratios_OB: jnp.array, ratios_OC: jnp.array, ratios_OD: jnp.array,
                    cos_theta_OA: jnp.array, cos_theta_OB: jnp.array, cos_theta_OC: jnp.array, cos_theta_OD: jnp.array,
                    roted_configurations_OA: jnp.array, roted_configurations_OB: jnp.array, roted_configurations_OC: jnp.array, roted_configurations_OD: jnp.array,
                    weigths: jnp.array
                    ):
    forward_probability_output_OA = ratio_weight(params, data, cos_theta_OA, ratios_OA, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)
    forward_probability_output_OB = ratio_weight(params, data, cos_theta_OB, ratios_OB, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)
    forward_probability_output_OC = ratio_weight(params, data, cos_theta_OC, ratios_OC, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)
    forward_probability_output_OD = ratio_weight(params, data, cos_theta_OD, ratios_OD, Rn_non_local,
                                                     Non_local_coes, Non_local_exps)
    jax.debug.print("forward_probability_output_OA_shape:{}", forward_probability_output_OA.shape)
    jax.debug.print("forward_probability_output_OB_shape:{}", forward_probability_output_OB.shape)
    jax.debug.print("forward_probability_output_OC_shape:{}", forward_probability_output_OC.shape)
    jax.debug.print("forward_probability_output_OD_shape:{}", forward_probability_output_OD.shape)
    roted_configurations_OA = jnp.transpose(roted_configurations_OA, (2, 0, 1, 3))
    roted_configurations_OB = jnp.transpose(roted_configurations_OB, (2, 0, 1, 3))
    roted_configurations_OC = jnp.transpose(roted_configurations_OC, (2, 0, 1, 3))
    roted_configurations_OD = jnp.transpose(roted_configurations_OD, (2, 0, 1, 3))
    jax.debug.print("roted_configurations_OA_shape:{}", roted_configurations_OA.shape)
    jax.debug.print("roted_configurations_OB_shape:{}", roted_configurations_OB.shape)
    jax.debug.print("roted_configurations_OC_shape:{}", roted_configurations_OC.shape)
    jax.debug.print("roted_configurations_OD_shape:{}", roted_configurations_OD.shape)
    norm = 1 + jnp.sum(weigths[0] * forward_probability_output_OA) + jnp.sum(weigths[1] * forward_probability_output_OB) + \
           jnp.sum(weigths[2] * forward_probability_output_OC) + jnp.sum(weigths[3] * forward_probability_output_OD)

    jax.debug.print("norm:{}", norm)

    forward_probability_output_OA = jnp.reshape(forward_probability_output_OA, (-1,))
    forward_probability_output_OB = jnp.reshape(forward_probability_output_OB, (-1,))
    forward_probability_output_OC = jnp.reshape(forward_probability_output_OC, (-1,))
    forward_probability_output_OD = jnp.reshape(forward_probability_output_OD, (-1,))
    forward_probability_output_total = jnp.concatenate([forward_probability_output_OA,
                                                       forward_probability_output_OB,
                                                       forward_probability_output_OC,
                                                       forward_probability_output_OD], axis=0)

    jax.debug.print("forward_probability_output_total_shape:{}", forward_probability_output_total.shape)


    def select_walker(a: jnp.array):
        r = np.random.rand()
        jax.debug.print("r:{}", r)
        return jnp.searchsorted(a, r)

    cdf = jnp.cumsum(forward_probability_output_total / norm)
    selected_moves = jnp.apply_along_axis(func1d=select_walker, axis=-1, arr=cdf)
    jax.debug.print("selected_moves:{}", selected_moves)


    return None



propose_t_moves_parallel = jax.pmap(jax.vmap(propose_t_moves,
                                             in_axes=(None, nn.AINetData(positions=0, atoms=0, charges=0), None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
                                             out_axes=0),
                                    in_axes=(0, nn.AINetData(positions=0, atoms=0, charges=0), None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
                                    out_axes=0)

output = propose_t_moves_parallel(params, data, Rn_non_local, Non_local_coes, Non_local_exps,
                         ratios_OA, ratios_OB, ratios_OC, ratios_OD,
                         cos_theta_OA, cos_theta_OB, cos_theta_OC, cos_theta_OD,
                         roted_configurations_OA, roted_configurations_OB, roted_configurations_OC, roted_configurations_OD, weights)



batch_lognetwork = jax.pmap(jax.vmap(log_network, in_axes=(None, 0, 0, 0), out_axes=0))


def comput_S(e_trial: float, e_est: float, branchcut: float, v2: jnp.array, tau: float, eloc: jnp.array, nelec: int):
    """here, we calculate the S. 24.11.2024."""
    v2 = jnp.sum(v2, axis=-1)
    #jax.debug.print("v2:{}", v2)
    #jax.debug.print("e_trial:{}", e_trial)
    #jax.debug.print("e_est:{}", e_est)
    eloc = jnp.real(eloc)
    e_est = jnp.real(e_est)
    e_trial = jnp.real(e_trial)
    e_cut = e_est-eloc
    #jax.debug.print("eloc:{}", eloc)
    #jax.debug.print("e_cut:{}", e_cut)
    #jax.debug.print("branchcut:{}", branchcut)
    e_cut = jnp.min(jnp.array([jnp.abs(e_cut[0]), branchcut]))*jnp.sign(e_cut)
    denominator = 1 + (v2 * tau/nelec) ** 2
    return e_trial - e_est + e_cut/denominator


def walkers_accept(x1, x2, ratio, key, nelectrons: int):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=ratio.shape, minval=0, maxval=1.0)
    cond = ratio > rnd
    #jax.debug.print("cond:{}", cond)
    cond = jnp.reshape(cond, (nelectrons, 1))
    jax.debug.print("cond:{}", cond)
    #x2_accept = x2[cond]
    #jax.debug.print("x2_accept:{}", x2_accept)

    x_new = jnp.where(cond, x2, x1)
    #xnew_accept = jnp.sum(x_new, where=cond)
    #jax.debug.print("xnew_accept:{}", xnew_accept)
    tdamp = jnp.sum(x_new) / jnp.sum(x2)
    jax.debug.print("tdamp:{}", tdamp)
    return x_new, jnp.abs(tdamp)


def limdrift(g, tau, acyrus):
    v2 = jnp.sum(g**2)
    taueff = (jnp.sqrt(1 + 2 * tau * acyrus * v2) - 1)/ (acyrus * v2)
    return g * taueff

def transition_probability(pos: jnp.array, grad_value_eff: jnp.array, key: chex.PRNGKey):
    """we have one parallel problem here. How to change the electron pos one by one on GPU? 19.11.2024."""
    x1 = pos
    jax.debug.print("x1:{}", x1)
    gauss = jax.random.normal(key=key, shape=(jnp.shape(x1)))
    x2 = x1 + gauss + grad_value_eff
    jax.debug.print("x2:{}", x2)
    return x2


def propose_drift_diffusion(lognetwork: nn.LogAINetLike, tstep: float, nelectrons: int, dim: int, local_energy: hamiltonian.LocalEnergyDMC):
    """this is just one step move."""
    logabs_f = utils.select_output(lognetwork, 1)
    sign_f = utils.select_output(lognetwork, 0)

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
        x1 = jnp.reshape(x1, (-1, 3))
        x1 = jnp.reshape(x1, (1, -1))
        x1 = jnp.repeat(x1, nelectrons, axis=0)
        x1 = jnp.reshape(x1, (nelectrons, nelectrons, dim))
        gauss = jax.random.normal(key=key, shape=(jnp.shape(grad_eff)))
        g = grad_eff + gauss
        g = jnp.reshape(g, (nelectrons, dim))
        order = jnp.arange(0, nelectrons, step=1)
        z = jnp.zeros(jnp.shape(x1))
        "here, I dont find any good ways to do it parallely. "
        "I finished it in this bad way. We need go back later. 22.11.2024."
        for i in order:
            temp = z[i]
            temp = temp.at[i].add(g[i])
            z = z.at[i].set(temp)

        #jax.debug.print("z_new:{}", z)
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
        ratios = wave_x2/wave_x1 ** 2 * t_probability
        #jax.debug.print("ratios:{}", ratios)
        jax.debug.print("initial_configuration:{}", initial_configuration)
        jax.debug.print("changed_configuration:{}", changed_configuration)
        """to be continued...22.11.2024. we need do the acceptance judgement later."""

        final_configuration, tdamp = walkers_accept(initial_configuration, changed_configuration, ratios, key, nelectrons)

        final_configuration = jnp.reshape(final_configuration, (-1))

        new_data = nn.AINetData(**(dict(data) | {'positions': final_configuration}))
        value_new_s, grad_new_s = grad_f_closure(new_data.positions)
        grad_new_eff_s = limdrift(grad_new_s, tstep, 0.25)
        #jax.debug.print("grad_old_eff_s:{}", grad_old_eff_s)
        #jax.debug.print("grad_new_eff_s:{}", grad_new_eff_s)
        """we need calculate the local energy here."""

        eloc_old, e_l_mat_old = local_energy(params, key, data)
        #jax.debug.print("eloc_old:{}", eloc_old)
        eloc_new, e_l_mat_old = local_energy(params, key, new_data)
        #jax.debug.print("eloc_new:{}", eloc_new)
        #tdamp =
        S_old = comput_S(e_trial=etrial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_old_eff_s), tau=tstep,eloc=eloc_old, nelec=nelectrons)
        S_new = comput_S(e_trial=etrial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_new_eff_s), tau=tstep,eloc=eloc_new, nelec=nelectrons)
        jax.debug.print("S_old:{}", S_old)
        jax.debug.print("S_new:{}", S_new)
        wmult = jnp.exp(tstep * tdamp * (0.5 * S_new + 0.5 * S_old))
        weights *= wmult
        jax.debug.print("weights:{}", weights)
        """we stop here currently.  
        We need finish the pseudopotential part first, then t-moves part, eventually back to this line. 28.11.2024."""
        return None
    return calculate_drift_diffusion



drift_diffusion = propose_drift_diffusion(lognetwork=signed_network, tstep=0.1, nelectrons=4, dim=3, local_energy=localenergy)
drift_diffusion_parallel = jax.vmap(drift_diffusion, in_axes=(None, 0, nn.AINetData(positions=0, atoms=0, charges=0),
                                                              None, None, None, None), out_axes=0)


def main_drift_diffusion(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData):
    """To make the different configuration have the different key, i.e. random process, we have to write the code in this way. 19.11.2024.
    Except that we need make the batched configurations, we also need batch electrons. 19.11.2024."""
    keys = jax.random.split(key, num=4)
    data = drift_diffusion_parallel(params, keys, data, loss, loss, 10.0, 1.0)


main_drift_diffusion_parallel = jax.pmap(main_drift_diffusion)
#output = main_drift_diffusion_parallel(params, subkeys, data)
