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
import kfac_jax

signed_network, data, params, log_network = main_adam.main()
key = jax.random.PRNGKey(1)
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
localenergy = hamiltonian.local_energy(f=signed_network)
localenergyDMC = hamiltonian.local_energy_dmc(f=signed_network)
total_energy_test = make_loss(log_network, local_energy=localenergy)
total_energy_test_pmap = jax.pmap(total_energy_test, in_axes=(0, 0, nn.AINetData(positions=0, atoms=0, charges=0),), out_axes=(0, 0))
loss, aux_data = total_energy_test_pmap(params, subkeys, data)
#jax.debug.print("loss:{}", loss)
#jax.debug.print("aux_data:{}", aux_data)

def compute_tmoves(lognetwork: nn.LogAINetLike):
    """For a given electron, evaluate all possible t-moves.
    Here, we need read the paper about T-moves.
    The implementation of T-moves is from the paper,
    'Nonlocal pseudopotentials and time-step errors in diffusion Monte Carlo' written by Anderson and Umrigar."""
    def calculate_ratio_weight(params: nn.ParamTree, data: nn.AINetData):
        jax.debug.print("data.positions:{}", data.positions)
        n = data.positions.shape[0]
        jax.debug.print("n:{}", n)
        return None

    return calculate_ratio_weight
    
'''
ratio_weight = compute_tmoves(lognetwork=log_network)
run_ratio_weight = jax.pmap(jax.vmap(ratio_weight, in_axes=(None, nn.AINetData(positions=0, atoms=0, charges=0)), out_axes=0))
output = run_ratio_weight(params, data)
'''

batch_lognetwork = jax.pmap(jax.vmap(log_network, in_axes=(None, 0, 0, 0), out_axes=0))


def comput_S(e_trial: float, e_est: float, branchcut: float, v2: float, tau: float, eloc: float, nelec: int):
    """here, we calculate the S. 24.11.2024."""
    jax.debug.print("v2:{}", v2)
    jax.debug.print("eloc:{}", eloc)
    #e_cut = e_est-eloc
    #e_cut = jnp.min(jnp.array([jnp.abs(e_cut), branchcut]))*jnp.sign(e_cut)
    #denominator = 1 + (v2 * tau/nelec) ** 2
    #return e_trial - e_est + e_cut/denominator


def walkers_accept(x1, x2, ratio, key, nelectrons: int):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=ratio.shape, minval=0, maxval=1.0)
    cond = ratio > rnd
    #jax.debug.print("cond:{}", cond)
    cond = jnp.reshape(cond, (nelectrons, 1))
    #jax.debug.print("cond:{}", cond)
    x_new = jnp.where(cond, x2, x1)
    return x_new


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
                                  etrial: float, e_est: float, branchcut_start: float):
        x1 = data.positions
        grad_value = jax.value_and_grad(logabs_f, argnums=1)

        def grad_f_closure(x):
            return grad_value(params, x, data.atoms, data.charges)


        value, grad = grad_f_closure(x1)
        grad_eff = limdrift(jnp.real(grad), tstep, 0.25)
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

        final_configuration = walkers_accept(initial_configuration, changed_configuration, ratios, key, nelectrons)

        final_configuration = jnp.reshape(final_configuration, (-1))
        #jax.debug.print("final_configuration:{}", final_configuration)
        new_data = nn.AINetData(**(dict(data) | {'positions': final_configuration}))
        grad_old_eff = grad_eff
        grad_new_eff = grad_new_eff
        #jax.debug.print("grad_old_eff:{}", grad_old_eff)
        #jax.debug.print("grad_new_eff:{}", grad_new_eff)
        """we need calculate the local energy here."""
        #jax.debug.print("data:{}", data)
        #jax.debug.print("new_data:{}", new_data)
        jax.debug.print("x1:{}", x1)
        jax.debug.print("x2:{}", x2)
        eloc_old = jax.vmap(local_energy, in_axes=(None, None, 0, None, None))(params, key, x1, data.atoms, data.charges)
        #eloc_new = localenergy(params, key, new_data)
        jax.debug.print("eloc_old:{}", eloc_old)
        """To be continued...27.11.2024"""
        #jax.debug.print("eloc_new:{}", eloc_new)
        #batch_local_energy = jax.vmap(local_energy, in_axes=(None, None, nn.AINetData(positions=0, atoms=0, charges=0),
        #                                                     ), out_axes=(0, 0))
        #e_l, e_l_mat = batch_local_energy(params, key, data)
        """we have more problems here. to be continued... 25.11.2024."""
        #S_old = comput_S(e_trial=etrial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_old_eff), tau=tstep,
        #                 eloc=eloc_old, nelec=nelectrons)
        return None
    return calculate_drift_diffusion



drift_diffusion = propose_drift_diffusion(lognetwork=signed_network, tstep=0.1, nelectrons=4, dim=3, local_energy=localenergyDMC)
drift_diffusion_parallel = jax.vmap(drift_diffusion, in_axes=(None, 0, nn.AINetData(positions=0, atoms=0, charges=0),
                                                              None, None, None), out_axes=0)


def main_drift_diffusion(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData):
    """To make the different configuration have the different key, i.e. random process, we have to write the code in this way. 19.11.2024.
    Except that we need make the batched configurations, we also need batch electrons. 19.11.2024."""
    keys = jax.random.split(key, num=4)
    data = drift_diffusion_parallel(params, keys, data, loss, loss, 10.0)


main_drift_diffusion_parallel = jax.pmap(main_drift_diffusion)
output = main_drift_diffusion_parallel(params, subkeys, data)

'''
def propose_tmoves(lognework: nn.LogAINetLike, data: nn.AINetData, ):
    """t-moves for dmc."""

def dmc_propagate(lognetwork: nn.LogAINetLike, data: nn.AINetData, tstep: float, nsteps: int,) -> jnp.array:
    """This is the dmc propagation function.
    we need introduce T-moves in the calculation. This means """
    pos = data.positions
'''
