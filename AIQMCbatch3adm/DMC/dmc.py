"""today, we start to do some dmc codes. Because of the problem of the parameter tree, we still have
one problem about the parallel calculation. But actually our calculation engine is working well."""
import chex
import jax
import jax.numpy as jnp
from AIQMCbatch3adm import nn
from AIQMCbatch3adm import main_adam
from AIQMCbatch3adm.utils import utils
import kfac_jax
signed_network, data, params, log_network = main_adam.main()
key = jax.random.PRNGKey(1)
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)

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


def limdrift(g, tau, acyrus=0.25):
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


def propose_drift_diffusion(lognetwork: nn.LogAINetLike, tstep: float, nelectrons: int, dim: int, transition_probability):
    logabs_f = utils.select_output(lognetwork, 1)
    sign_f = utils.select_output(lognetwork, 0)
    t_p_parallel = jax.vmap(transition_probability, in_axes=(0, 0, 0), out_axes=0)
    def calculate_drift_diffusion(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData,):
        #jax.debug.print("data.positions:{}", data.positions)
        #jax.debug.print("data.atoms:{}", data.atoms)
        #jax.debug.print("data.charges:{}", data.charges)
        #jax.debug.print("key: {}", key)
        x1 = data.positions
        grad_value = jax.value_and_grad(logabs_f, argnums=1)

        def grad_f_closure(x):
            return grad_value(params, x, data.atoms, data.charges)

        value, grad = grad_f_closure(x1)
        grad_eff = limdrift(jnp.real(grad), tstep)
        subkeys = jax.random.split(key, num=nelectrons)
        #  jax.debug.print("x1:{}", x1)
        jax.debug.print("x1:{}", x1)
        x1 = jnp.reshape(x1, (1, -1))
        x1 = jnp.repeat(x1, nelectrons, axis=0)
        x1 = jnp.reshape(x1, (nelectrons, nelectrons, 3))
        jax.debug.print("x1:{}", x1)
        #x1 = jnp.reshape(x1, (nelectrons, dim))
        #grad_eff = jnp.reshape(grad_eff, (nelectrons, dim))
        #jax.debug.print("x1:{}", x1)
        #x2 = t_p_parallel(x1, grad_eff, subkeys)
        #jax.debug.print("x2:{}", x2)
        #gauss = jax.random.normal(key=key, shape=(jnp.shape(x1)))
        #jax.debug.print("x1:{}", x1)
        #x2 = x1 + gauss + grad_eff
        #jax.debug.print("x2:{}", x2)
        #value_new, grad_new = grad_f_closure(x2)
        #forward = gauss**2
        #backward = (gauss + grad + grad_new)**2
        #t_probability = jnp.exp(1/(2 * tstep) * (forward - backward))
        #jax.debug.print("gauss:{}", gauss)
        #jax.debug.print("grad_eff:{}", grad_eff)

        return None
    return calculate_drift_diffusion



drift_diffusion = propose_drift_diffusion(lognetwork=signed_network, tstep=0.1, nelectrons=4, dim=3, transition_probability=transition_probability)
drift_diffusion_parallel = jax.vmap(drift_diffusion, in_axes=(None, 0, nn.AINetData(positions=0, atoms=0, charges=0)), out_axes=0)


def main_drift_diffusion(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData):
    """To make the different configuration have the different key, i.e. random process, we have to write the code in this way. 19.11.2024.
    Except that we need make the batched configurations, we also need batch electrons. 19.11.2024."""
    keys = jax.random.split(key, num=4)
    output = drift_diffusion_parallel(params, keys, data)


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
