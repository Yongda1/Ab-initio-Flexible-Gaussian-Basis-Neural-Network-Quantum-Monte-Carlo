"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
import jax
from jax import numpy as jnp
from jax import lax
#from AIQMCpretrain1.wavefunction_Ynlm import nn
from modified_ferminet.ferminet import networks as nn
from modified_ferminet.ferminet.utils import utils


def limdrift(g, tau, acyrus):
    v2 = jnp.sum(g**2)
    taueff = (jnp.sqrt(1 + 2 * tau * acyrus * v2) - 1) / (acyrus * v2)
    return g * taueff



def walkers_accept(x1, x2, acceptance, key, nelectrons: int, batch_size: int):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=acceptance.shape, minval=0, maxval=1.0)
    cond = acceptance > rnd
    cond = jnp.reshape(cond, (batch_size, nelectrons, 1))
    #jax.debug.print("cond:{}", cond)
    x_new = jnp.where(cond, x2, x1)
    return x_new, subkey


def walkers_update(logabs_f: nn.LogFermiNetLike,
                   params: nn.ParamTree,
                   data: nn.FermiNetData,
                   key: chex.PRNGKey,
                   tstep: float,
                   ndim: int,
                   nelectrons: int,
                   batch_size: int, #this batch_size should be the number of walkers on each GPU
                   i=0):
    """params: batch_params.
    Something is wrong here. Probably it is due to the delay update of walkers. 8.4.2025. """
    key, subkey = jax.random.split(key)
    x1 = data.positions
    grad_value = jax.grad(logabs_f, argnums=1)
    atoms = data.atoms[0]
    charges = data.charges[0]
    spins = data.spins[0]

    def grad_f_closure(x):
        return grad_value(params, x, spins, atoms, charges)

    #grad_test = jax.vmap(grad_value, in_axes=(None, 0, 0, 0, 0))(params, data.positions, data.spins, data.atoms, data.charges)
    #jax.debug.print("grad_test:{}", grad_test)
    grad_f = jax.vmap(grad_f_closure, in_axes=0)
    #jax.debug.print("x1:{}", x1)
    grad = grad_f(x1)
    #jax.debug.print("grad:{}", grad)
    initial_configuration = jnp.reshape(x1, (batch_size, nelectrons, ndim))
    x1 = jnp.reshape(jnp.reshape(x1, (batch_size, -1, ndim)), (batch_size, 1, -1))
    x1 = jnp.reshape(jnp.repeat(x1, nelectrons, axis=1), (batch_size, nelectrons, nelectrons, ndim))
    gauss = jnp.sqrt(tstep) * jax.random.normal(key=key, shape=(jnp.shape(grad)))

    grad_eff = limdrift(grad, tstep, 0.25)

    g = grad_eff * tstep + gauss
    g = jnp.reshape(g, (batch_size, nelectrons, ndim))
    order = jnp.arange(0, nelectrons, step=1)
    order = jnp.repeat(order[None, ...], batch_size, axis=0)
    """maybe some thing is wrong here. 8.4.2025."""
    def change_configurations(order: jnp.array, g: jnp.array):
        z = jnp.zeros((nelectrons, ndim))
        temp = z.at[order].add(g[order])
        return temp

    change_configurations_parallel = jax.vmap(jax.vmap(change_configurations, in_axes=(0, None)), in_axes=(0, 0), out_axes=0)
    #jax.debug.print("g:{}", g)
    z = change_configurations_parallel(order, g)
    #jax.debug.print("z:{}", z)
    x2 = x1 + z
    changed_configuration = g + initial_configuration
    x2 = jnp.reshape(x2, (batch_size, nelectrons, -1))
    grad_new = jax.vmap(jax.vmap(grad_f_closure, in_axes=0, out_axes=0), in_axes=0)(x2)
    grad_new_eff = limdrift(grad_new, tstep, 0.25)
    grad_eff = jnp.repeat(grad_eff, nelectrons, axis=0)
    grad_eff = jnp.reshape(grad_eff, (batch_size, nelectrons, -1))
    gauss = jnp.sqrt(tstep) * jax.random.normal(key=key, shape=(jnp.shape(grad_eff)))
    forward = gauss ** 2
    backward = (gauss + (grad_eff + grad_new_eff) * tstep) ** 2
    t_probability = jnp.exp((forward - backward)/(2 * tstep))
    t_probability = jnp.reshape(t_probability, (batch_size, nelectrons, nelectrons, ndim))
    t_probability = jnp.sum(t_probability, axis=-1)

    def return_t(t_pro_inner: jnp.array):
        return jnp.diagonal(t_pro_inner)

    return_t_parallel = jax.vmap(return_t, in_axes=0)
    t_pro = return_t_parallel(t_probability)
    logabs_f_vmap = jax.vmap(jax.vmap(logabs_f, in_axes=(None, 0, None, None, None,)), in_axes=(None, 0, None, None, None))
    #jax.debug.print("x2:{}", x2)
    wave_x2 = logabs_f_vmap(params, x2, spins, atoms, charges)
    x1 = jnp.reshape(x1, (batch_size, nelectrons, -1))
    wave_x1 = logabs_f_vmap(params, x1, spins, atoms, charges)
    acceptance = jnp.abs(jnp.exp(wave_x2 - wave_x1)) ** 2 * t_pro
    final_configuration, newkey = walkers_accept(initial_configuration,
                                                 changed_configuration,
                                                 acceptance,
                                                 key,
                                                 nelectrons,
                                                 batch_size)
    #jax.debug.print("i:{}", i)
    #jax.debug.print("final:{}", final_configuration)
    final_configuration = jnp.reshape(final_configuration, (batch_size, -1))
    new_data = nn.FermiNetData(**(dict(data) | {'positions': final_configuration}))
    return new_data, newkey

'''
def generate_batch_key(batch_size: int):
    def get_keys(key: chex.PRNGKey):
        keys = jax.random.split(key, num=batch_size)
        return keys
    return get_keys
'''

def main_monte_carlo(f: nn.FermiNetLike,
                     tstep: float,
                     ndim: int,
                     nelectrons: int,
                     nsteps: int,
                     batch_size: int):
    """create mont carlo sample loop. One loop is used here. However, we should circumvent it. Later, we optimize it."""
    logabs_f = utils.select_output(f, 1)
    #jax.debug.print("batch_size_each_GPU:{}", batch_size)
    @jax.jit
    def mc_step(params: nn.ParamTree, data: nn.FermiNetData, key: chex.PRNGKey, ):


        def step_fn(i, x):
            return walkers_update(logabs_f, params, *x, tstep=tstep, ndim=ndim, nelectrons=nelectrons, batch_size=batch_size, i=i)

        new_data, new_key = lax.fori_loop(lower=0, upper=nsteps, body_fun=step_fn, init_val=(data, key))
        pmove = 1.0
        return new_data, pmove

    return mc_step

'''
structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['H', 'H']
atoms = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
charges = jnp.array([1.0, 1.0])
spins = jnp.array([1.0, -1.0])
signed_network, data, params, log_network = main(atoms=atoms,
                                                 charges=charges,
                                                 spins=spins,
                                                 tstep=0.02,
                                                 nelectrons=2,
                                                 natoms=2,
                                                 ndim=3,
                                                 batch_size=4,
                                                 iterations=1,
                                                 structure=structure,)

key = jax.random.PRNGKey(seed=1)
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
mc_step = main_monte_carlo(f=signed_network, tstep=0.1, ndim=3, nelectrons=2, nsteps=50, batch_size=4)
mc_step_parallel = jax.pmap(mc_step)
new_data = mc_step_parallel(params=params, data=data, key=subkeys)
jax.debug.print("new_data:{}", new_data)
'''
