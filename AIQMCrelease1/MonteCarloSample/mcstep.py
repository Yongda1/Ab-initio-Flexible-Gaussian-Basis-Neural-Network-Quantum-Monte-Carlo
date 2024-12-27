"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
from AIQMCrelease1.wavefunction import nn
import jax
from jax import lax
from jax import numpy as jnp
from AIQMCrelease1.main import main_adam
from AIQMCrelease1.utils import utils
import kfac_jax


def limdrift(g, tau, acyrus):
    v2 = jnp.sum(g**2)
    taueff = (jnp.sqrt(1 + 2 * tau * acyrus * v2) - 1)/ (acyrus * v2)
    return g * taueff


def walkers_accept(x1, x2, ratio, key, nelectrons: int):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=ratio.shape, minval=0, maxval=1.0)
    cond = ratio > rnd
    cond = jnp.reshape(cond, (nelectrons, 1))
    x_new = jnp.where(cond, x2, x1)
    return x_new, subkey


def monte_carlo(logwavefunction: nn.AINetLike,
                tstep: float,
                ndim: int,
                nelectrons: int,):

    logabs_f = utils.select_output(logwavefunction, 1)
    #sign_f = utils.select_output(logwavefunction, 0)

    def walkers_update(params: nn.ParamTree,
                       data: nn.AINetData,
                       key: chex.PRNGKey):
        """params: batch_params"""
        key, subkey = jax.random.split(key)
        x1 = data.positions
        grad_value = jax.grad(logabs_f, argnums=1)
        grad_value = jax.value_and_grad(logabs_f, argnums=1)

        def grad_f_closure(x):
            return grad_value(params, x, data.atoms, data.charges)

        value, grad = grad_f_closure(x1)
        grad_eff = limdrift(jnp.real(grad), tstep, 0.25)
        #grad_old_eff_s = grad_eff
        initial_configuration = jnp.reshape(x1, (nelectrons, ndim))
        x1 = jnp.reshape(jnp.repeat(jnp.reshape(jnp.reshape(x1, (-1, ndim)), (1, -1)), nelectrons, axis=0), (nelectrons, nelectrons, ndim))
        gauss = jax.random.normal(key=key, shape=(jnp.shape(grad_eff)))
        g = grad_eff + gauss
        g = jnp.reshape(g, (nelectrons, ndim))
        order = jnp.arange(0, nelectrons, step=1)

        def change_configurations(order: jnp.array, g: jnp.array):
            z = jnp.zeros((nelectrons, ndim))
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
        grad_eff = jnp.repeat(jnp.reshape(jnp.reshape(grad_eff, (-1, ndim)), (1, -1)), nelectrons, axis=0)
        backward = (gauss + grad_eff + grad_new_eff)**2
        t_probability = jnp.exp(1/(2 * tstep) * (forward - backward))
        t_probability = jnp.reshape(t_probability, (nelectrons, nelectrons, ndim))
        t_probability = jnp.sum(t_probability, axis=-1)
        t_probability = jnp.diagonal(t_probability)
        """now, we need calculate the wavefunction ratio. 22.11.2024."""
        x1 = jnp.reshape(x1, (nelectrons, -1))
        logabs_f_vmap = jax.vmap(logabs_f, in_axes=(None, 0, None, None,))
        wave_x2 = logabs_f_vmap(params, x2, data.atoms, data.charges)
        wave_x1 = logabs_f_vmap(params, x1, data.atoms, data.charges)
        ratios = jnp.abs(wave_x2/wave_x1) ** 2 * t_probability
        ratios = ratios * (wave_x2 / wave_x1)
        final_configuration, newkey = walkers_accept(initial_configuration,
                                                     changed_configuration,
                                                     ratios,
                                                     key,
                                                     nelectrons)
        final_configuration = jnp.reshape(final_configuration, (-1))
        new_data = nn.AINetData(**(dict(data) | {'positions': final_configuration}))
        return new_data, newkey
    return walkers_update


def generate_batch_key(batch_size: int):
    def get_keys(key: chex.PRNGKey):
        keys = jax.random.split(key, num=batch_size)
        return keys
    return get_keys


def main_monte_carlo(f: nn.AINetLike, key: chex.PRNGKey, params: nn.ParamTree, batch_size: int,):
    """create mont carlo sample loop. One loop is used here. However, we should circumvent it. Later, we optimize it."""
    generate_keys = generate_batch_key(batch_size=batch_size)
    generate_keys_parallel = jax.pmap(generate_keys)
    mc = monte_carlo(logwavefunction=f, tstep=0.1, ndim=3, nelectrons=16)
    mc_parallel = jax.pmap(jax.vmap(mc, in_axes=(None, 0, 0)))

    def mc_step(nsteps: int, data: nn.AINetData):
        keys_batched = generate_keys_parallel(key=key)
        for i in range(nsteps):
            jax.debug.print("i:{}", i)
            new_data, newkeys = mc_parallel(params, data, keys_batched)
            data = new_data
            keys_batched = newkeys
            jax.debug.print("data:{}", data.positions)
        return data

    return mc_step

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
key = jax.random.PRNGKey(seed=1)
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
mc_step = main_monte_carlo(f=signed_network, key=subkeys, params=batch_params, batch_size=4)
new_data = mc_step(nsteps=10, data=data)
'''

