from AIQMCrelease2.wavefunction import nn
from AIQMCrelease2.Energy import pphamiltonian
from AIQMCrelease2.utils import utils
import jax.numpy as jnp
import chex
import jax


def limdrift(g, tau, acyrus):
    v2 = jnp.sum(g**2)
    taueff = (jnp.sqrt(1 + 2 * tau * acyrus * v2) - 1)/ (acyrus * v2)
    return g * taueff


def walkers_accept(x1, x2, acceptance, key, nelectrons: int, batch_size: int):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=acceptance.shape, minval=0, maxval=1.0)
    cond = acceptance > rnd
    cond = jnp.reshape(cond, (batch_size, nelectrons, 1))
    x_new = jnp.where(cond, x2, x1)
    tdamp = jnp.sum(x_new) / jnp.sum(x2)
    return x_new, subkey, tdamp


def propose_drift_diffusion(logabs_f: nn.LogAINetLike,
                            tstep: float,
                            ndim: int,
                            nelectrons: int,
                            batch_size: int):
    def drift_diffusion(params, key: chex.PRNGKey, data: nn.AINetData):
        key, subkey = jax.random.split(key)
        x1 = data.positions
        grad_value = jax.grad(logabs_f, argnums=1)
        atoms = data.atoms[0]
        charges = data.charges[0]
        spins = data.spins[0]

        def grad_f_closure(x):
            return grad_value(params, x, spins, atoms, charges)

        grad_f = jax.vmap(grad_f_closure, in_axes=0)
        grad = grad_f(x1)
        initial_configuration = jnp.reshape(x1, (batch_size, nelectrons, ndim))
        x1 = jnp.reshape(jnp.reshape(x1, (batch_size, -1, ndim)), (batch_size, 1, -1))
        x1 = jnp.reshape(jnp.repeat(x1, nelectrons, axis=1), (batch_size, nelectrons, nelectrons, ndim))
        gauss = jnp.sqrt(tstep) * jax.random.normal(key=key, shape=(jnp.shape(grad)))

        grad_eff = limdrift(grad, tstep, 0.25)

        g = grad_eff * tstep + gauss
        g = jnp.reshape(g, (batch_size, nelectrons, ndim))
        order = jnp.arange(0, nelectrons, step=1)
        order = jnp.repeat(order[None, ...], batch_size, axis=0)

        def change_configurations(order: jnp.array, g: jnp.array):
            z = jnp.zeros((nelectrons, ndim))
            temp = z.at[order].add(g[order])
            return temp

        change_configurations_parallel = jax.vmap(jax.vmap(change_configurations, in_axes=(0, None)), in_axes=(0, 0),
                                                  out_axes=0)
        z = change_configurations_parallel(order, g)
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
        t_probability = jnp.exp((forward - backward) / (2 * tstep))
        t_probability = jnp.reshape(t_probability, (batch_size, nelectrons, nelectrons, ndim))
        t_probability = jnp.sum(t_probability, axis=-1)

        def return_t(t_pro_inner: jnp.array):
            return jnp.diagonal(t_pro_inner)

        return_t_parallel = jax.vmap(return_t, in_axes=0)
        t_pro = return_t_parallel(t_probability)
        logabs_f_vmap = jax.vmap(jax.vmap(logabs_f, in_axes=(None, 0, None, None, None,)),
                                 in_axes=(None, 0, None, None, None))
        wave_x2 = logabs_f_vmap(params, x2, spins, atoms, charges)
        x1 = jnp.reshape(x1, (batch_size, nelectrons, -1))
        wave_x1 = logabs_f_vmap(params, x1, spins, atoms, charges)
        acceptance = jnp.abs(jnp.exp(wave_x2 - wave_x1)) ** 2 * t_pro
        final_configuration, newkey, tdamp = walkers_accept(initial_configuration,
                                                     changed_configuration,
                                                     acceptance,
                                                     key,
                                                     nelectrons,
                                                     batch_size)

        final_configuration = jnp.reshape(final_configuration, (batch_size, -1))
        new_data = nn.AINetData(**(dict(data) | {'positions': final_configuration}))

        grad_new_s = grad_f(new_data.positions)
        grad_new_eff_s = limdrift(grad_new_s, tstep, 0.25)

        return new_data, newkey, tdamp, grad_eff, grad_new_eff_s

    return drift_diffusion