"""This module tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
import jax
from jax import numpy as jnp
from jax import lax
from GaussianNet.wavefunction import networks
from GaussianNet.tools.utils import utils


def walkers_update(f: networks.GaussianNetLike,
                   tstep: float,
                   ndim: int,
                   nelectrons: int,):
    """single configuration calculation. Later, we need use pmap and vmap to make the data be single batch."""
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    def mcstep(data: networks.GaussianNetData, params: networks.ParamTree, key: chex.PRNGKey,):
        grad_f = jax.grad(logabs_f, argnums=1)
        #jax.debug.print("data:{}", data)
        #jax.debug.print("key:{}", key)
        def grad_f_closure(x):
            return grad_f(params, x, data.spins, data.atoms, data.charges)

        primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

        grad_phase = jax.grad(phase_f, argnums=1)

        def grad_phase_closure(x):
            return grad_phase(params, x, data.spins, data.atoms, data.charges)

        phase_primal, dgrad_phase = jax.linearize(
            grad_phase_closure, data.positions)

        O_old = primal + 1.j * phase_primal
        O_old = jnp.reshape(O_old, (nelectrons, ndim))
        x1 = data.positions
        x1 = jnp.reshape(x1, (nelectrons, ndim))
        x_new = jnp.zeros_like(x1)
        #jax.debug.print("O_old:{}", O_old)
        #jax.debug.print("x1:{}", x1)
        for i in range(len(x1)):
            key_inner, key_new_inner = jax.random.split(key)
            gauss = jnp.sqrt(tstep) * jax.random.normal(key=key_new_inner, shape=(jnp.shape(x1[i])))
            O_eff = O_old[i]
            #jax.debug.print("O_eff:{}", O_eff)
            temp = O_eff + gauss + x1[i]
            #jax.debug.print("temp:{}", temp)
            x2 = x1.at[i].set(temp)
            x_2_temp = jnp.reshape(x2, (-1))
            x_1_temp = jnp.reshape(x1, (-1))
            wave_x1_mag = logabs_f(params, x_1_temp, data.spins, data.atoms, data.charges)
            wave_x2_mag = logabs_f(params, x_2_temp, data.spins, data.atoms, data.charges)
            wave_x1_phase = phase_f(params, x_1_temp, data.spins, data.atoms, data.charges)
            wave_x2_phase = phase_f(params, x_2_temp, data.spins, data.atoms, data.charges)
            ratio = ((wave_x2_mag + 1.j * wave_x2_phase) / (wave_x1_mag + 1.j * wave_x1_phase)).real ** 2
            forward = jnp.sum(gauss ** 2)
            primal_x2, dgrad_f_x2 = jax.linearize(grad_f_closure, x_2_temp)
            phase_primal_x2, dgrad_phase_x2 = jax.linearize(grad_phase_closure, x_2_temp)

            O_new = primal_x2 + 1.j * phase_primal_x2
            O_new = jnp.reshape(O_new, (nelectrons, ndim))
            O_new_eff = O_new[i]
            #jax.debug.print("forward:{}", forward)
            #jax.debug.print("O_eff:{}", O_eff)
            #jax.debug.print("O_new_eff:{}", O_new_eff)
            backward = jnp.sum((gauss + (O_eff + O_new_eff) * tstep) ** 2)
            #jax.debug.print("backward:{}", backward)
            t_pro = jnp.exp(1 / (2 * tstep) * (forward - backward))
            #jax.debug.print("ratio:{}", ratio)
           #jax.debug.print("t_pro:{}", t_pro)
            ratio_total = jnp.abs(ratio) * t_pro
            # ratio_total = ratio_total * jnp.sign(ratio)
            rnd = jax.random.uniform(key, shape=ratio_total.shape, minval=0, maxval=1.0)
            cond = ratio_total > rnd
            #jax.debug.print("cond:{}", cond)
            x_new = x_new.at[i].set(jnp.where(cond, x2[i], x1[i]))

        x_new = jnp.reshape(x_new, (-1))
        #jax.debug.print("x_new:{}", x_new)
        data = networks.GaussianNetData(**(dict(data) | {'positions': x_new}))
        new_key, new_key2 = jax.random.split(key)
        return data, new_key2
    return mcstep





