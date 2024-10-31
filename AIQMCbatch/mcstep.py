"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
from AIQMCbatch import nn
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from AIQMCbatch import main_kfac
from AIQMCbatch.utils import utils
"""Tomorrow, we are going to finish the walkers moving part. But differently from FermiNet, we will use the traditional moving strategy.
19.08.2024. no worry, everything will fine."""

#signed_network, data, batch_params, batch_network, batch_phase_network = main_kfac.main()
#key = jax.random.PRNGKey(seed=1)


def walkers_accept(x1, x2, ratio, key):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=ratio.shape, minval=0, maxval=1.0)
    cond = ratio > rnd
    cond = jnp.reshape(cond, (1, 4, 1))
    x_new = jnp.where(cond, x2, x1)
    return x_new, subkey



def walkers_update(params: nn.ParamTree, batch_phase:nn.LogAINetLike, batch_f: nn.LogAINetLike, single_f:nn.LogAINetLike,
                   data: nn.AINetData, key: chex.PRNGKey, tstep:float, ndim=3, blocks=2, batch_size=4, nelectrons=4, i=0):
    """params: batch_params"""
    key, subkey = jax.random.split(key)
    x1 = data.positions
    #phase_f = utils.select_output(single_f, 0)
    logabs_f = utils.select_output(single_f, 1)
    grad_value = jax.pmap(jax.vmap(jax.grad(logabs_f, argnums=1), in_axes=(None, 0, 0, 0), out_axes=0), in_axes=0, out_axes=0)
    #phase_grad_value = jax.vmap(jax.grad(phase_f, argnums=1), in_axes=(None, 1, None, None), out_axes=0)

    def grad_f_closure(x):
        return grad_value(params, x, data.atoms, data.charges)

    #def phase_grad_f_closure(x):
    #    return phase_grad_value(params, x, data.atoms, data.charges)

    primal_1, dgrad_f_1 = jax.linearize(grad_f_closure, x1)
    gauss = np.random.normal(scale=tstep, size=(jnp.shape(x1)))
    primal_1 = jnp.reshape(primal_1, (jnp.shape(x1)))
    x2 = x1 + gauss + primal_1*tstep
    primal_2, dgrad_f_1 = jax.linearize(grad_f_closure, x2)
    primal_2 = jnp.reshape(primal_2, (jnp.shape(x2)))
    forward = jnp.square(gauss)
    backward = jnp.square(gauss + tstep * (primal_1 + primal_2))
    t_probability = jnp.exp(1/(2 * tstep) * (forward - backward))
    phase_1 = batch_phase(params, x1, data.atoms, data.charges)
    phase_2 = batch_phase(params, x2, data.atoms, data.charges)
    value_1 = batch_f(params, x1, data.atoms, data.charges)
    value_2 = batch_f(params, x2, data.atoms, data.charges)
    ratio = phase_2*jnp.exp(value_1)/(phase_1*jnp.exp(value_2))
    ratio = jnp.square(jnp.abs(ratio))
    "here, the shape of the array is number of batch_size, the number of electrons and the dimensions."
    t_probability = jnp.sum(jnp.sum(jnp.reshape(t_probability, (4, 4, 3)), axis=-1), axis=-1)
    ratio = ratio*t_probability
    x_new, next_key = walkers_accept(x1=x1, x2=x2, ratio=ratio, key=subkey)
    new_data = nn.AINetData(**(dict(data) | {'positions': x_new}))
    return new_data, next_key


def make_mc_step(phasenetwork, batchnetwork, signednetwork, nsteps=10):

    @jax.jit
    def mcmc_step(params: nn.ParamTree, data, key):
        def step_fn(i, x):
            return walkers_update(params, phasenetwork, batchnetwork, signednetwork, *x, tstep=0.1, i=i)

        new_data, key = lax.fori_loop(lower=0, upper=nsteps, body_fun=step_fn, init_val=(data, key))
        #jax.debug.print("new_data:{}", new_data)
        return new_data

    return mcmc_step


#walker_move = make_mc_step(batch_phase_network, batch_network, signed_network, nsteps=10)
#newdata = walker_move(params=batch_params, data=data, key=key)