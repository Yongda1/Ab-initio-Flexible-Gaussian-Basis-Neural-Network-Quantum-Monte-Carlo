"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
from AIQMCbatch2 import nn
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
#from AIQMCbatch2 import main_kfac
from AIQMCbatch2.utils import utils
#import kfac_jax
"""Tomorrow, we are going to finish the walkers moving part. But differently from FermiNet, we will use the traditional moving strategy.
19.08.2024. no worry, everything will fine."""
"""due to the parallel problem about the optimizer, the walkers should move like non-batch version. Without pmap. 1.11.2024. This also means that
we cannot test the codes before we finish it."""
"""we have to think in this way. The mcstep function has been done, pmap, So, the data.pos also has been mapped to different
devices. so we can remove pmap in front of batch_network and batch_phase."""


#signed_network, data, batch_params, batch_network, batch_phase_network = main_kfac.main()
#key = jax.random.PRNGKey(seed=1)
#sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
#sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)


def walkers_accept(x1, x2, ratio, key):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=ratio.shape, minval=0, maxval=1.0)
    cond = ratio > rnd
    cond = jnp.reshape(cond, (1, 4, 1))
    x_new = jnp.where(cond, x2, x1)
    return x_new, subkey


def walkers_update(params: nn.ParamTree,
                   single_f: nn.AINetLike,
                   data: nn.AINetData,
                   key: chex.PRNGKey,
                   tstep=0.1, ndim=3, blocks=2, batch_size=4, nelectrons=4,
                   i=0):
    """params: batch_params"""
    key, subkey = jax.random.split(key)
    x1 = data.positions
    logabs_f = utils.select_output(single_f, 1)
    sign_f = utils.select_output(single_f, 0)
    batch_logabs_f = jax.vmap(logabs_f, in_axes=(None, 0, 0, 0), out_axes=0)
    batch_sign_f = jax.vmap(sign_f, in_axes=(None, 0, 0, 0), out_axes=0)
    grad_value = jax.vmap(jax.grad(logabs_f, argnums=1), in_axes=(None, 0, 0, 0), out_axes=0)

    # phase_grad_value = jax.vmap(jax.grad(phase_f, argnums=1), in_axes=(None, 1, None, None), out_axes=0)

    def grad_f_closure(x):
        return grad_value(params, x, data.atoms, data.charges)

    # def phase_grad_f_closure(x):
    #    return phase_grad_value(params, x, data.atoms, data.charges)

    primal_1, dgrad_f_1 = jax.linearize(grad_f_closure, x1)
    gauss = np.random.normal(scale=tstep, size=(jnp.shape(x1)))
    primal_1 = jnp.reshape(primal_1, (jnp.shape(x1)))
    x2 = x1 + gauss + primal_1 * tstep
    primal_2, dgrad_f_1 = jax.linearize(grad_f_closure, x2)
    primal_2 = jnp.reshape(primal_2, (jnp.shape(x2)))
    forward = jnp.square(gauss)
    backward = jnp.square(gauss + tstep * (primal_1 + primal_2))
    t_probability = jnp.exp(1 / (2 * tstep) * (forward - backward))
    phase_1 = batch_sign_f(params, x1, data.atoms, data.charges)
    phase_2 = batch_sign_f(params, x2, data.atoms, data.charges)
    value_1 = batch_logabs_f(params, x1, data.atoms, data.charges)
    value_2 = batch_logabs_f(params, x2, data.atoms, data.charges)
    ratio = phase_2 * jnp.exp(value_1) / (phase_1 * jnp.exp(value_2))
    ratio = jnp.square(jnp.abs(ratio))
    "here, the shape of the array is number of batch_size, the number of electrons and the dimensions."
    t_probability = jnp.sum(jnp.sum(jnp.reshape(t_probability, (4, 4, 3)), axis=-1), axis=-1)
    ratio = ratio * t_probability
    x_new, next_key = walkers_accept(x1=x1, x2=x2, ratio=ratio, key=key)
    new_data = nn.AINetData(**(dict(data) | {'positions': x_new}))
    "the following line for making the input and output have the same shape. I know it is not good. But it is working." \
    "4 is batch size. 12 is the number of position."
    new_data.positions = jnp.reshape(new_data.positions, (4, 12)) 
    return new_data, next_key


def make_mc_step(signednetwork, nsteps=10):
    @jax.jit
    def mcmc_step(params: nn.ParamTree, data: nn.AINetData, key):
        def step_fn(i, x):
            return walkers_update(params, signednetwork, *x, tstep=0.1, i=i)

        new_data, key = lax.fori_loop(lower=0, upper=nsteps, body_fun=step_fn, init_val=(data, key))
        #jax.debug.print("new_data:{}", new_data)
        return new_data
    #jax.debug.print("data:{}", data)
    return mcmc_step


#jax.debug.print("data:{}", data)
#mcstep = make_mc_step(signed_network, nsteps=10)
#mcstep_pmap = jax.pmap(mcstep, donate_argnums=1)
#data = mcstep_pmap(batch_params, data, subkeys)
#jax.debug.print("data:{}", data)

