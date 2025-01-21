"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
from AIQMC import nn
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from AIQMC import main
from AIQMC.utils import utils
"""Tomorrow, we are going to finish the walkers moving part. But differently from FermiNet, we will use the traditional moving strategy.
19.08.2024. no worry, everything will fine."""

signed_network, data_non_batch, batch_params, non_batch_network, non_phase_network = main.main()
print("data_non_batch", data_non_batch)
print("params", batch_params)
key = jax.random.PRNGKey(seed=1)


def walkers_accept(x1, x2, ratio, key):
    print("---------------------------")
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=ratio.shape, minval=0, maxval=1.0)
    #jax.debug.print("ratio:{}", ratio)
    #jax.debug.print("rnd:{}", rnd)
    cond = ratio > rnd
    #jax.debug.print("cond:{}", cond)
    cond = jnp.reshape(cond, (1, 4, 1))
    #jax.debug.print("cond:{}", cond)
    #jax.debug.print("x1:{}", x1)
    #jax.debug.print("x2:{}", x2)
    x_new = jnp.where(cond, x2, x1)
    return x_new, subkey



def walkers_update(params: nn.ParamTree, batch_phase:nn.LogAINetLike, batch_f: nn.LogAINetLike, single_f:nn.LogAINetLike,
                   data: nn.AINetData, key: chex.PRNGKey, tstep:float, ndim=3, blocks=2, batch_size=4, nelectrons=4, i=0):
    """we have atoms array in the data. So, we don't need this parameter again.
    This method can move walkers one step."""
    """Now, I have met the largest problem so far. I dont understand how the shape of data can be fitted by the gradient function of JAX.
    and we have one problem about the input shape. how to make the shape of the data class be compatible with the batched calculation.
    20.08.2024, it is really a big problem for us. Now, we have more and more bugs."""
    key, subkey = jax.random.split(key)
    x1 = data.positions
    #spins = data.spins
    #atoms = data.atoms
    #charges = data.charges
    print("---------------------------")
    """propose move"""
    """now, we have two problems, one is about the sign, the other one is about the gradient calculation.23.08.2024. """
    #a = batch_f(params, x1, data.atoms, data.charges)
    phase_f = utils.select_output(single_f, 0)
    logabs_f = utils.select_output(single_f, 1)
    grad_value = jax.vmap(jax.grad(logabs_f, argnums=1), in_axes=(None, 1, None, None), out_axes=0)
    phase_grad_value = jax.vmap(jax.grad(phase_f, argnums=1), in_axes=(None, 1, None, None), out_axes=0)
    #jax.debug.print("a:{}", a)
    #jax.debug.print("grad_value:{}", grad_value)


    def grad_f_closure(x):
        return grad_value(params, x, data.atoms, data.charges)

    #def phase_grad_f_closure(x):
    #    return phase_grad_value(params, x, data.atoms, data.charges)

    """26.08.2024, here we meet one problem about complex number. which one do we need calculate \partial log(|\psi|) or \partial log(\psi)
    as O(r).
    we need figure this out later.
    27.08.2024, now we are sure that \partial log(\psi) should be used, i.e. quantum velocity. Therefore, we dont need 
    the gradient of phase currently. Our strategy for calculating the gradient of positions is still working."""
    primal_1, dgrad_f_1 = jax.linearize(grad_f_closure, x1)
    #jax.debug.print("x1:{}", x1)
    gauss = np.random.normal(scale=tstep, size=(jnp.shape(x1)))
    primal_1 = jnp.reshape(primal_1, (jnp.shape(x1)))
    #jax.debug.print("reshape_primal:{}", primal_1)
    """29.08.2024, we need think how to add minus velocity into this movement."""
    x2 = x1 + gauss + primal_1*tstep
    #jax.debug.print("x2:{}", x2)
    primal_2, dgrad_f_1 = jax.linearize(grad_f_closure, x2)
    primal_2 = jnp.reshape(primal_2, (jnp.shape(x2)))
    #jax.debug.print("reshape_primal:{}", primal_2)
    forward = jnp.square(gauss)
    #jax.debug.print("forward:{}", forward)
    backward = jnp.square(gauss + tstep * (primal_1 + primal_2))
    #jax.debug.print("backward:{}", backward)
    t_probability = jnp.exp(1/(2 * tstep) * (forward - backward))
    #jax.debug.print("t_probability:{}", t_probability)
    phase_1 = batch_phase(params, x1, data.atoms, data.charges)
    #jax.debug.print("phase_1:{}", phase_1)
    jax.debug.print("x1:{}", x1)
    phase_2 = batch_phase(params, x2, data.atoms, data.charges)
    value_1 = batch_f(params, x1, data.atoms, data.charges)
    jax.debug.print("value_1:{}", value_1)
    value_2 = batch_f(params, x2, data.atoms, data.charges)
    ratio = phase_2*jnp.exp(value_1)/(phase_1*jnp.exp(value_2))
    """26.08.2024, we have more problem about the calculation of the ratio. We continue tomorrow."""
    ratio = jnp.square(jnp.abs(ratio))
    "here, the shape of the array is number of batch_size, the number of electrons and the dimensions."
    t_probability = jnp.sum(jnp.sum(jnp.reshape(t_probability, (4, 4, 3)), axis=-1), axis=-1)
    #jax.debug.print("ratio:{}", ratio)
    #jax.debug.print("t_probability:{}", t_probability)
    ratio = ratio*t_probability
    #jax.debug.print("ratio:{}", ratio)
    """26.08.2024, here we need calculate the accept index according the array 'ratio'."""
    """now, we need do the function of walkers accepting. 27.08.2024."""
    x_new, next_key = walkers_accept(x1=x1, x2=x2, ratio=ratio, key=subkey)
    new_data = nn.AINetData(**(dict(data) | {'positions': x_new}))
    #jax.debug.print("new_data:{}", new_data)
    return new_data, next_key


#new_data, next_key = walkers_update(params=batchparams, batch_phase=phasenetwork, batch_f=batchnetwork, single_f=signednetwork, tstep=0.1, data=data, key=key)
#jax.debug.print("new_data:{}", new_data)
#output1 = walkers_accept(x1=x1, x2=x2, ratio=ratio, key=key)
"""28.08.2024, we have one problem about the loop calculation. we are not sure about how to do the loop. Our one step calculation is working well. But the loop not. I dont understand.
We are still not sure which part is wrong, nn_wrong.py or this loop?"""

#new_data, next_key = walkers_update(params=batchparams, batch_phase=phasenetwork, batch_f=batchnetwork, single_f=signednetwork, tstep=0.1, data=data, key=key)


def make_mc_step(phasenetwork, batchnetwork, signednetwork, nsteps=10):

    @jax.jit
    def mcmc_step(params: nn.ParamTree, data, key):
        def step_fn(i, x):
            return walkers_update(params, phasenetwork, batchnetwork, signednetwork, *x, tstep=0.1, i=i)

        new_data, key = lax.fori_loop(lower=0, upper=nsteps, body_fun=step_fn, init_val=(data, key))
        #jax.debug.print("new_data:{}", new_data)
        return new_data

    return mcmc_step


walker_move = make_mc_step(non_phase_network, non_batch_network, signed_network, nsteps=10)
newdata = walker_move(params=batch_params, data=data_non_batch, key=key)