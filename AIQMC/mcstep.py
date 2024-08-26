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

signednetwork, phasenetwork, logabsnetwork, batchnetwork, batchparams, data = main.main()
print("data.positions", data.positions)
print("params", batchparams)
key = jax.random.PRNGKey(seed=1)

def walkers_update(params: nn.ParamTree, batch_phase:nn.LogAINetLike, batch_f: nn.LogAINetLike, single_f:nn.LogAINetLike, data: nn.AINetData, key: chex.PRNGKey, tstep:float, ndim=3, blocks=2, batch_size=4, nelectrons=4):
    """we have atoms array in the data. So, we don't need this parameter again.
    This method can move walkers one step."""
    """Now, I have met the largest problem so far. I dont understand how the shape of data can be fitted by the gradient function of JAX.
    and we have one problem about the input shape. how to make the shape of the data class be compatible with the batched calculation.
    20.08.2024, it is really a big problem for us. Now, we have more and more bugs."""
    key, subkey = jax.random.split(key)
    x1 = data.positions
    spins = data.spins
    atoms = data.atoms
    charges = data.charges
    print("---------------------------")
    print("x1", x1)
    print("spins", spins)
    print("atoms", atoms)
    print("charges", charges)
    """propose move"""
    """now, we have two problems, one is about the sign, the other one is about the gradient calculation.23.08.2024. """
    #a = batch_f(params, x1, data.atoms, data.charges)
    phase_f = utils.select_output(single_f, 0)
    logabs_f = utils.select_output(single_f, 1)
    grad_value = jax.vmap(jax.grad(logabs_f, argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)
    phase_grad_value = jax.vmap(jax.grad(phase_f, argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)
    #jax.debug.print("a:{}", a)
    #jax.debug.print("grad_value:{}", grad_value)


    def grad_f_closure(x):
        return grad_value(params, x, data.atoms, data.charges)

    def phase_grad_f_closure(x):
        return phase_grad_value(params, x, data.atoms, data.charges)

    """26.08.2024, here we meet one problem about complex number. which one do we need calculate \partial log(|\psi|) or \partial log(\psi)
    as O(r).
    we need figure this out later."""
    primal_1, dgrad_f_1 = jax.linearize(grad_f_closure, x1)
    jax.debug.print("x1:{}", x1)
    gauss = np.random.normal(scale=tstep, size=(jnp.shape(x1)))
    primal_1 = jnp.reshape(primal_1, (jnp.shape(x1)))
    jax.debug.print("reshape_primal:{}", primal_1)
    x2 = x1 + gauss + primal_1*tstep
    jax.debug.print("x2:{}", x2)
    primal_2, dgrad_f_1 = jax.linearize(grad_f_closure, x2)
    primal_2 = jnp.reshape(primal_2, (jnp.shape(x2)))
    jax.debug.print("reshape_primal:{}", primal_2)
    forward = jnp.square(gauss)
    jax.debug.print("forward:{}", forward)
    backward = jnp.square(gauss + tstep * (primal_1 + primal_2))
    jax.debug.print("backward:{}", backward)
    t_probability = jnp.exp(1/(2 * tstep) * (forward - backward))
    jax.debug.print("t_probability:{}", t_probability)
    phase_1 = batch_phase(params, x1, data.atoms, data.charges)
    jax.debug.print("phase_1:{}", phase_1)
    phase_2 = batch_phase(params, x2, data.atoms, data.charges)
    value_1 = batch_f(params, x1, data.atoms, data.charges)
    value_2 = batch_f(params, x2, data.atoms, data.charges)
    ratio = phase_2*jnp.exp(value_1)/(phase_1*jnp.exp(value_2))
    """26.08.2024, we have more problem about the calculation of the ratio. We continue tomorrow."""
    ratio = jnp.square(jnp.abs(ratio))
    jax.debug.print("ratio:{}", ratio)
    """26.08.2024, here we need calculate the accept index according the array 'ratio'."""
    #accept_index =







output = walkers_update(params=batchparams, batch_phase=phasenetwork, batch_f=batchnetwork, single_f=signednetwork, tstep=0.1, data=data, key=key)



def make_mc_step( batch_network, batch_per_device, atoms: jnp.ndarray, steps=10,  ndim=3, blocks=2):
    inner_fun = walkers_update()