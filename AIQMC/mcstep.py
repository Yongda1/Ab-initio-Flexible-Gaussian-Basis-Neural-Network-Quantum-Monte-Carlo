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

batchnetwork, batchparams, data = main.main()
print("data.positions", data.positions)
print("params", batchparams)
key = jax.random.PRNGKey(seed=1)

def walkers_update(params: nn.ParamTree, f: nn.LogAINetLike, data: nn.AINetData, key: chex.PRNGKey, ndim=3, blocks=2, batch_size=4, nelectrons=4):
    """we have atoms array in the data. So, we don't need this parameter again."""
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
    """now, we have two problems, one is about the sign, the other one is about the gradient calcualtion.23.08.2024. """
    a = f(params, x1, data.atoms, data.charges)
    jax.debug.print("a:{}", a)
    grad_f = jax.grad(f, argnums=1)(params, data.positions, data.atoms, data.charges)


    #def grad_f_closure(x):
        #return grad_f(params, x, data.atoms, data.charges)

    """jax.linearize evaluate the JVP value."""
    #primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)
    #jax.debug.print("primal:{}", primal)
    #jax.debug.print("dgrad_f:{}", dgrad_f)
    #phase_f = utils.select_output(f, argnum=0)
    #logabs_f = utils.select_output(f, argnum=1)
    #print("phase_f", phase_f)
    #print("logabs_f", logabs_f)
    #grad_f = jax.grad(logabs_f, argnums=1)(params, x1, atoms, charges)
    """Now, I have met the largest problem so far. I dont understand how the shape of data can be fitted by the gradient function of JAX.
    and we have one problem about the input shape. how to make the shape of the data class be compatible with the batched calculation.
    20.08.2024, it is really a big problem for us. Now, we have more and more bugs."""

output = walkers_update(params=batchparams, f=batchnetwork, data=data, key=key)



def make_mc_step( batch_network, batch_per_device, atoms: jnp.ndarray, steps=10,  ndim=3, blocks=2):
    inner_fun = walkers_update()