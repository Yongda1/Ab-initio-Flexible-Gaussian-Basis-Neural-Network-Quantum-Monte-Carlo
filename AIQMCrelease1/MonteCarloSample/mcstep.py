"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
from AIQMCrelease1.wavefunction import nn
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from AIQMCrelease1.main import main_adam
from AIQMCrelease1.utils import utils
import kfac_jax
"""Tomorrow, we are going to finish the walkers moving part. But differently from FermiNet, we will use the traditional moving strategy.
19.08.2024. no worry, everything will fine."""
"""due to the parallel problem about the optimizer, the walkers should move like non-batch version. Without pmap. 1.11.2024. This also means that
we cannot test the codes before we finish it."""
"""we have to think in this way. The mcstep function has been done, pmap, So, the data.pos also has been mapped to different
devices. so we can remove pmap in front of batch_network and batch_phase."""
"""we need rewrite this part. 22.12.2024. to be continued..."""

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
                   tstep: float,
                   ndim: int,
                   batch_size: int,
                   nelectrons: int,
                   i: int):
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
    t_probability = jnp.sum(jnp.sum(jnp.reshape(t_probability, (batch_size, nelectrons, ndim)), axis=-1), axis=-1)
    ratio = ratio * t_probability
    x_new, next_key = walkers_accept(x1=x1, x2=x2, ratio=ratio, key=key)
    new_data = nn.AINetData(**(dict(data) | {'positions': x_new}))
    "the following line for making the input and output have the same shape. I know it is not good. But it is working."
    "4 is batch size. 12 is the number of position."
    new_data.positions = jnp.reshape(new_data.positions, (batch_size, 48))
    return new_data, next_key


def make_mc_step(signednetwork, nsteps=10):
    @jax.jit
    def mcmc_step(params: nn.ParamTree,
                  data: nn.AINetData,
                  key,
                  tstep=0.1,
                  ndim=3,
                  batch_size=4,
                  nelectrons=16,):
        def step_fn(i, x):
            return walkers_update(params, signednetwork, *x, tstep=tstep, ndim=ndim, batch_size=batch_size, nelectrons=nelectrons, i=i)

        new_data, key = lax.fori_loop(lower=0, upper=nsteps, body_fun=step_fn, init_val=(data, key))
        #jax.debug.print("new_data:{}", new_data)
        return new_data
    #jax.debug.print("data:{}", data)
    return mcmc_step


jax.debug.print("data:{}", data)
mcstep = make_mc_step(signed_network, nsteps=10)
mcstep_pmap = jax.pmap(mcstep, in_axes=(0, 0, 0, None, None, None, None, None), out_axes=0, donate_argnums=1)
data = mcstep_pmap(params=batch_params,
                   data=data,
                   key=subkeys,
                   tstep=0.1,
                   ndim=3,
                   batch_size=4,
                   nelectrons=16,
                   i=0)
jax.debug.print("data:{}", data)

