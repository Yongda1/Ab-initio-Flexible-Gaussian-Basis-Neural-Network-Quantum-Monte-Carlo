"""This moudle tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
from AIQMCrelease1.wavefunction_f import nn
from AIQMCrelease1 import constants
import numpy as np
import jax
from jax import numpy as jnp
from jax import lax


def _harmonic_mean(x, atoms):
    """Calculates the harmonic mean of each electron distance to the nuclei."""
    ae = x - atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
    return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)


def _log_prob_gaussian(x, mu, sigma):
    """Calculates the log probability of Gaussian with diagonal covariance."""
    numer = jnp.sum(-0.5 * ((x - mu) ** 2) / (sigma ** 2), axis=[1, 2, 3])
    denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
    return numer - denom


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
    """Given state, proposal, and probabilities, execute MH accept/reject step."""
    key, subkey = jax.random.split(key)
    rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
    cond = ratio > rnd
    x_new = jnp.where(cond[..., None], x2, x1)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += jnp.sum(cond)
    return x_new, key, lp_new, num_accepts


def mh_update(
        params: nn.ParamTree,
        f: nn.AINetLike,
        data: nn.AINetData,
        key: chex.PRNGKey,
        lp_1,
        num_accepts,
        stddev=0.02,
        atoms=None,
        ndim=3,
        blocks=1,
        i=0, ):
    del i, blocks  # electron index ignored for all-electron moves
    key, subkey = jax.random.split(key)
    x1 = data.positions
    n = x1.shape[0]
    x1 = jnp.reshape(x1, [n, -1, 1, ndim])
    hmean1 = _harmonic_mean(x1, atoms)
    x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
    lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
    hmean2 = _harmonic_mean(x2, atoms)
    lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)
    lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)
    ratio = lp_2 + lq_2 - lp_1 - lq_1
    x1 = jnp.reshape(x1, [n, -1])
    x2 = jnp.reshape(x2, [n, -1])
    x_new, key, lp_new, num_accepts = mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts)
    new_data = nn.AINetData(**(dict(data) | {'positions': x_new}))
    return new_data, key, lp_new, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   atoms=None,
                   ndim=3,
                   blocks=1):
    """Creates the MCMC step function."""
    inner_fun = mh_update

    @jax.jit
    def mcmc_step(params, data, key, width):
        """Performs a set of MCMC steps."""
        pos = data.positions

        def step_fn(i, x):
            return inner_fun(
                params,
                batch_network,
                *x,
                stddev=width,
                atoms=atoms,
                ndim=ndim,
                blocks=blocks,
                i=i)

        nsteps = steps * blocks
        logprob = 2.0 * batch_network(
            params, pos, data.spins, data.atoms, data.charges
        )
        new_data, key, _, num_accepts = lax.fori_loop(
            0, nsteps, step_fn, (data, key, logprob, 0.0)
        )
        pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
        pmove = constants.pmean(pmove)
        return new_data, pmove

    return mcmc_step


def update_mcmc_width(
        t: int,
        width: jnp.ndarray,
        adapt_frequency: int,
        pmove: jnp.ndarray,
        pmoves: np.ndarray,
        pmove_max: float = 0.55,
        pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
    t_since_mcmc_update = t % adapt_frequency
    # update `pmoves`; `pmove` should be the same across devices
    pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
    if t > 0 and t_since_mcmc_update == 0:
        if np.mean(pmoves) > pmove_max:
            width *= 1.1
        elif np.mean(pmoves) < pmove_min:
            width /= 1.1
    return width, pmoves


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
