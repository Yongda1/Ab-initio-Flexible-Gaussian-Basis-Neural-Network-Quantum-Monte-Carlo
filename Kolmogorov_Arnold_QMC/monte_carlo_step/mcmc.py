import chex
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.kan_networks_case_one as networks


def _harmonic_mean(x, atoms):
  """Calculates the harmonic mean of each electron distance to the nuclei.

  Args:
    x: electron positions. Shape (batch, nelectrons, 1, ndim). Note the third
      dimension is already expanded, which allows for avoiding additional
      reshapes in the MH algorithm.
    atoms: atom positions. Shape (natoms, ndim)

  Returns:
    Array of shape (batch, nelectrons, 1, 1), where the (i, j, 0, 0) element is
    the harmonic mean of the distance of the j-th electron of the i-th MCMC
    configuration to all atoms.
  """
  ae = x - atoms[None, ...]
  r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
  return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)

def _log_prob_gaussian(x, mu, sigma):
  """Calculates the log probability of Gaussian with diagonal covariance.

  Args:
    x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
    mu: means of Gaussian distribution. Same shape as or broadcastable to x.
    sigma: standard deviation of the distribution. Same shape as or
      broadcastable to x.

  Returns:
    Log probability of Gaussian distribution with shape as required for
    mh_update - (batch, nelectron, 1, 1).
  """
  numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
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

def mh_update(params,
              f,
              data,
              key: chex.PRNGKey,
              lp_1,
              num_accepts,
              stddev=0.02,
              atoms=None,
              ndim=3,
              blocks=1,
              i=0):
    """f: log value of wavefunction."""
    del i, blocks  # electron index ignored for all-electron moves
    key, subkey = jax.random.split(key)
    x1 = data.positions
    #jax.debug.print("x1:{}", x1)
    if atoms is None:  # symmetric proposal, same stddev everywhere
        x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
        lp_2 = 2.0 * f(
            params, x2, data.spins, data.atoms, data.charges
        )  # log prob of proposal
        ratio = lp_2 - lp_1
    else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
        n = x1.shape[0]
        x1 = jnp.reshape(x1, [n, -1, 1, ndim])
        hmean1 = _harmonic_mean(x1, atoms)  # harmonic mean of distances to nuclei

        x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
        lp_2 = 2.0 * f(
            params, x2, data.spins, data.atoms, data.charges
        )  # log prob of proposal
        hmean2 = _harmonic_mean(x2, atoms)  # needed for probability of reverse jump

        lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)  # forward probability
        lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)  # reverse probability
        ratio = lp_2 + lq_2 - lp_1 - lq_1

        x1 = jnp.reshape(x1, [n, -1])
        x2 = jnp.reshape(x2, [n, -1])
    x_new, key, lp_new, num_accepts = mh_accept(
        x1, x2, lp_1, lp_2, ratio, key, num_accepts)
    #jax.debug.print("num_accepts:{}", num_accepts)
    new_data = networks.KANetsData(**(dict(data) | {'positions': x_new}))
    return new_data, key, lp_new, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   atoms=None,
                   nidm=3,
                   blocks=1,):
    """here, data is batched."""
    inner_fun = mh_update
    def mcmc_step(params, data, key, width):
        pos = data.positions
        nsteps = steps * blocks
        #jax.debug.print("nsteps:{}", nsteps)
        logprob = 2.0 * batch_network(params, pos, data.spins, data.atoms, data.charges)
        #jax.debug.print("logprob:{}", logprob)
        """it is kind of stupid. i hate loop. However, currently it is working."""
        for i in range(steps):
            data, key, logprob, num_accepts = mh_update(params, batch_network, data, key, logprob, 0.0, )
        #jax.debug.print("new_data:{}", data)
        return data

    return mcmc_step

