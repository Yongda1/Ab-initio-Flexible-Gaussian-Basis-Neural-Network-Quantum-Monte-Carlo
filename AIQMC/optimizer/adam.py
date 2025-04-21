from AIQMC import constants
import optax
from typing import Optional, Mapping, Sequence, Tuple, Union
import kfac_jax
from AIQMC.wavefunction import networks
import jax.numpy as jnp
from typing_extensions import Protocol
from AIQMC.loss_function import loss as qmc_loss_functions
import chex
import jax
import functools


OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.FermiNetData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def null_update(
    params: networks.ParamTree,
    data: networks.FermiNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)
    #data, pmove = mcmc_step(params, data, mcmc_key)

    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data, pmove

  return step