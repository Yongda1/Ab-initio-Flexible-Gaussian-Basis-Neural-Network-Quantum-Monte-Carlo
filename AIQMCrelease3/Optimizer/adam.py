from AIQMCrelease3.wavefunction_Ynlm import nn
import jax
import optax
import jax.numpy as jnp
import chex
import functools
from typing_extensions import Protocol
from typing import Optional, Mapping, Sequence, Tuple, Union
import kfac_jax
from AIQMCrelease3.Loss import pploss as qmc_loss_functions
from AIQMCrelease3 import constants


OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[nn.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]

class OptUpdate(Protocol):

  def __call__(
      self,
      params: nn.ParamTree,
      data: nn.AINetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly."""


StepResults = Tuple[
    nn.AINetData,
    nn.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData]

class Step(Protocol):

  def __call__(
      self,
      data: nn.AINetData,
      params: nn.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,) -> StepResults:
    """Performs one set of MCMC moves and an optimization step."""


def make_opt_update_step(evaluate_loss,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
    loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

    def opt_update(params: nn.ParamTree, data: nn.AINetData, opt_state: Optional[optax.OptState], key: chex.PRNGKey) -> OptUpdateResults:
        (loss, aux_data), grad = loss_and_grad(params, key, data)
        grad = constants.pmean(grad)
        #jax.debug.print("grad:{}", grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux_data
    return opt_update


def make_training_step(optimizer_step: OptUpdate) -> Step:
    @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
    def step(data: nn.AINetData,
             params: nn.ParamTree,
             state: Optional[optax.OptState],
             key: chex.PRNGKey,) -> StepResults:
        mcmc_key, loss_key = jax.random.split(key, num=2)
        new_params, new_state, loss, aux_data = optimizer_step(params,
                                                               data,
                                                               state,
                                                               loss_key)
        '''
        new_params = jax.lax.cond(jnp.isnan(loss),
                                  lambda: params,
                                  lambda: new_params)
        new_state = jax.lax.cond(jnp.isnan(loss),
                                 lambda: state,
                                 lambda: new_state)
        '''
        return data, new_params, new_state, loss, aux_data
    return step

