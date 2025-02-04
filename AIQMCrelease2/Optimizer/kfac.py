import chex
import optax
import jax
import functools
from typing_extensions import Protocol
from typing import Optional, Tuple, Union
from AIQMCrelease2.wavefunction import nn
import kfac_jax
import jax.numpy as jnp
from AIQMCrelease2.Loss import pploss as qmc_loss_functions
from AIQMCrelease2 import constants

OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]

OptUpdateResults = Tuple[
    nn.ParamTree, Optional[OptimizerState], jnp.array, Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):
    def __call__(self, params: nn.ParamTree, data: nn.AINetData, opt_state: optax.OptState,
                 key: chex.PRNGKey) -> OptUpdateResults:
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
            key: chex.PRNGKey,
    ) -> StepResults:
        """Performs one set of MCMC moves and an optimization step."""


def make_kfac_training_step(
        damping: float,
        optimizer: kfac_jax.Optimizer,
        reset_if_nan: bool = False) -> Step:

    #mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(damping))
    copy_tree = constants.pmap(functools.partial(jax.tree_util.tree_map, lambda x: (1.0 * x).astype(x.dtype)))

    def step(
            data: nn.AINetData,
            params: nn.ParamTree,
            state: kfac_jax.Optimizer.State,
            key: chex.PRNGKey,) -> StepResults:
        """A full update iteration for KFAC: MCMC steps + optimization."""
        mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)

        if reset_if_nan:
            old_params = copy_tree(params)
            old_state = copy_tree(state)

        new_params, new_state, stats = optimizer.step(
            params=params,
            state=state,
            rng=loss_keys,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )

        if reset_if_nan and jnp.isnan(stats['loss']):
            new_params = old_params
            new_state = old_state
        return data, new_params, new_state, stats['loss'], stats['aux']

    return step
