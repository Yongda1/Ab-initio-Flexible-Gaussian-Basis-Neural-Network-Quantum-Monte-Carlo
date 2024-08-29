"""Evaluates the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union
import chex
from AIQMC import nn
from AIQMC import pseudopotential as pp
from AIQMC.utils import utils
#import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol
from AIQMC import main


signednetwork, data, batchparams = main.main()
print("data.positions", data.positions)
#print("params", batchparams)
key = jax.random.PRNGKey(seed=1)


Array = Union[jnp.ndarray, np.ndarray]

class LocalEnergy(Protocol):
    def __call__(self, params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) \
            -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Returns the local energy of a Hamiltonian at a configuration."""


class MakeLocalEnergy(Protocol):
    def __call__(self, f: nn.AINetLike, charges: jnp.ndarray, nspins: Sequence[int], use_scan: bool = False,
                 complex_output: bool = False, **kwargs: Any) -> LocalEnergy:
        """Builds the LocalEnergy function."""


KineticEnergy = Callable[[nn.ParamTree, nn.AINetData], jnp.ndarray]


def local_kinetic_energy(f: nn.AINetLike, use_scan: bool = False, complex_output: bool = False, laplacian_method: str = 'default') -> KineticEnergy:
    """Create the function for the local kinetic energy, -1/2 \nabla^2 ln|f|.
    29.08.2024 here our codes will be completely different from other codes due to the introduction of angular functions.
    I need take some notes on my slides.
    29.08.2024 I dont understand angular functions, complex number? how to calculate kinetic energy by real number? Why is it a real number?"""
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)
    if laplacian_method == 'default':
        def _lapl_over_f(params, data):
            #grad_f = jax.grad(logabs_f, argnums=1)
            #grad_phase = jax.grad(phase_f, argnums=1)
            """29.08.2024 take care of the following function, we need write argnums=1 two times."""
            second_grad_value = jax.vmap(jax.jacfwd(jax.jacrev(logabs_f, argnums=1), argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)
            hessian_value_logabs = second_grad_value(params, data.positions, data.atoms, data.charges)
            jax.debug.print("hessian_value_logabs:{}", hessian_value_logabs)
            grad_phase = jax.vmap(jax.vjp(phase_f), in_axes=(None, 1, 1, 1), out_axes=0)
            def grad_phase_closure(x):
                return grad_phase(params, x, data.atoms, data.charges)
            phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, data.positions)
            jax.debug.print("phase_primal:{}", phase_primal)
            #jax.debug.print("shape_of_hessian:{}", hessian_value_logabs.shape)
            #phase_grad_value = jax.vmap(jax.grad(phase_f, argnums=1), in_axes=(None, 1, 1, 1), out_axes=0)

            #def grad_f_closure(x):
            #    return grad_value(params, x, data.atoms, data.charges)

            #def grad_phase_closure(x):
            #    return phase_grad_value(params, x, data.atoms, data.charges)

            #primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)
            #phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, data.positions)
            #jax.debug.print("primal:{}", primal)
            #jax.debug.print("dgrad_f:{}", dgrad_f)
            #jax.debug.print("phase_primal:{}", phase_primal)
            #jax.debug.print("dgrad_phase:{}", dgrad_phase)


    return _lapl_over_f

lap_over_f = local_kinetic_energy(signednetwork)
output = lap_over_f(batchparams, data)