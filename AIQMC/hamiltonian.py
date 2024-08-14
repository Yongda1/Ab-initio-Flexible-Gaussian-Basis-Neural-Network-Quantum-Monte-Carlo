"""Evaluates the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union
import chex
from AIQMC import nn
from AIQMC import pseudopotetential as pp
from AIQMC.utils import utils
import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol

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
    """Create the function for the local kinetic energy, -1/2 \nabla^2 ln|f|."""
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)
    if laplacian_method == 'default':
        def _lapl_over_f(params, data):
            n = data.positions.shape[0]
