"""Evaluates the pseudopotential Hamiltonian on a wavefunction. 04.09.2024."""

from typing import Sequence
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

def make_pp_potential(charges: jnp.array, symbols: Sequence[str], quad_degree: int=4, ecp: str='ccecp', complex_output: bool=True):
    """we contiune this tomorrow. 04.09.2024."""