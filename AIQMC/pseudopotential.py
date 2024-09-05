"""Evaluates the pseudopotential Hamiltonian on a wavefunction. 04.09.2024."""

from typing import Sequence
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


r_ae = jnp.array([[[0.21824889, 0.3565338], [0.1946077, 0.32006422], [0.4780831,  0.138754], [0.41992992, 0.19055614]],
                 [[0.16530964, 0.29526055], [0.15191387, 0.22501956], [0.3564806, 0.05262673], [0.45009968, 0.16455044]],
                 [[0.35471296, 0.65752304], [0.08244702, 0.36039594], [0.48147705, 0.13537169], [0.1520589, 0.22781217]],
                 [[0.08920264, 0.26871547], [0.20597123, 0.25272587], [0.23355496, 0.22838382], [0.32041857, 0.20322587]]])


rn_local = jnp.array([1, 3, 2])
rn_non_local = jnp.array([2])
local_coefficient = jnp.array([4.00000, 57.74008, -25.81955])
nonlocal_coefficient = jnp.array([52.13345])
local_exponent = jnp.array([14.43502, 8.39889, 7.38188])
nonlocal_exponent = jnp.array([7.76079])


def get_v_l(r_ae: jnp.array, rn_local: jnp.array, rn_non_local: jnp.array,
                           local_coefficient: jnp.array, local_exponent: jnp.array,
                           non_local_coefficient: jnp.array, non_local_exponent: jnp.array,
                           symbols: Sequence[str]):
    """here, we are not going to write general codes. Currently, we only implement the C atom."""
    jax.debug.print("r_ae:{}", r_ae)
    rn_local = rn_local -2
    jax.debug.print("rn_local:{}", rn_local)
    """here, we need match the shape of rn_local with r_ae, coefficient, exponent.05.09.2024."""

output = get_v_l(r_ae=r_ae, rn_local=rn_local, rn_non_local=rn_non_local, local_coefficient=local_coefficient,
                 local_exponent=local_exponent, non_local_coefficient=nonlocal_coefficient,
                 non_local_exponent=nonlocal_exponent, symbols=['C', 'C'])


def ecp_ea(r_ae: jnp.array, batch_size:int, charges: jnp.array, symbols: Sequence[str], quad_degree: int=4, ecp: str='ccecp', complex_output: bool=True,):
    """we contiune this tomorrow. 04.09.2024.
    here, we use the method in Pyqmc to do pseduopotential calculation. Later, we can use grid to discrete the pseudopotential."""
    output = get_v_l(r_ae, symbols)