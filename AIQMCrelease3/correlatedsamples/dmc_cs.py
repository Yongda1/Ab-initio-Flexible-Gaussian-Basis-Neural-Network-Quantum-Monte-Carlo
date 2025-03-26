"""
1. we generate secondary walks from the reference walk according to the space-warp transformation.
2. In the averages, we retain the ratios of the secondary and primary wave functions as in VMC.
3. The secondary weights are the primary ones multiplied by the product of the factors exp[S_s(R^s', R^s, tau_s) - S(R',R, tau)] for the last N_proj generations.
N_proj is chosen large enough to project out the secondary ground state, but small enough to avoid a considerable increase in the fluctuations.
In the exponential factors, we introduced tau_s because the secondary moves are effectively proposed with a different time step, tau_s.
"""
import jax.numpy as jnp

def secondary_weights(primary_weights: jnp.array, tau_s: jnp.array, N_proj: int, R_prime: jnp.array, R: jnp.array,
                      ) -> jnp.array:
    """calculate the factors exp[S_s(R^s', R^s, tau_s) - S(R',R, tau)], however, before we do this calculation,
    we still store the DMC data correctly. 26.3.2025.
    we already rewrite the storing function of DMC modules. 26.3.2025"""