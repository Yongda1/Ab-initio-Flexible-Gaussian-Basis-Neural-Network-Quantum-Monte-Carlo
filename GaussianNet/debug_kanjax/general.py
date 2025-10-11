import jax
import jax.numpy as jnp


@jax.jit
def solve_single_lstsq(A_single, B_single):
    AtA = jnp.dot(A_single.T, A_single)
    AtB = jnp.dot(A_single.T, B_single)
    single_solution = jax.scipy.linalg.solve(AtA, AtB, assume_a='pos')
    return single_solution



@jax.jit
def solve_full_lstsq(A_full, B_full):
    solve_full = jax.vmap(solve_single_lstsq, in_axes=(0, 0))
    full_solution = solve_full(A_full, B_full)
    return full_solution
