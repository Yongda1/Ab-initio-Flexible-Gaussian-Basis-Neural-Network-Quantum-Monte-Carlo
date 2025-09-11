import jax
import jax.numpy as jnp
import math
import itertools

@jax.jit
def generate_sub_blocks(indices: jnp.array, orbitals_uu: jnp.array):
    output = jnp.concatenate([jnp.array([orbitals_uu[indices[0]]]), jnp.array([orbitals_uu[indices[1]]])], axis=0)
    return output

generate_sub_blocks_vmap = jax.vmap(generate_sub_blocks, in_axes=(0, None))

@jax.jit
def generate_sub_sub_blocks(indices: jnp.array, orbitals_uu_sub: jnp.array):
    output = jnp.concatenate([jnp.array([orbitals_uu_sub[:, indices[0]]]), jnp.array([orbitals_uu_sub[:, indices[1]]])], axis=0)
    return output

generate_sub_sub_blocks_vmap = jax.vmap(jax.vmap(generate_sub_sub_blocks, in_axes=(0, None)), in_axes=(None, 0))

@jax.jit
def calculate_determinant(sub_sub_sub_blocks: jnp.array):
    output = jnp.linalg.det(sub_sub_sub_blocks)
    return output

calculate_determinant_vmap = jax.vmap(jax.vmap(calculate_determinant, in_axes=0), in_axes=0)

def split_matrix(orbitals_determinant: jnp.ndarray, n_spin: int) -> jnp.ndarray:
    """We extract information only from one determinant. 10.09.2025."""
    #jax.debug.print("orbitals_uu:{}", orbitals_determinant)
    array1 = jnp.arange(n_spin)
    #jax.debug.print("array1:{}", array1)
    array2 = jnp.array(list(itertools.combinations(array1, 2)))
    #jax.debug.print("array2:{}", array2)
    array4 = generate_sub_blocks_vmap(array2, orbitals_determinant)
    #jax.debug.print("array4:{}", array4)
    array5 = generate_sub_sub_blocks_vmap(array2, array4)
    #jax.debug.print("array5:{}", array5)
    det_value = calculate_determinant(array5)
    #jax.debug.print("det_value:{}", det_value)
    return det_value




