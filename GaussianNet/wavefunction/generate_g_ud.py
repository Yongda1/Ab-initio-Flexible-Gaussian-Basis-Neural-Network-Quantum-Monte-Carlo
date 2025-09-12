import jax
import jax.numpy as jnp


def multiplication_0(orbitals_determinant_uu: jnp.ndarray, orbitals_determinant_dd: jnp.ndarray,):
    #jax.debug.print("temp1:{}", orbitals_determinant_uu)
    #jax.debug.print("temp2:{}", orbitals_determinant_dd)
    temp1 = orbitals_determinant_uu * orbitals_determinant_dd
    #jax.debug.print("temp3:{}", temp1)
    return temp1

multiplication_0_vmap =jax.vmap(jax.vmap(multiplication_0, in_axes=(0, None)), in_axes=(0, None))

def split_matrix( orbitals_determinant_uu: jnp.ndarray,
                  orbitals_determinant_dd: jnp.ndarray,
                  orbitals_determinant_ud: jnp.ndarray,
                  orbitals_determinant_du: jnp.ndarray):
    """first, we calculate symmetric matrix f_S."""
    jax.debug.print("orbitals_determinant_uu:{}", orbitals_determinant_uu)
    jax.debug.print("orbitals_determinant_dd:{}", orbitals_determinant_dd)
    jax.debug.print("orbitals_determinant_ud:{}", orbitals_determinant_ud)
    jax.debug.print("orbitals_determinant_du:{}", orbitals_determinant_du)
    output = multiplication_0_vmap(orbitals_determinant_uu, orbitals_determinant_dd)
    #jax.debug.print("output:{}", output.shape)
    output1 = multiplication_0_vmap(orbitals_determinant_ud, orbitals_determinant_du)
    #jax.debug.print("output1:{}", output1.shape)
    f_s = output + output1
    f_t = output - output1
    #jax.debug.print("f_S:{}", f_s.shape)
    return f_s, f_t