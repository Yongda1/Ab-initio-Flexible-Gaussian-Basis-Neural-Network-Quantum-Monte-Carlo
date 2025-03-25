import jax.numpy as jnp
import jax

from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.wavefunction_Ynlm.nn import construct_input_features
atoms = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
charges = jnp.array([4.0, 4.0])
spins = jnp.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
lattice = jnp.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 10]])

key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
pos, spins = init_electrons(subkey, structure=lattice, atoms=atoms, charges=charges,
                            electrons=spins,
                            batch_size=1, init_width=0.5)
pos = jnp.reshape(pos, (-1))

ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)

"""we need put this method into the VMC and DMC."""
def enforcepbc(lattice: jnp.array, pos: jnp.array):
    """it is working for single configuration. However, we used two different strategies for VMC and DMC. For convenience,
    please use vmap or pmap to make this function be proper for VMC and DMC respectively. 17.3.2025.
    we just need add this function into the walkers move module."""
    recpvecs = jnp.linalg.inv(lattice)
    pos = jnp.reshape(pos, (-1, 3))

    def calculate(pos_inner: jnp.array, recpvecs_inner: jnp.array):
        return pos_inner * recpvecs_inner

    calculate_parallel = jax.vmap(jax.vmap(calculate, in_axes=(0, 1)), in_axes=(0, None))
    temp = calculate_parallel(pos, recpvecs)
    temp = jnp.sum(temp, axis=1)
    temp1 = jnp.divmod(temp, 1)

    def calculate_final_pos(temp1_inner1: jnp.array, lattice_inner: jnp.array):
        return jnp.dot(temp1_inner1, lattice_inner)
    
    calculate_final_pos_parallel = jax.vmap(jax.vmap(calculate_final_pos, in_axes=(0, 0)), in_axes=(0, None))
    final_pos = calculate_final_pos_parallel(temp1[1], lattice)
    final_pos = jnp.sum(final_pos, axis=1)
    final_pos = jnp.reshape(final_pos, (-1))
    return final_pos


final_pos_pbc = enforcepbc(lattice, pos)
jax.debug.print("fianl_pos_pbc:{}", final_pos_pbc)
