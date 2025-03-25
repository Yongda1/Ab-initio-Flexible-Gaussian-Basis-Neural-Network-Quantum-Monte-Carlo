"""here, we try to implement the correlated samples form SWCT. 10.3.2025."""
import jax.numpy as jnp
import jax
#from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.wavefunction_Ynlm.nn import construct_input_features

'''
atoms = jnp.array([[0.0, 0.0, 0.0], [2/3, 1/3, 0.0]])
new_atoms = jnp.array([[0.2, 0.2, 0.2], [2/3-0.1, 1/3+0.1, 0.0]])
charges = jnp.array([4.0, 4.0])
spins = jnp.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
lattice = jnp.array([[0.5 * jnp.sqrt(3), 0.5, 0],
                     [0.5 * jnp.sqrt(3), -0.5, 0],
                     [0, 0, 10]])
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
pos, spins = init_electrons(subkey, structure=lattice, atoms=atoms, charges=charges,
                            electrons=spins,
                            batch_size=1, init_width=0.5)
pos = jnp.reshape(pos, (-1))
'''

def correlated_samples(atoms: jnp.array, new_atoms: jnp.array, pos: jnp.array):
    """create the new samples for the atom displacement."""
    jax.debug.print("new_atoms:{}", new_atoms)
    jax.debug.print("pos:{}", pos)
    deltaR = new_atoms - atoms
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
    k_r_R = 1 / (r_ae**4)
    denominator = jnp.sum(jnp.sum(k_r_R, axis=-1), axis=-1, keepdims=True)

    def devided(k_r_R_inner: jnp.array, denominator_inner: jnp.array):
        return k_r_R_inner / denominator_inner

    devided_parallel = jax.vmap(devided, in_axes=(0, 0))
    output = devided_parallel(k_r_R, denominator)

    def multiply(output_inner: jnp.array, deltaR_inner: jnp.array):
        return output_inner * deltaR_inner

    multiply_parallel = jax.vmap(multiply, in_axes=(0, None))
    move = multiply_parallel(output, deltaR)
    move = jnp.sum(move, axis=1)
    move = jnp.reshape(move, (-1))
    new_pos = pos + move
    return new_pos

'''
newpos = correlated_samples(atoms, new_atoms, pos)
jax.debug.print("pos:{}", pos)
jax.debug.print("newpos:{}", newpos)
'''