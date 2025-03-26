"""we need calculate the jacobian.25.3.2025."""
import jax.numpy as jnp
import jax
from AIQMCrelease3.wavefunction_Ynlm.nn import construct_input_features

'''
from AIQMCrelease3.initial_electrons_positions.init import init_electrons
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
def weights_jacobian(pos: jnp.array, atoms: jnp.array, new_atoms: jnp.array):
    """to be continued... 25.3.2025.
    calculate the jacobian of walker i"""
    jax.debug.print("pos:{}", pos)
    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
    deltaR = new_atoms - atoms

    def jacobian_element(ae_inner: jnp.array, atoms_inner: jnp.array):
        temp1 = jnp.sum(-4 * jnp.abs(ae_inner) ** (-5) * (1 - atoms_inner), axis=-1, keepdims=True)
        temp2 = deltaR[:, 0] * (-4 * jnp.abs(ae_inner) ** (-5) * (1 - atoms_inner))
        temp3 = jnp.sum(temp2 / temp1, axis=-1, keepdims=True) + 1
        return temp3

    '''
    temp1 = jnp.sum(-4 * jnp.abs(ae[:, :, 0]) ** (-5) * (1 - atoms[:, 0]), axis=-1, keepdims=True)
    # jax.debug.print("temp1:{}", temp1)
    temp2 = deltaR[:, 0] * (-4 * jnp.abs(ae[:, :, 0]) ** (-5) * (1 - atoms[:, 0]))
    # jax.debug.print("temp2:{}", temp2)
    temp3 = jnp.sum(temp2 / temp1, axis=-1, keepdims=True) + 1
    # jax.debug.print("temp3:{}", temp3)
    '''
    x = jacobian_element(ae[:, :, 0], atoms[:, 0])
    y = jacobian_element(ae[:, :, 1], atoms[:, 1])
    z = jacobian_element(ae[:, :, 2], atoms[:, 2])
    #jax.debug.print("x:{}", x)
    #jax.debug.print("y:{}", y)
    #jax.debug.print("z:{}", z)
    jacobian = jnp.prod(x * y * z)
    #jax.debug.print("jacobian:{}", jacobian)
    return jacobian
'''
output = weights_jacobian(pos, atoms, new_atoms)
'''