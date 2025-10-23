import jax.numpy as jnp
import jax
from kan_networks_case_one import make_kan_net
from spin_indices import jastrow_indices_ee, jastrow_indices_ae

"""we make the example for C atom which has six electrons.23.10.2025."""
seed = 23
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
atoms = jnp.array([[0.0, 0.0, 0.0]])
pos = jnp.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
charges = jnp.array([0.0])
spins_test = jnp.array([[1., 1., - 1.,]])
spins = spins_test
spin_jastrow = jnp.array([1., -1., 1., -1., 1., -1.])
parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spin_jastrow, nelectrons=6)
jax.debug.print("parallel_indices:{}", parallel_indices)
jax.debug.print("antiparallel_indices:{}", antiparallel_indices)


"""parameters for neural networks. We also need add different grid range for different layer ? 23.10.2025."""
layer_dims = jnp.array([4, 4, 4, 6])
g = jnp.array([3, 3, 3,])
k = jnp.array([3, 3, 3,])

# the first number of nodes of layer_dims must be 4 because it is the number of features.
# the last number of nodes of layer_dims must be 6 because it is the number of electrons.
kan_init, kan_apply = make_kan_net(nspins=(3,3),
                                   charges=charges,
                                   nelectrons=6,
                                   nfeatures=4,
                                   n_parallel=n_parallel,
                                   n_antiparallel=n_antiparallel,
                                   parallel_indices=parallel_indices,
                                   antiparallel_indices=antiparallel_indices,
                                   g=g,
                                   k=k,
                                   natoms=1,
                                   ndims=3,
                                   layer_dims=layer_dims)

params = kan_init(subkey)
#jax.debug.print("params:{}", params)
#jax.debug.print("params_embedding_single:{}", params['layers']['embedding_layer'][0]['single'])
wavefunction_value = kan_apply(params, pos, spins, atoms, charges)
jax.debug.print("wavefunction_value:{}", wavefunction_value)