import jax.numpy as jnp
import jax
from kan_networks_case_one import make_kan_net
from spin_indices import jastrow_indices_ee, jastrow_indices_ae

"""we make the example for C atom which has six electrons.23.10.2025."""
seed = 42
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
atoms = jnp.array([[0.0, 0.0, 0.0]])
pos = jnp.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
charges = jnp.array([0.0])
spins_test = jnp.array([[1., 1., 1., -1, -1, -1]])
spins = spins_test
spin_jastrow = jnp.array([1., 1., 1., -1., -1., -1.])
parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spin_jastrow, nelectrons=6)
jax.debug.print("parallel_indices:{}", parallel_indices)
jax.debug.print("antiparallel_indices:{}", antiparallel_indices)


"""parameters for neural networks. We also need add different grid range for different layer ? 23.10.2025."""
layer_dims = jnp.array([4, 4, 4, 8])
g = jnp.array([3, 3, 3,])
k = jnp.array([3, 3, 3,])
grid_range = jnp.array([[0, 1], [0, 1], [0, 1]])
# the first number of nodes of layer_dims must be 4 because it is the number of features.
# the last number of nodes of layer_dims must be 6 because it is the number of electrons.
kan_init, kan_apply, orbitals_apply = make_kan_net(nspins=(3, 3),
                                                   charges=charges,
                                                   nelectrons=6,
                                                   nfeatures=4,
                                                   n_parallel=n_parallel,
                                                   n_antiparallel=n_antiparallel,
                                                   parallel_indices=parallel_indices,
                                                   antiparallel_indices=antiparallel_indices,
                                                   grid_range=grid_range,
                                                   g=g,
                                                   k=k,
                                                   natoms=1,
                                                   ndims=3,
                                                   layer_dims=layer_dims,
                                                   g_envelope=3,
                                                   k_envelope=3,
                                                   grid_range_envelope=jnp.array([0, 5]),
                                                   chebyshev=True,
                                                   spline=False,
                                                   add_residual=False,
                                                   add_bias=True,
                                                   external_weights=True,
                                                   envelope_chebyshev=True,
                                                   envelope_spline=False,
                                                   envelope_simple=False,)

params = kan_init(subkey)
#jax.debug.print("params:{}", params)
#jax.debug.print("params_embedding_single:{}", params['layers']['embedding_layer'][0]['single'])
#jax.debug.print("params:{}", params)
mask = jax.tree.map(lambda x: x == 1, params)
#for x in params:
#    jax.debug.print("x:{}", x)
#jax.debug.print("mask:{}", type(mask))
#jax.debug.print("mask:{}", mask)
'''
jax.debug.print("mask_envelope:{}", mask['envelope'])
envelope_opt = True
if envelope_opt:
    #new_value = True
    my_dict = {k: v == False for k, v in mask['envelope'].items()}
    #mask['envelope'] = {k: new_value for k in mask['envelope']}
    jax.debug.print("mask:{}", mask['envelope'])
'''
wavefunction_value = kan_apply(params, pos, spins, atoms, charges)
jax.debug.print("wavefunction_value:{}", wavefunction_value)