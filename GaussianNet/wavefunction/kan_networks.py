from jaxkan.KAN import KAN
import jax.numpy as jnp
import jax
import chex
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union




ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]

def construct_input_features(
        pos: jnp.ndarray,
        atoms: jnp.ndarray,
        ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs inputs to Fermi Net from raw electron and atomic positions."""
    assert atoms.shape[1] == ndim
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


def make_kan_features(natoms: int, ndim: int = 3):
    def init() -> Tuple[Tuple[int], Param]:
        return (natoms * (ndim +1),), {}

    def apply(ae, r_ae) -> jnp.ndarray:
        ae_features = jnp.concatenate((r_ae, ae), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features
    return init, apply



def make_kan_net_layers(nspins: Tuple[int, int],
                        natoms: int,
                        nelectrons: int,
                        charges: jnp.ndarray,
                        feature_layer,
                        layer_dims: jnp.ndarray,
                        grid_range: jnp.ndarray = jnp.array([0., 1.]),
                        residual: bool = True,
                        bias: bool = True,
                        external_weights: bool = True,
                        ):

    def init(key: chex.PRNGKey):
        """here, we initialize the parameters of KANets wave function. 9.10.2025."""
        params = {}

        """we also need initialize the gird for each layer."""

        for i in range(len(layer_dims)-1):
            """because the number of nodes in KANets is controlled by layer_dims=jnp.ndarray([3, 4, 5, 6]). Therefore, the """
            layer_params = {}


        return params
    def apply_layer():
        return None

    return None




def make_orbitals(nspins: Tuple[int, int],
                  charges: jnp.ndarray,
                  equivariant_layers):
    def init(key: chex.PRNGKey) -> ParamTree:
        params = {}
        return params

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray) -> jnp.ndarray:
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        jax.debug.print("ae:{}", ae)
        jax.debug.print("r_ae: {}", r_ae)

        return None
    return init, apply







def make_kan_net(nspins: Tuple[int, int],
                 charges: jnp.ndarray,
                 nelectrons: jnp.ndarray,
                 layer_dims : jnp.ndarray,
                 natoms: int,
                 ndims: int=3,
                 ):
    feature_layer = make_kan_features(natoms=natoms, ndim=ndims)
    kan_equivariant_layers = make_kan_net_layers(
        nspins=nspins,
        charges=charges,
        natoms=natoms,
        nelectrons=nelectrons,
        feature_layer=feature_layer,
        layer_dims=layer_dims,
    )

    orbitals_init, orbitals_apply = make_orbitals(nspins=nspins,
                                                  charges=charges,
                                                  equivariant_layers=kan_equivariant_layers,)

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(key)

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray,):
        determinant = orbitals_apply(params, pos, spins, atoms, charges)
        return determinant

    return init, apply


seed = 23
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
atoms = jnp.array([[0.0, 0.0, 0.0]])
pos = jnp.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
charges = jnp.array([0.0])
spins_test = jnp.array([[1., 1., 1., - 1., - 1., -1.]])
spins = spins_test
layer_dims = jnp.array([6, 8, 8, 1])
kan_init, kan_apply = make_kan_net(nspins=(3,3),
                                   charges=charges,
                                   nelectrons=6,
                                   natoms=1,
                                   ndims=3,
                                   layer_dims=layer_dims)

params = kan_init(subkey)
wavefunction_value = kan_apply(params, pos, spins, atoms, charges)
jax.debug.print("wavefunction_value:{}", wavefunction_value)

'''
n_in = 6
n_out = 1
n_hidden = 8
seed = 42
layer_dims = [n_in, n_hidden, n_hidden, n_out]
req_params = {'G': 10,'external_weights':True}
model = KAN(layer_dims=layer_dims,
            layer_type='Spline',
            required_parameters=req_params,
            seed=seed)
print(model.layers.Param)
'''