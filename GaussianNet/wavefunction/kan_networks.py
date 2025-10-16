from jaxkan.KAN import KAN
import jax.numpy as jnp
import jax
import chex
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import kan_networks_blocks
import kan_envelopes



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



def make_kan_net_layers(layer_dims: jnp.ndarray):

    def init(key: chex.PRNGKey):
        """here, we initialize the parameters of KANets wave function. 9.10.2025."""
        params = {}
        """we also need initialize the gird for each layer."""
        layers = []
        for i in range(len(layer_dims)-1):
            """because the number of nodes in KANets is controlled by layer_dims=jnp.ndarray([3, 4, 5, 6]). Therefore, the """
            layer_params = {}
            dimension_in = int(layer_dims[i])
            dimension_out = int(layer_dims[i+1])
            layer_params['single'] = kan_networks_blocks.init_ka_layer(key=key,
                                                  n_in=dimension_in,
                                                  n_out=dimension_out,
                                                  g=3,
                                                  k=3,
                                                  add_residual=True,
                                                  add_bias=True,
                                                  external_weights=True)
            layers.append(layer_params)
            dimension_in = int(layer_dims[i+1])

        params['embedding_layer'] = layers
        return params


    def apply_layer(params: Mapping[str, ParamTree],
                    h_one: jnp.ndarray,
                    n_in: int,
                    n_out: int,
                    g: int,
                    k: int,
                    grid_range: jnp.ndarray,
                    ):
        h_one_next = kan_networks_blocks.forward_each_layer(x=h_one,
                                                            n_in=n_in,
                                                            n_out=n_out,
                                                            g=g,
                                                            k=k,
                                                            grid_range=grid_range,
                                                            c_basis = params['c_basis'],
                                                            c_spl = params['c_spl'],
                                                            bias = params['bias'],
                                                            c_res = params['c_res'])
        return h_one_next

    def apply(params,
              ae: jnp.ndarray,):
        h_one = ae
        for i in range(len(layer_dims)-1):
            h_one = apply_layer(
                                params=params['embedding_layer'][i]['single'],
                                h_one=h_one,
                                n_in= int(layer_dims[i]),
                                n_out= int(layer_dims[i+1]),
                                g=3,
                                k=3,
                                grid_range=jnp.array([0, 1]))

        return h_one

    return init, apply




def make_orbitals(nspins: Tuple[int, int],
                  charges: jnp.ndarray,
                  equivariant_layers_init,
                  equivariant_layers_apply,):
    #equivariant_layers_init, equivariant_layers_apply = equivariant_layers()


    def init(key: chex.PRNGKey) -> ParamTree:
        params = {}
        key, subkey, key_map, key_envelope, key_orbital_1, key_orbital_2, key_orbital_3= jax.random.split(key, num=7)
        """we finished the parameters initialization of equivariant layers."""
        params['layers'] = equivariant_layers_init(subkey)
        params['map_h_to_orbitals'] = jax.random.normal(key_map, (3, 1, 4))
        params['envelopes'] = jax.random.normal(key_envelope, (3, 1, 1))
        params['orbital_1'] = kan_envelopes.init_ka_orbitals_layer(key=key_orbital_1, n_in=1, n_out=1, g=3, k=3)
        params['orbital_2'] = kan_envelopes.init_ka_orbitals_layer(key=key_orbital_2, n_in=1, n_out=1, g=3, k=3)
        params['orbital_3'] = kan_envelopes.init_ka_orbitals_layer(key=key_orbital_3, n_in=1, n_out=1, g=3, k=3)
        return params

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray) -> jnp.ndarray:
        """To construct the determinant, we follow suc a rule. r1 r2 r3 -> orbital1 to get, orbital1(r1), orbital1(2), orbital1(3).
        Therefore, the shape of coe_eff and coe_eff_second should be like
                            orbital1(r1), orbital1(r2), orbital1(r3)
                            orbital2(r1), orbital2(r2), orbital2(r3)
                            orbital3(r1), orbital3(r2), orbital3(r3)
        """
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        jax.debug.print("ae:{}", ae)
        jax.debug.print("r_ae: {}", r_ae)
        input = jnp.concatenate((r_ae, ae), axis=2).reshape(3, 4)
        jax.debug.print("input:{}", input)
        """we need think more about the orbitals construction."""
        h_to_orbitals = equivariant_layers_apply(params['layers'], input)
        #h_to_orbitals = jnp.expand_dims(h_to_orbitals, 1)
        #jax.debug.print("params['map_h_to_orbitals']:{}", params['map_h_to_orbitals'])
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        coe_eff = jnp.sum(h_to_orbitals * params['map_h_to_orbitals'], axis=-1)
        #jax.debug.print("coe_eff:{}", coe_eff)
        #jax.debug.print("params_envelope:{}", params['envelopes'])
        #jax.debug.print("coe_eff:{}", coe_eff)
        r_ae = jnp.reshape(r_ae, (1, 1, 3))
        jax.debug.print("r_ae:{}", r_ae)
        coe_eff_second = jnp.exp(-1 * params['envelopes'] * r_ae).reshape(3, 3)
        #jax.debug.print("coe_eff_second:{}", coe_eff_second)
        total_coe = coe_eff + coe_eff_second
        #jax.debug.print("total_coe: {}", total_coe)
        r_ae = jnp.reshape(r_ae, (3, 1,))
        jax.debug.print("r_ae:{}", r_ae)
        orbitals1_spline = kan_envelopes.forward_each_layer(x=r_ae, n_in=1, n_out=1, g=3, k=3, grid_range=jnp.array([0, 1]),
                                    c_basis=params['orbital_1']['c_basis'])
        orbitals2_spline = kan_envelopes.forward_each_layer(x=r_ae, n_in=1, n_out=1, g=3, k=3,
                                                            grid_range=jnp.array([0, 1]),
                                                            c_basis=params['orbital_2']['c_basis'])
        orbitals3_spline = kan_envelopes.forward_each_layer(x=r_ae, n_in=1, n_out=1, g=3, k=3,
                                                            grid_range=jnp.array([0, 1]),
                                                            c_basis=params['orbital_3']['c_basis'])

        orbitals_spline_total = jnp.concatenate([orbitals1_spline, orbitals2_spline, orbitals3_spline]).reshape(3, 3)
        jax.debug.print("orbitals1_spline:{}", orbitals_spline_total)
        determinant = coe_eff * orbitals_spline_total
        jax.debug.print("determinant:{}", determinant)
        return determinant
    return init, apply







def make_kan_net(nspins: Tuple[int, int],
                 charges: jnp.ndarray,
                 nelectrons: int,
                 layer_dims : jnp.ndarray,
                 natoms: int,
                 ndims: int=3,
                 ):
    feature_layer = make_kan_features(natoms=natoms, ndim=ndims)
    kan_equivariant_layers_init, kan_equivariant_layers_apply = make_kan_net_layers(layer_dims=layer_dims,)

    orbitals_init, orbitals_apply = make_orbitals(nspins=nspins,
                                                  charges=charges,
                                                  equivariant_layers_init=kan_equivariant_layers_init,
                                                  equivariant_layers_apply=kan_equivariant_layers_apply)

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
pos = jnp.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3])
charges = jnp.array([0.0])
spins_test = jnp.array([[1., 1., - 1.,]])
spins = spins_test
layer_dims = jnp.array([3, 4, 4, 3])
kan_init, kan_apply = make_kan_net(nspins=(3,3),
                                   charges=charges,
                                   nelectrons=3,
                                   natoms=1,
                                   ndims=3,
                                   layer_dims=layer_dims)

params = kan_init(subkey)

jax.debug.print("params:{}", params)
jax.debug.print("params_embedding_single:{}", params['layers']['embedding_layer'][0]['single'])
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