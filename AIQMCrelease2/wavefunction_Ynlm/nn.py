import enum
import functools
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
import chex
from AIQMCrelease2.wavefunction_Ynlm import network_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
from jax.scipy.special import sph_harm


AILayers = Tuple[Tuple[int, int], ...]
AIYnlmLayers = Tuple[Tuple[int, int], ...]
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]

@chex.dataclass
class AINetData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


class InitLayersAI(Protocol):
    def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        """"""

class ApplyLayersAI(Protocol):
    def __call__(self,
                 params: ParamTree,
                 ae: jnp.array,
                 r_ae: jnp.array,
                 ee: jnp.array,
                 r_ee: jnp.array,
                 spins: jnp.array,
                 charges: jnp.array) -> jnp.array:
        """"""


class FeatureInit(Protocol):
    def __call__(self) -> Tuple[Tuple[int, int], Param]:
        """"""


class FeatureApply(Protocol):
    def __call__(self, ae: jnp.array, r_ae: jnp.array, ee: jnp.array, r_ee: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """"""


@attr.s(auto_attribs=True)
class FeatureLayer:
    init: FeatureInit
    apply: FeatureApply


class InitAINet(Protocol):
    def __call__(self, key: chex.PRNGKey) -> ParamTree:
        """"""

class AINetLike(Protocol):
    def __call__(self,
                 params: ParamTree,
                 electrons: jnp.array,
                 spins: jnp.array,
                 atoms: jnp.array,
                 charges: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """"""

class OrbitalAILike(Protocol):

    def __call__(
            self,
            params: ParamTree,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> Sequence[jnp.ndarray]:
        """"""


@attr.s(auto_attribs=True)
class Network:
    init: InitAINet
    apply: AINetLike
    orbitals: OrbitalAILike

def construct_input_features(
        pos: jnp.ndarray,
        atoms: jnp.ndarray,
        ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert atoms.shape[1] == ndim
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (
            jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


def make_ainet_features(natoms: int, ndim: int = 3, rescale_inputs: bool = False) -> FeatureLayer:

    def init() -> Tuple[Tuple[int, int], Param]:
        return (natoms * (ndim + 1), ndim + 1), {}

    def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.array, jnp.array]:
        if rescale_inputs:
            log_r_ae = jnp.log(1 + r_ae)  # grows as log(r) rather than r
            ae_features = jnp.concatenate((log_r_ae, ae * log_r_ae / r_ae), axis=2)

            log_r_ee = jnp.log(1 + r_ee)
            ee_features = jnp.concatenate((log_r_ee, ee * log_r_ee / r_ee), axis=2)

        else:
            ae_features = jnp.concatenate((r_ae, ae), axis=2)
            ee_features = jnp.concatenate((r_ee, ee), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features, ee_features

    return FeatureLayer(init=init, apply=apply)


def construct_symmetric_features(h_one: jnp.array,
                                 h_two: jnp.array,
                                 nspins: Tuple[int, int]) -> jnp.array:

    spin_partitions = network_blocks.array_partitions(nspins)
    #jax.debug.print("spin_partitions:{}", spin_partitions)
    h_ones = jnp.split(h_one, spin_partitions, axis=0)
    h_twos = jnp.split(h_two, spin_partitions, axis=0)
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]
    features = [h_one] + g_one + g_two
    return jnp.concatenate(features, axis=1)


def make_ai_net_layers(nspins: Tuple[int, int],
                       nelectrons: int,
                       natoms: int,
                       hidden_dims,
                       hidden_dims_Ynlm,
                       feature_layer) -> Tuple[InitLayersAI, ApplyLayersAI]:

    def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        params = {}
        key, subkey = jax.random.split(key, num=2)
        (num_one_features, num_two_features), params['input'] = feature_layer.init()
        nchannels = len([nspin for nspin in nspins if nspin > 0])
        jax.debug.print("nchannels:{}", nchannels)

        def nfeatures(out1, out2):
            return (nchannels + 1) * out1 + nchannels * out2

        dims_one_in = num_one_features
        jax.debug.print("dims_one_in:{}", dims_one_in)
        dims_two_in = num_two_features
        jax.debug.print("dims_two_in:{}", dims_two_in)
        key1, subkey1 = jax.random.split(key)
        layers = []
        layers_y = []
        for i in range(len(hidden_dims)):
            layer_params = {}
            layer_params_y = {}
            key, single_key, single_y_key, *double_keys = jax.random.split(key1, num=5)
            dims_two_embedding = dims_two_in
            dims_one_in = nfeatures(dims_one_in, dims_two_embedding)
            dims_one_out, dims_two_out = hidden_dims[i]
            jax.debug.print("dims_one_in:{}", dims_one_in)
            """someting is wrong about the dimension. solve it later. 18.2.2025."""
            layer_params['single'] = network_blocks.init_linear_layer(single_key,
                                                                      in_dim=dims_one_in,
                                                                      out_dim=dims_one_out,
                                                                      include_bias=True)
            layer_params['single_Ynlm'] = network_blocks.init_linear_layer(single_y_key,
                                                                           in_dim=dims_one_in,
                                                                           out_dim=dims_one_out,
                                                                           include_bias=True)
            if i < len(hidden_dims)-1:
                ndouble_channels = 1
                layer_params['double'] = []
                for ichannel in range(ndouble_channels):
                    layer_params['double'].append(network_blocks.init_linear_layer(double_keys[ichannel],
                                                                                   in_dim=dims_two_in,
                                                                                   out_dim=dims_two_out,
                                                                                   include_bias=True))

            layers.append(layer_params)
            layers_y.append(layer_params_y)
            dims_one_in = dims_one_out
            dims_two_in = dims_two_out

        params['streams'] = layers
        params['streams_y'] = layers_y
        output_dims = dims_one_in
        return output_dims, params

    def apply_layer(params: Mapping[str, ParamTree],
                    h_one: jnp.array,
                    h_two: Tuple[jnp.array, ...],
                    ) -> Tuple[jnp.array, Tuple[jnp.array, ...]]:
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
        #jax.debug.print("h_one:{}", h_one)
        #jax.debug.print("h_two:{}", h_two)
        #jax.debug.print("nspins:{}", nspins)
        h_two_embedding = h_two[0]
        h_one_in = construct_symmetric_features(h_one, h_two_embedding, nspins)
        #jax.debug.print("h_one_in:{}", h_one_in)
        h_one_next = jnp.tanh(network_blocks.linear_layer(h_one_in, **params['single']))
        h_one = residual(h_one, h_one_next)
        return h_one, h_two
    
    def apply(params,
              ae: jnp.array,
              r_ae: jnp.array,
              ee: jnp.array,
              r_ee: jnp.array,) -> jnp.array:
        ae_features, ee_features = feature_layer.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)

        temp = ae / r_ae
        #jax.debug.print("temp:{}", temp)
        #jax.debug.print("params:{}", params)
        h_one = ae_features
        h_two = [ee_features]

        for i in range(len(hidden_dims)):
            #jax.debug.print("params['streams'][i]['single']:{}", params['streams'][i]['single'])
            h_one, h_two = apply_layer(params['streams'][i],
                                       h_one,
                                       h_two)

        h_to_orbitals = h_one

        return h_to_orbitals
    return init, apply




def make_ai_net(nspins: Tuple[int, int],
                charges: jnp.array,
                ndim: int = 3,
                natoms: int = 2,
                nelectrons: int = 8,
                determinants: int = 1,
                bias_orbitals: bool = True,
                rescale_inputs: bool = False,
                hidden_dims: AILayers = ((4, 2), (4, 2), (4, 2)),
                hidden_dims_Ynlm: AIYnlmLayers = ((4, 2), (4, 2), (4, 2)),):

    feature_layer = make_ainet_features(natoms, ndim=ndim, rescale_inputs=rescale_inputs)

    equivariant_layers = make_ai_net_layers(nspins, nelectrons, natoms, hidden_dims, hidden_dims_Ynlm, feature_layer)
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    return equivariant_layers_init, equivariant_layers_apply



from AIQMCrelease2.initial_electrons_positions.init import init_electrons
atoms = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
charges = jnp.array([4.0, 4.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
natoms = 2
ndim = 3
nspins = (4, 4)
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
equivariant_layers_init, equivariant_layers_apply = make_ai_net(ndim=ndim,
                                                                nspins=nspins,
                                                                determinants=1,
                                                                charges=charges,)

dims_orbital_in, params = equivariant_layers_init(subkey)
pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                                    electrons=spins,
                                    batch_size=1, init_width=1.0)

pos = jnp.reshape(pos, (-1))
ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
#jax.debug.print("ae:{}", ae)
#jax.debug.print("r_ae:{}", r_ae)
#jax.debug.print("ee:{}", ee)
#jax.debug.print("r_ee:{}", r_ee)
output = equivariant_layers_apply(params, ae, r_ae, ee, r_ee)