"""
This is the structure of Neural Network. Actually, it is the wave function.
Because probably we wont use the way of Ferminet to construct Nerual Network,
we only use some init functions for full connected layers.
We only consider real number.
Because we need a new ansatz, we have to change the feature layer for the neural network. We need construct
own input layer vector h.
"""

import enum
import functools
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
# from AIQMC import envelopes
from AIQMC import nnblocks
#from AIQMC import Jastrow
import jax
import chex
import jax.numpy as jnp
from jax import Array
from typing_extensions import Protocol
import numpy as np

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]

@chex.dataclass
class AINetData:
    """Data passed to network."""
    positions: Any
    spins: Any
    atoms: Any
    charges: Any

class LogAINetLike(Protocol):

    def __call__(self, params: ParamTree,
                 electrons: jnp.ndarray, spins: jnp.ndarray, atoms: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
        """Retruns the log magnitude of the wavefunction."""


class FeatureInit(Protocol):

    def __call__(self) -> Tuple[Tuple[int, int], Param]:
        """Create the learnable parameters for the feature input layer."""


class FeatureApply(Protocol):

    def __call__(self, ae_ee: jnp.ndarray, **params: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Creates the features to pass into the network."""


@attr.s(auto_attribs=True)
class FeatureLayer:
    init: FeatureInit
    apply: FeatureApply

class InitLayersFn(Protocol):
    def __call__(self, key:chex.PRNGKey) -> Tuple[int, ParamTree]:
        """Returns output dim and initizalized parameters for the interaction layers."""

class ApplyLayersFn(Protocol):
    def __call__(self, params, ae: jnp.ndarray, ee: jnp.ndarray, nelectrons: int = 4) -> jnp.ndarray:
        """Forward evaluation of the interaction layers."""


"""Let us finish the first function for constructing input features. 
We do not need the norm of r_ae and r_ee vectors as input features, 
because we use one-body and two body Jastrow factors. Here, we need think how many features will be used.
Currenly, we only use a full connected layers"""


def construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Construct inputs to AINet from raw electron and atomic positions.
    Here, we assume that the electron spin is up and down along the axis=0 in array pos.
    So, the pairwise distance ae also follows this order.
        pos: electron positions, Shape(nelectrons * dim)
        atoms: atom positions. Shape(natoms, ndim)
    """
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    print("ae", ae)
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    print("ee", ee)
    # here, we flat array to delete 0 distance.notes, this array is only working for C atom which has 4 electrons.
    #ee = jnp.reshape(jnp.delete(jnp.reshape(ee, [-1, ndim]), jnp.array([0, 5, 10, 15]), axis=0), [-1, 3, ndim])
    #ae_ee = jnp.concatenate((ae, ee), axis=1)
    return ae, ee


def make_ainet_features(natoms: int = 2, nelectrons: int = 4, ndim: int = 3) -> FeatureLayer:

    def init() -> Tuple[Tuple[int, int], Param]:
        """This is the number of per electron-atom pair, electron-electron features.
        We need use two streams for ae and ee. And it is different from FermiNet. Because our convolution layer has
        a different structure.For simplicity, we only use the full connected layer.
        Maybe later, we will spend some time to rewrite this part."""
        return (natoms * ndim, nelectrons * ndim), {}

    def apply(ae, ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ae_features = ae
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1]) #reshape the ae vector to a line.
        ee_features = ee
        return ae_features, ee_features

    return FeatureLayer(init=init, apply=apply)

pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0], [1, 1, 1]])
ae, ee = construct_input_features(pos, atoms, ndim=3)
feature_layer = make_ainet_features(natoms=3, nelectrons=4, ndim=3)
ae_features, ee_features = feature_layer.apply(ae, ee)
print("ae_features", ae_features)
print("ee_features", ee_features)


def construct_symmetric_features(ae_features: jnp.ndarray, ee_features: jnp.ndarray, nelectrons: int) -> jnp.ndarray:
    """here, we dont spit spin up and spin down electrons, so this function is not necessary."""
    ee_features = jnp.reshape(ee_features, [nelectrons, -1])
    #print("ee", ee)
    #print("ae", ae)
    h = jnp.concatenate((ae_features, ee_features), axis=-1)
    return h

h = construct_symmetric_features(ae_features, ee_features, nelectrons=4)
print("h", h)

def make_ai_net_layers(nspins: Tuple[int, int], natoms: int, feature_layer) -> Tuple[InitLayersFn, ApplyLayersFn]:
    """
    we have two streams ae and ee. So, the hidden layers must also have two parts.
    :param nspins:
    :param natoms:
    :param feature_layer: ae_features is shape 4* 6 i.e. 6=(2 * 3) ee_features is 
    :param hidden_dims:
    :return:
    """
    def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        params = {}
        (num_one_features, num_two_features), params['input'] = feature_layer.init()
        key, subkey = jax.random.split(key)
        nfeatures = num_one_features + num_two_features        
        layers = []
        hidden_dims = jnp.ndarray([16, 16, nfeatures])
        """here, we have some problems. Please be careful about the input dimensions and output dimensions.
        Here, we only use one full connected. Therefore, the output dimensions should be the input for next layer."""
        dims_in = nfeatures
        for i in range(len(hidden_dims)):
            layer_params = {}
            dims_out = hidden_dims[i]
            layer_params['ae_ee'] = nnblocks.init_linear_layer(key, in_dim=dims_in, out_dim=dims_out, include_bias=True)
            layers.append(layer_params)
            dims_in = dims_out #this is why we need reset the input dimensions.

        params['streams_linear_layer'] = layers
        output_dims = nfeatures

        return output_dims, params

    def apply_layer(params: Mapping[str, ParamTree], h_in: jnp.ndarray) -> jnp.ndarray:
        h_next = jnp.tanh(nnblocks.linear_layer(h_in, **params['ae_ee']))
        return h_next

    def apply(params, ae: jnp.ndarray, ee: jnp.ndarray, nelectrons: int = 4) -> jnp.ndarray:
        ae_features, ee_features = feature_layer.apply(ae, ee, **params['input'])
        nfeatures = 2 * 3 + 4 * 3
        hidden_dims = jnp.ndarray([16, 16, nfeatures])
        h_in = construct_symmetric_features(ae_features, ee_features, nelectrons)

        for i in range(len(hidden_dims)):
            h = apply_layer(params['streams_linear_layer'][i], h_in=h_in)
            h_in = h

        h_to_orbitals = h
        return h_to_orbitals

    return init, apply


def make_orbitals(equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn]) -> ...:
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    """here, we need use complicated Jastrow factor. So this module could cost us some time."""
    jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply = Jastrow.get_jastrow

    def init(key: chex.PRNGKey) -> ParamTree:
        """Returns initial random parameters for creating orbitals."""
        key, subkey = jax.random.split(key)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)



#def make_ai_net(nspins: Tuple[int, int], charges: jnp.ndarray, ndim: int=3, determinants: int=16, hidden_dims= (8, 8, 8)):
    #natoms = charges.shape[0]
    #feature_layer = make_ainet_features(natoms, nspins, ndim)
    #equivariant_layers = make_ai_net_layers(nspins, charges.shape[0])