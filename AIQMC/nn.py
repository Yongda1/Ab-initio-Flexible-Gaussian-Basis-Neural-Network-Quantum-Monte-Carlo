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
import chex
from AIQMC import envelopes
#from ferminet import jastrows
from AIQMC import nnblocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]

class FeatureInit(Protocol):

    def __call__(self) -> Tuple[Tuple[int, int], Param]:
        """Create the learnable parameters for the feature input layer."""

class FeatureApply(Protocol):

    def __call__(self, ae: jnp.ndarray, r_ae: jnp.ndarray, ee: jnp.ndarray, r_ee: jnp.ndarray, **params: jnp.ndarray) \
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Creates the features to pass into the network."""

@attr.s(auto_attribs=True)
class FeatureLayer:
    init: FeatureInit
    apply: FeatureApply



def construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct inputs to AINet from raw electron and atomic positions."""
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


def make_ainet_features(natoms: int,
                        nspins: Tuple[int, int],
                        ndim: int = 3,):
    def init() -> Tuple[Tuple[int, int], Param]:
        return (natoms * (ndim + 1), ndim + 1), {}

    def apply(ae, r_ae, ee, r_ee) ->Tuple[jnp.ndarray, jnp.ndarray]:
        ae_features = jnp.concatenate((r_ae, ae), axis=2)
        ee_features = jnp.concatenate((r_ee, ee), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features, ee_features

    return FeatureLayer(init=init, apply=apply)

pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0]])
print(atoms.shape[1])
ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
print("ae", ae)
print("ee", ee)
print("r_ae", r_ae)
print("r_ee", r_ee)
ae_features, ee_features = make_ainet_features.apply(natoms=1, nspins=[2,2])
"""For this part, we need think how to design the input features for the neural network."""
def make_ai_net_layers(nspins: Tuple[int, int], natoms: int, feature_layer, hidden_dims) -> Tuple[InitLayersFn, ApplyLayersFn]:
    """creates the permutation-equivariant and interaction layers for FermiNet."""
    def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        params = {}
        key, nuclear_key = jax.random.split(key, num=2)
        (num_one_features, num_two_features), params['input'] = (feature_layer.init())
        nchannels = len([nspin for nspin in nspins if nspin > 0])

        def nfeatures(out1, out2):
            return (nchannels + 1) * out1 + nchannels * out2

        dims_one_in = num_one_features
        dims_two_in = num_two_features

        for i in range(len(len(hidden_dims))):
            layer_params = {}
            dims_one_out, dims_two_out = hidden_dims[i]
            layer_params['single'] = 




def make_ai_net(nspins: Tuple[int, int],
                charges: jnp.ndarray,
                ndim: int = 3,
                determinants: int = 16,
                hidden_dims = ((16, 8), (16, 8), (16, 8)),
                separate_spin_channels: bool = False,
                ) -> Network:
    natoms = charges.shape[0]
    feature_layer = make_ainet_features(natoms, nspins, ndim=ndim)
    equivariant_layers = make_ai_net_layers(nspins, charges.shape[0], feature_layer=feature_layer, hidden_dims)
