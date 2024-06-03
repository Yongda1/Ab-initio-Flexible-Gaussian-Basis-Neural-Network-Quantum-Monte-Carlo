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
from AIQMC import Jastrow
import jax
import chex
import jax.numpy as jnp
from jax import Array
from typing_extensions import Protocol
import numpy as np

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]


class FeatureInit(Protocol):

    def __call__(self) -> Tuple[int, Param]:
        """Create the learnable parameters for the feature input layer."""


class FeatureApply(Protocol):

    def __call__(self, ae_ee: jnp.ndarray, **params: jnp.ndarray) \
            -> jnp.ndarray:
        """Creates the features to pass into the network."""


@attr.s(auto_attribs=True)
class FeatureLayer:
    init: FeatureInit
    apply: FeatureApply


"""Let us finish the first function for constructing input features. We do not need the norm of r_ae and r_ee vectors as input
features, because we use one-body and two body Jastrow factors. Here, we need think how many features will be used."""


def construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3) \
        -> jnp.ndarray:
    """Construct inputs to AINet from raw electron and atomic positions.
        pos: electron positions, Shape(nelectrons * dim)
        atoms: atom positions. Shape(natoms, ndim)"""
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    # here, we flat array to delete 0 distance.notes, this array is only working for C atom which has 4 electrons.
    ee = jnp.reshape(jnp.delete(jnp.reshape(ee, [-1, ndim]), jnp.array([0, 5, 10, 15]), axis=0), [-1, 3, ndim])
    ae_ee = jnp.concatenate((ae, ee), axis=1)
    return ae_ee


pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0]])
ae_ee = construct_input_features(pos, atoms, ndim=3)
#print(ae_ee)


def make_ainet_features(natoms: int, ndim: int = 3, ):
    def init() -> Tuple[int, Param]:
        """This is the number of features. We dont use two streams for ae and ee. So we dont need a tuple."""
        return (natoms * ndim + 3 * ndim), {}

    def apply(ae_ee) -> jnp.ndarray:
        ae_ee_input = jnp.reshape(ae_ee, [len(ae_ee), -1])
        return ae_ee_input

    return FeatureLayer(init=init, apply=apply)


def construct_symmetric_features(h: jnp.ndarray, nspins: Tuple[int, int]) -> jnp.ndarray:
    """here, we dont spit spin up and spin down electrons, so this function is not necessary."""
    return h

feature_layer = make_ainet_features(natoms=1)
ae_ee_input = feature_layer.apply(ae_ee)
print(ae_ee_input)
"""currently, we do not specify the output of make_ai_net_layer"""
hidden_dims = (8, 8, 8)

def make_ai_net_layers(nspins: Tuple[int, int], natoms: int, feature_layer, hidden_dims):
    """Create the permutation-equivariant and interaction layers. Because we are only using one stream for the input,
    we dont need an extra function for calculating the number of features.
    We use an auxiliary stream "Z_eff" to help construct the output function. When there is only one atom, this means less.
    We do not distinguish different channels for different spins."""
    def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        params = {}
        key, eff_charge_key = jax.random.split(key, num=2)
        (num_features), params['input'] = feature_layer.init()
        #params['effective_charge'] = nnblocks.init_linear_layer(
        #    eff_charge_key, in_dim=natoms, out_dim=natoms, include_bias=True)
        key, subkey = jax.random.split(key)
        layers = []
        for i in range(len(hidden_dims)):
            single_key, double_key = jax.random.split(key, num=2)
            layer_params = {}
            dims_in = num_features
            dims_out = hidden_dims[i]
            layer_params['single'] = nnblocks.init_linear_layer(
                single_key, in_dim=dims_in, out_dim=dims_out, include_bias=True)
            layers.append(layer_params)

        output_dims = hidden_dims[-1]
        params['streams'] = layers
        return output_dims, params

    def apply_layer(params: Mapping[str, ParamTree], h: jnp.ndarray):
        residual = lambda x, y: (x + y)/jnp.sqrt(2.0) if x.shape == y.shape else y
        h_in = construct_symmetric_features(h, nspins=nspins)
        h_next = jnp.tanh(nnblocks.linear_layer(h_in, **params['single']))
        h = residual(h, h_next)
        return h

    def apply(params, ae_ee: jnp.ndarray, spins: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
        ae_ee_input = feature_layer.apply(ae_ee=ae_ee, **params['input'])
        h = ae_ee_input
        for i in range(len(hidden_dims)):
            h = apply_layer(params['streams'][i], h)

        h_to_orbitals = construct_symmetric_features(h)
        return h_to_orbitals
    return init, apply


def make_orbitals(nspins: Tuple[int, int], charges: jnp.ndarray, equivariant_layers):
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply = Jastrow.get_jastrow

    def init(key: chex.PRNGKey) -> ParamTree:
        """Returns initial random parameters for creating orbitals."""
        key, subkey = jax.random.split(key)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)



def make_ai_net(nspins: Tuple[int, int], charges: jnp.ndarray, ndim: int=3, determinants: int=16, hidden_dims= (8, 8, 8)):
    natoms = charges.shape[0]
    feature_layer = make_ainet_features(natoms, nspins, ndim)
    equivariant_layers = make_ai_net_layers(nspins, charges.shape[0])