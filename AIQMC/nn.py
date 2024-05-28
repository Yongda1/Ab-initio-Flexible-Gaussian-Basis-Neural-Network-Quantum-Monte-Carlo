"""
This is the structure of Neural Network. Actually, it is the wave function.
Because probably we wont use the way of Ferminet to construct Nerual Network,
we only use some init functions for full connected layers.
We only consider real number.
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

'''
def make_ai_net(nspins: Tuple[int, int],
                charges: jnp.ndarray,
                ndim: int = 3,
                determinants: int = 16,
                hidden_dims: ainetLayers=((16, 8), (16, 8), (16, 8)),
                separate_spin_channels: bool = False,
                ) -> Network:
    natoms = charges.shape[0]
    feature_layer = make_ainet_features(natoms, nspins, ndim=ndim)
'''
