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
from AIQMC import envelopes
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
    #print("ae", ae)
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    #print("ee", ee)
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
        #print("natoms", natoms)
        #print("ndim", ndim)
        return (natoms * ndim, nelectrons * ndim), {}

    def apply(ae, ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ae_features = ae
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1]) #reshape the ae vector to a line.
        ee_features = ee
        return ae_features, ee_features

    return FeatureLayer(init=init, apply=apply)


def construct_symmetric_features(ae_features: jnp.ndarray, ee_features: jnp.ndarray, nelectrons: int) -> jnp.ndarray:
    """here, we dont spit spin up and spin down electrons, so this function is not necessary."""
    ee_features = jnp.reshape(ee_features, [nelectrons, -1])
    #print("ee", ee)
    #print("ae", ae)
    h = jnp.concatenate((ae_features, ee_features), axis=-1)
    return h

#h = construct_symmetric_features(ae_features, ee_features, nelectrons=4)


def make_ainet_layers(feature_layer) -> Tuple[InitLayersFn, ApplyLayersFn]:
    """
    we already check the shape of input and output. It is working well.19/07/2024
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
        #print("num_one_features, num_two_features", num_one_features, num_two_features)
        #print("nfeatures", nfeatures)
        layers = []
        hidden_dims = jnp.array([4, 4, nfeatures])
        """here, we have some problems. Please be careful about the input dimensions and output dimensions.
        Here, we only use one full connected. Therefore, the output dimensions should be the input for next layer."""
        dims_in = nfeatures
        for i in range(len(hidden_dims)):
            layer_params = {}
            dims_out = hidden_dims[i]
            layer_params['ae_ee'] = nnblocks.init_linear_layer(key, in_dim=dims_in, out_dim=dims_out, include_bias=True)
            layers.append(layer_params)
            dims_in = dims_out #this is why we need reset the input dimensions.

        params['linear_layer'] = layers
        output_dims = nfeatures
        #print("output_dims", output_dims)
        #print("params", params)
        return output_dims, params

    def apply_layer(params: Mapping[str, ParamTree], h_in: jnp.ndarray) -> jnp.ndarray:
        h_next = jnp.tanh(nnblocks.linear_layer(h_in, **params['ae_ee']))
        return h_next

    def apply(params, ae: jnp.ndarray, ee: jnp.ndarray, nelectrons: int = 4) -> jnp.ndarray:
        ae_features, ee_features = feature_layer.apply(ae, ee,)
        (num_one_features, num_two_features), params['input'] = feature_layer.init()
        nfeatures = num_one_features + num_two_features
        hidden_dims = jnp.array([4, 4, nfeatures])
        h_in = construct_symmetric_features(ae_features, ee_features, nelectrons)

        for i in range(len(hidden_dims)):
            h = apply_layer(params['linear_layer'][i], h_in=h_in)
            h_in = h

        h_to_orbitals = h
        #print("h_to_orbitals", h_to_orbitals)
        return h_to_orbitals

    return init, apply



"""this part is to be finished. 19/07/2024."""
def make_orbitals(natoms: int, nelectrons: int, num_angular: int, equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],) -> ...:
    """equivaiant_layers_init and equivariant_layers_apply are init, apply of make_ai_net_layers, separately, i.e.
    it is interaction layers."""
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    """we need rewrite the Jastrow part as a class.24/07/2024."""
    jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply = Jastrow.get_jastrow(jastrow="Pade")
    envelope = envelopes.make_GTO_envelope()

    def init(key: chex.PRNGKey) -> ParamTree:
        """Returns initial random parameters for creating orbitals."""
        key, subkey = jax.random.split(key)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)
        output_dims = dims_orbital_in
        """Here, we should put the envelope function."""
        params['envelope'] = envelope.init(natoms=2, nelectrons=4)
        params['jastrow_ae'] = jastrow_ae_init(nelectron=4, charges=jnp.array([2, 2]))
        params['jastrow_ee'] = jastrow_ee_init()
        orbitals = []

        """this out_dim should be the number of electrons."""
        for _ in jnp.arange(nelectrons):
            key, subkey = jax.random.split(key)
            orbitals.append(nnblocks.init_linear_layer(subkey, in_dim=dims_orbital_in, out_dim=nelectrons, include_bias=False))
        """we also add the coefficients for the first orbitals part. We need think what the in_dim and out_dim are.26/07/20204."""
        #coe_orbitals = []
        #coe_orbitals.append(nnblocks.init_linear_layer(key, in_dim=natoms, out_dim=nelectrons, include_bias=False))
        params['orbital'] = orbitals
        #params['ceo_orbitals'] = coe_orbitals
        print("params", params)
        #print("params_envelope", params['envelope'])

        return params

    def apply(params, pos: jnp.ndarray, atoms: jnp.ndarray):
        ae, ee = construct_input_features(pos, atoms, ndim=3)
        h_to_orbitals = equivariant_layers_apply(params=params['layers'], ae=ae, ee=ee)
        #print("h_to_orbitals", h_to_orbitals)
        #print("params[orbital]", params['orbital'])
        """the initial envelope function finished without the diffusion part.24/07/2024."""
        #xi = params['envelope']
        #print("type of xi", type(xi))
        envelope_factor = envelope.apply(ae, params=params['envelope'], natoms=2, nelectrons=4)
        print("envelope_factor", envelope_factor)
        #h_to_orbitals = jnp.reshape(h_to_orbitals, (nelectrons, 1, -1))
        #print("h_to_orbitals_modified", h_to_orbitals)
        #for h, p in zip(h_to_orbitals, params['orbital']):
        #    print("h", h)
        #    print("p", p)
        #print("params[orbitals]", params['orbital'])
        orbitals_first = jnp.array([nnblocks.linear_layer_no_b(h, **p) for h, p in zip(h_to_orbitals, params['orbital'])])
        print("orbitals_first", orbitals_first)
        """we need match the shape of orbitals.24/07/2024.Then we need debug the two functions.
        we need add r^l * e^(-r^2) on the orbitals_first. then times by envelope_factor.
        we need think about the output dimensions. then consider how to construct orbitals.
        Currently, we have more problems about how to think the orbitals. 29/07/2024.
        whatever, we can start to construct the matrix now.
        Before, we made a mistake about the orbitals. So, here we have to rewrite the envelope part. 
        We have 16 elements in the matrix. We also need 16 corresponding elements in the envelope functions."""
        envelope_factor = jnp.reshape(jnp.array(envelope_factor), (nelectrons, nelectrons))
        print('envelope_factor', envelope_factor)
        orbitals_end = jnp.array([a * b for a, b in zip(orbitals_first, envelope_factor)])
        print('orbitals_end', orbitals_end)
        orbitals_end = jnp.transpose(orbitals_end)
        print('orbitals_end', orbitals_end)
        #h_to_orbitals = envelope_factor * h_to_orbitals
        #print(jnp.shape(orbitals_first))
        #orbitals_first = jnp.reshape(jnp.array(orbitals_first), (-1))
        #temp1 = orbitals_first * orbitals_first
        #print(temp1)
        #orbitals_end = orbitals_first * jnp.exp(orbitals_first * orbitals_first) * envelope_factor
        """let's add Jastrow here. We need use the deter property, k^n det(A) = det(kA). 
        We have some bugs here.29/07/2024, please solve it tommorrow.
        we already solve it. The bug is from the sign in the one-body Jastrow."""
        temp1 = jastrow_ae_apply(ae=ae, nelectron=4, charges=jnp.array([2, 2]), params=params['jastrow_ae']) / nelectrons
        print('temp1', temp1)
        jastrow = jnp.exp(jastrow_ae_apply(ae=ae, nelectron=4, charges=jnp.array([2, 2]),
                                           params=params['jastrow_ae'])/nelectrons +
                          jastrow_ee_apply(ee=ee, nelectron=4, params=params['jastrow_ee'])/nelectrons)
        print('jastrow', jastrow)
        """here, we only have one determinant. Notes: currently, the wave_function is a determinant.
        Today, we finished the construction of the wave function as the single determinant 07.08.2024."""
        wave_function = jastrow*orbitals_end
        return wave_function

    return init, apply


pos = jnp.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5])
atoms = jnp.array([[0, 0, 0], [0.2, 0.2, 0.2]])
ae, ee = construct_input_features(pos, atoms, ndim=3)
feature_layer1 = make_ainet_features(natoms=2, nelectrons=4, ndim=3)
#ae_features, ee_features = feature_layer1.apply(ae, ee)
#print("ae_features", ae_features)
#print("ee_features", ee_features)
equivariant_layers = make_ainet_layers(feature_layer=feature_layer1)
a = jax.random.PRNGKey(seed=1)
#output_dims, params = init(key=a)
#h_to_orbitals = apply(params=params, ae=ae, ee=ee,)
init, apply = make_orbitals(natoms=2, nelectrons=4, num_angular=4, equivariant_layers=equivariant_layers)
parameters = init(a)
initialization = apply(params=parameters, pos=pos, atoms=atoms)


#def make_ai_net(nspins: Tuple[int, int], charges: jnp.ndarray, ndim: int=3, determinants: int=16, hidden_dims= (8, 8, 8)):
    #natoms = charges.shape[0]
    #feature_layer = make_ainet_features(natoms, nspins, ndim)
    #equivariant_layers = make_ai_net_layers(nspins, charges.shape[0])