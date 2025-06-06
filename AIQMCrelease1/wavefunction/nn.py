"""
Create the many-body wave-function based on the ansatz from Yongda Huang at 17.12.2024.
For first steps, we construct the feature layers. The important change is the join of effective charges.
"""

from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
from AIQMCrelease1.wavefunction import envelopes
from AIQMCrelease1.wavefunction import nnblocks
from AIQMCrelease1.wavefunction import Jastrow
import jax
import chex
import jax.numpy as jnp
from typing_extensions import Protocol
import numpy as np
from jax.config import config; config.update("jax_enable_x64", True)


ParamTree = Union[
    jnp.array, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']
]
Param = MutableMapping[str, jnp.array]


@chex.dataclass
class AINetData:
    """Data passed to network."""
    positions: Any
    atoms: Any
    charges: Any


class LogAINetLike(Protocol):

    def __call__(self, params: ParamTree,
                 electrons: jnp.array, spins: jnp.array, atoms: jnp.array, charges: jnp.array) -> jnp.array:
        """Returns the log magnitude of the wavefunction."""


class FeatureInit(Protocol):

    def __call__(self) -> Tuple[Tuple[int, int], Param]:
        """Create the learnable parameters for the feature input layer."""


class FeatureApply(Protocol):

    def __call__(self, ae_inner: jnp.array, r_ae_inner: jnp.array, ee_inner: jnp.array, r_ee_inner: jnp.array, **params: jnp.array) \
            -> Tuple[jnp.array, jnp.array]:
        """Creates the features to pass into the network."""


@attr.s(auto_attribs=True)
class FeatureLayer:
    init: FeatureInit
    apply: FeatureApply


class InitLayersFn(Protocol):
    def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        """Returns output dim and initizalized parameters for the interaction layers."""


class ApplyLayersFn(Protocol):
    def __call__(self, params: ParamTree, ae: jnp.array, ee: jnp.array, r_ae: jnp.array, r_ee: jnp.array) -> jnp.array:
        """Forward evaluation of the interaction layers."""


class InitAINet(Protocol):
    def __call__(self, key: chex.PRNGKey) -> ParamTree:
        """Return initialized parameters for the network."""


class OrbitalsAILike(Protocol):
    def __call__(self, params: ParamTree, pos: jnp.array, atoms: jnp.array, charges: jnp.array) -> Sequence[jnp.array]:
        """Forward evaluation of the AINet up to the orbitals."""


class AINetLike(Protocol):
    def __call__(self, params: ParamTree, pos: jnp.array, atoms: jnp.array, charges: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """Return the sign and log magnitude of the wavefunction.
        Here, we also need add the spin configuration information into the input array."""


@attr.s(auto_attribs=True)
class Network:
    init: InitAINet
    apply: AINetLike
    orbitals: OrbitalsAILike


def construct_input_features(pos: jnp.array, atoms: jnp.array, ndim: int = 3) \
        -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """Construct inputs to AINet from raw electron and atomic positions.
    Here, we assume that the electron spin is up and down along the axis=0 in array pos.
    So, the pairwise distance ae also follows this order.
        pos: electron positions, Shape(nelectrons * dim)
        atoms: atom positions. Shape(natoms, ndim)
    """
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


def make_ainet_features(natoms: int, ndim: int) -> FeatureLayer:

    def init() -> Tuple[Tuple[int, int], Param]:
        """This is the number of per electron-atom pair, electron-electron features.
        We need use two streams for ae and ee. And it is different from FermiNet. Because our convolution layer has
        a different structure.For simplicity, we only use the full connected layer.
        Maybe later, we will spend some time to rewrite this part
        Different from FermiNet, they just use the following format because they did the average for the two electrons stream, 
        then add it into the input vector. The second number ndim + 1 does not mean the number of two electrons features.

        something is wrong. However, I cannot find where. 15.1.2025. The return value is Nan, that makes the kinetic energy be Nan."""
        return (natoms * (ndim + 1), ndim + 1), {}

    def apply(ae_inner, r_ae_inner, ee_inner, r_ee_inner) -> Tuple[jnp.array, jnp.array]:
        #jax.debug.print("ae_inner:{}", ae_inner)
        #jax.debug.print("r_ae_inner:{}", r_ae_inner)
        #jax.debug.print("ee_inner:{}", ee_inner)
        #jax.debug.print("r_ee_inner:{}", r_ee_inner)
        ae_features = jnp.concatenate((r_ae_inner, ae_inner), axis=2)
        ee_features = jnp.concatenate((r_ee_inner, ee_inner+0.1), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features, ee_features

    return FeatureLayer(init=init, apply=apply)


def construct_symmetric_features(h_one: jnp.array, h_two: jnp.array) -> jnp.array:
    """here, we dont split spin up and spin down electrons, so this function is not necessary.
    to be continued..., probably, I made a mistake about the wavefunction format."""
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_one]
    g_two = [jnp.mean(h, axis=0, keepdims=True) for h in h_two]
    g_two = jnp.reshape(jnp.array(g_two), (-1, 4))
    return jnp.concatenate([h_one, jnp.array(g_one), g_two], axis=1)


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
        key, subkey, single_shape_key, double_shape_key = jax.random.split(key, num=4)

        def nfeatures(out1, out2):
            return out1 + out2 + 1
        
        dims_one_in = num_one_features
        dims_two_in = num_two_features

        layers = []
        hidden_dims = ((4, 2), (4, 2), (4, 2), (4, 2))

        dims_one_in = nfeatures(dims_one_in, dims_two_in)
        output_dims = dims_one_in
        for i in range(len(hidden_dims)):
            dims_one_out, dims_two_out = hidden_dims[i]
            layer_params = {}
            layer_params['single'] = nnblocks.init_linear_layer(key,
                                                                in_dim=dims_one_in,
                                                                out_dim=dims_one_out,
                                                                include_bias=True)
            layer_params['single_shape_control'] = nnblocks.init_linear_layer(single_shape_key,
                                                                              in_dim=dims_one_out,
                                                                              out_dim=num_one_features,
                                                                              include_bias=True)

            layers.append(layer_params)
            #dims_one_in = dims_one_out
            #dims_two_in = dims_two_out
        params['linear_layer'] = layers
        return output_dims, params

    def apply_layer(params: Mapping[str, ParamTree], h_one: jnp.array, h_two: jnp.array) -> Tuple[jnp.array, jnp.array]:
        """rewrite this function.
        the input dimension is wrong for the loop. 13.1.2025. we solve tomm."""
        residual = lambda x, y: (x + y) / jnp.square(2.0)
        h_one_in = construct_symmetric_features(h_one, h_two)
        h_one_next = jnp.tanh(nnblocks.vmap_linear_layer(h_one_in, params['single']['w'], params['single']['b']))
        h_one_next_reshape = jnp.tanh(nnblocks.vmap_linear_layer(h_one_next, params['single_shape_control']['w'], params['single_shape_control']['b']))
        h_one = residual(h_one, h_one_next_reshape)
        return h_one, h_two

    def apply(params, ae: jnp.array, ee: jnp.array, r_ae: jnp.array, r_ee: jnp.array) -> jnp.array:
        ae_features, ee_features = feature_layer.apply(ae_inner=ae,  ee_inner=ee, r_ae_inner=r_ae, r_ee_inner=r_ee)
        #jax.debug.print("ae_features:{}", ae_features)
        #jax.debug.print("ee_features:{}", ee_features)
        (num_one_features, num_two_features), params['input'] = feature_layer.init()
        #nfeatures = num_one_features + num_two_features
        hidden_dims = ((8, 2), (8, 2), (8, 2), (8, 2))
        h_one = ae_features
        h_two = ee_features

        """here, please notice that with calculating more iterations, the dimension of h will be added one more."""
        for i in range(len(hidden_dims)):
            h_one, h_two = apply_layer(params['linear_layer'][i], h_one=h_one, h_two=h_two)

        h_to_orbitals = construct_symmetric_features(h_one, h_two)
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        return h_to_orbitals
    return init, apply


def make_orbitals(natoms: int,
                  nelectrons: int,
                  num_angular: int,
                  n_parallel: int,
                  n_antiparallel: int,
                  charges: jnp.array,
                  parallel_indices: jnp.array,
                  antiparallel_indices: jnp.array,
                  atom_jastrow_indices: jnp.array,
                  charged_jastrow_needed: jnp.array,
                  equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn],) -> ...:
    """equivaiant_layers_init and equivariant_layers_apply are init, apply of make_ai_net_layers, separately, i.e.
    it is interaction layers."""
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    jastrow_ae_init, jastrow_ae_apply, jastrow_ee_init, jastrow_ee_apply = Jastrow.get_jastrow(jastrow="Pade",
                                                                                               atom_indices=atom_jastrow_indices,
                                                                                               charges_needed=charged_jastrow_needed)
    envelope = envelopes.make_GTO_envelope()

    def init(key: chex.PRNGKey) -> ParamTree:
        """Returns initial random parameters for creating orbitals."""
        key, subkey = jax.random.split(key)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)
        #output_dims = dims_orbital_in
        params['envelope'] = envelope.init(natoms=natoms, nelectrons=nelectrons)
        params['jastrow_ae'] = jastrow_ae_init(nelectron=nelectrons)
        params['jastrow_ee'] = jastrow_ee_init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
        orbitals = []
        diffuse_coefficients = []
        exponent = []
        """this out_dim should be the number of electrons."""
        for _ in jnp.arange(nelectrons):
            key, subkey = jax.random.split(key)
            orbitals.append(nnblocks.init_linear_layer(subkey, in_dim=dims_orbital_in, out_dim=nelectrons, include_bias=False))
            diffuse_coefficients.append(nnblocks.init_linear_layer(subkey, in_dim=dims_orbital_in, out_dim=nelectrons, include_bias=False))
            exponent.append(nnblocks.init_linear_layer(subkey, in_dim=nelectrons, out_dim=nelectrons, include_bias=False))
        """we also add the coefficients for the first orbitals part. We need think what the in_dim and out_dim are.26/07/20204."""

        params['orbital'] = orbitals
        params['diffuse'] = diffuse_coefficients
        params['exponent'] = exponent
        jax.debug.print("orbitals:{}", orbitals)
        return params

    def apply(params, pos: jnp.array, atoms: jnp.array, charges: jnp.array):
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        #jax.debug.print("ae:{}", ae)
        #jax.debug.print("r_ae:{}", r_ae)
        h_to_orbitals = equivariant_layers_apply(params=params['layers'],
                                                 ae=ae, ee=ee,
                                                 r_ae=r_ae,
                                                 r_ee=r_ee,)
        jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        envelope_factor = envelope.apply(ae, xi=params['envelope']['xi'], natoms=natoms, nelectrons=nelectrons)
        for h, p in zip(h_to_orbitals, params['orbital']):
            jax.debug.print("h:{}", h)
            jax.debug.print("p:{}", p)
            x = nnblocks.linear_layer_no_b(h, **p)
            jax.debug.print("x:{}", x)

        r_effective = jnp.array([nnblocks.linear_layer_no_b(h, **p) for h, p in zip(h_to_orbitals, params['orbital'])])
        diffuse_part = jnp.array([nnblocks.linear_layer_no_b(h, **p) for h, p in zip(h_to_orbitals, params['diffuse'])])
        r_effective = r_effective**2
        jax.debug.print("r_effective:{}", r_effective)
        diffuse_part = jnp.exp(-1 * diffuse_part) + 1
        #jax.debug.print("r_effective:{}", r_effective)
        #jax.debug.print("diffuse_part:{}", diffuse_part)
        #jax.debug.print("envelop_factor:{}", envelope_factor)
        envelope_factor = jnp.reshape(jnp.array(envelope_factor), (nelectrons, nelectrons))
        exponent_part = jnp.array([nnblocks.linear_layer_no_b(h, **p) for h, p in zip(r_effective, params['exponent'])])
        orbitals_end = envelope_factor * jnp.exp(-1 * r_effective) * diffuse_part * jnp.power(r_effective, exponent_part)

        """let's add Jastrow here. We need use the deter property, k^n det(A) = det(kA). 
        We have some bugs here.29/07/2024, please solve it tomorrow.
        we already solve it. The bug is from the sign in the one-body Jastrow.
        Maybe Jastrow factors are wrong?
        Yes, Jastrow factors are wrong. We need rewrite it.
        we tried to rewrite some part of the Jastrow factors. however it still not working. Why???"""

        jastrow = jnp.exp(jastrow_ae_apply(ae=ae,
                                           params=params['jastrow_ae'])/nelectrons +
                          jastrow_ee_apply(ee=ee,
                                           parallel_indices=parallel_indices,
                                           antiparallel_indices=antiparallel_indices,
                                           params=params['jastrow_ee'])/nelectrons)
        
        #jax.debug.print("jastrow:{}", jastrow)
        #wave_function = jastrow * orbitals_end
        wave_function = orbitals_end

        return wave_function

    return init, apply


def make_ai_net(ndim: int,
                natoms: int,
                nelectrons: int,
                num_angular: int,
                n_parallel: int,
                n_antiparallel: int,
                parallel_indices: jnp.array,
                antiparallel_indices: jnp.array,
                charges: jnp.array,
                atom_jastrow_indices: jnp.array,
                charged_jastrow_needed: jnp.array,
                full_det: bool = True) -> Network:
    """Creates functions for initializing parameters and evaluating AInet.
    07.08.2024 we still have some problems about this module ,for example, """
    feature_layer1 = make_ainet_features(natoms=natoms, ndim=ndim)
    equivariant_layers = make_ainet_layers(feature_layer=feature_layer1)
    orbitals_init, orbitals_apply = make_orbitals(natoms=natoms,
                                                  nelectrons=nelectrons,
                                                  num_angular=num_angular,
                                                  n_parallel=n_parallel,
                                                  n_antiparallel=n_antiparallel,
                                                  charges=charges,
                                                  parallel_indices=parallel_indices,
                                                  antiparallel_indices=antiparallel_indices,
                                                  atom_jastrow_indices=atom_jastrow_indices,
                                                  charged_jastrow_needed=charged_jastrow_needed,
                                                  equivariant_layers=equivariant_layers)

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(subkey)

    def apply(params, pos: jnp.array, atoms: jnp.array, charges: jnp.array) -> Tuple[jnp.array, jnp.array]:

        """logdet_matmul function still has problems. We need solve it later.12.08.2024.!!!"""
        """the shape of orbitals is not correct. We can solve this problem tomorrow.22.08.2024.
        Is hamiltonian wrong?"""
        orbitals = orbitals_apply(params, pos, atoms, charges)
        #jax.debug.print("orbitals:{}", orbitals)
        #jax.debug.print("value_wavefunction:{}", nnblocks.slogdet(orbitals))
        return nnblocks.slogdet(orbitals)

    return Network(init=init, apply=apply, orbitals=orbitals_apply)


"""This part for debugging this module."""
'''
key = jax.random.PRNGKey(1)
spins = jnp.array([1.0, -1.0])
temp = jnp.reshape(spins, (2, 1)) #6 is the number of electrons.
spins = jnp.reshape(spins, (1, 2))
spins_total = spins * temp
spins_total_uptriangle = jnp.triu(spins_total, k=1)
sample = jnp.zeros_like(a=spins_total_uptriangle)
parallel = jnp.where(spins_total_uptriangle > sample, spins_total_uptriangle, sample)
antiparallel = jnp.where(spins_total_uptriangle < sample, spins_total_uptriangle, sample)
parallel_indices = jnp.nonzero(parallel)
antiparallel_indices = jnp.nonzero(antiparallel)
parallel_indices = jnp.array(parallel_indices)
antiparallel_indices = jnp.array(antiparallel_indices)
n_parallel = len(parallel_indices[0])
n_antiparallel = len(antiparallel_indices[0])

"""for debug."""
pos = jnp.array([1.1, 1.2, 1.3, 2.1, 2.2, 2.3])
atoms = jnp.array([[1, 1, 1], [2, 2, 2]])
#ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
charges=jnp.array([1, 1])
natoms=2
charges_jastrow = np.array(charges)
charges_indices_jastrow = np.arange(natoms)
atom_jastrow_indices = []
charged_jastrow_needed = []
for i in range(len(charges_indices_jastrow)):
    temp = np.repeat(charges_indices_jastrow[i], charges_jastrow[i])
    temp1 = np.repeat(charges_jastrow[i], charges_jastrow[i])
    atom_jastrow_indices.append(temp)
    charged_jastrow_needed.append(temp1)

atom_jastrow_indices = jnp.array(np.hstack(np.array(atom_jastrow_indices)))
charged_jastrow_needed = jnp.array(np.hstack(np.array(charged_jastrow_needed)))


network = make_ai_net(ndim=3,
                      natoms=2,
                      nelectrons=2,
                      num_angular=4,
                      n_parallel=n_parallel,
                      n_antiparallel=n_antiparallel,
                      parallel_indices=parallel_indices,
                      antiparallel_indices=antiparallel_indices,
                      atom_jastrow_indices=atom_jastrow_indices,
                      charged_jastrow_needed=charged_jastrow_needed,
                      charges=jnp.array([1, 1]),
                      full_det=True)

params = network.init(key)
output = network.apply(params, pos, atoms, charges=jnp.array([1, 1]))
'''