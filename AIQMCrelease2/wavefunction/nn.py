# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Fermionic Neural Network in JAX."""
import enum
import functools
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import attr
import chex
from AIQMCrelease1.wavefunction_f import envelopes
from AIQMCrelease1.wavefunction_f import jastrows
from AIQMCrelease1.wavefunction_f import network_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol

AILayers = Tuple[Tuple[int, int], ...]
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]


@chex.dataclass
class AINetData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


class InitAINet(Protocol):

    def __call__(self, key: chex.PRNGKey) -> ParamTree:
        """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class AINetLike(Protocol):
    def __call__(
            self,
            params: ParamTree,
            electrons: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray, ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the sign and log magnitude of the wavefunction."""


class LogAINetLike(Protocol):
    def __call__(
            self,
            params: ParamTree,
            electrons: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray, ) -> jnp.ndarray:
        """Returns the log magnitude of the wavefunction."""


class OrbitalAILike(Protocol):
    def __call__(
            self,
            params: ParamTree,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray, ) -> Sequence[jnp.ndarray]:
        """Forward evaluation of the Fermionic Neural Network up to the orbitals."""


class InitLayersAI(Protocol):
    def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        """Returns output dim and initialized parameters for the interaction layers."""


class ApplyLayersAI(Protocol):
    def __call__(
            self,
            params: ParamTree,
            ae: jnp.ndarray,
            r_ae: jnp.ndarray,
            ee: jnp.ndarray,
            r_ee: jnp.ndarray,
            spins: jnp.ndarray,
            charges: jnp.ndarray, ) -> jnp.ndarray:
        """Forward evaluation of the equivariant interaction layers."""


class FeatureInit(Protocol):
    def __call__(self) -> Tuple[Tuple[int, int], Param]:
        """Creates the learnable parameters for the feature input layer. """


class FeatureApply(Protocol):
    def __call__(
            self,
            ae: jnp.ndarray,
            r_ae: jnp.ndarray,
            ee: jnp.ndarray,
            r_ee: jnp.ndarray,
            **params: jnp.ndarray, ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Creates the features to pass into the network."""


@attr.s(auto_attribs=True)
class FeatureLayer:
    init: FeatureInit
    apply: FeatureApply


class FeatureLayerType(enum.Enum):
    STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):
    def __call__(
            self,
            natoms: int,
            nspins: Sequence[int],
            ndim: int,
            **kwargs: Any,
    ) -> FeatureLayer:
        """Builds the FeatureLayer object."""


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
    """Options controlling the overall network architecture."""
    ndim: int = 3
    determinants: int = 1
    states: int = 0
    full_det: bool = True
    rescale_inputs: bool = False
    bias_orbitals: bool = False
    envelope: envelopes.Envelope = attr.ib(default=attr.Factory(envelopes.make_isotropic_envelope, takes_self=False))
    feature_layer: FeatureLayer = None
    complex_output: bool = False


@attr.s(auto_attribs=True, kw_only=True)
class AINetOptions(BaseNetworkOptions):
    """Options controlling the FermiNet architecture."""
    hidden_dims: AILayers = ((256, 32), (256, 32), (256, 32), (256, 32))
    separate_spin_channels: bool = False
    schnet_electron_electron_convolutions: Tuple[int, ...] = ()
    nuclear_embedding_dim: int = 0
    electron_nuclear_aux_dims: Tuple[int, ...] = ()
    schnet_electron_nuclear_convolutions: Tuple[int, ...] = ()
    use_last_layer: bool = False


@attr.s(auto_attribs=True)
class Network:
    options: BaseNetworkOptions
    init: InitAINet
    apply: AINetLike
    orbitals: OrbitalAILike


def _split_spin_pairs(
        arr: jnp.ndarray,
        nspins: Tuple[int, int], ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Splits array into parallel and anti-parallel spin channels."""
    if len(nspins) != 2:
        raise ValueError(
            'Separate spin channels has not been verified with spin sampling.')
    up_up, up_down, down_up, down_down = network_blocks.split_into_blocks(arr, nspins)
    trailing_dims = jnp.shape(arr)[2:]
    parallel_spins = [
        up_up.reshape((-1,) + trailing_dims),
        down_down.reshape((-1,) + trailing_dims),
    ]
    antiparallel_spins = [
        up_down.reshape((-1,) + trailing_dims),
        down_up.reshape((-1,) + trailing_dims),
    ]
    return (jnp.concatenate(parallel_spins, axis=0),
            jnp.concatenate(antiparallel_spins, axis=0),)


def _combine_spin_pairs(
        parallel_spins: jnp.ndarray,
        antiparallel_spins: jnp.ndarray,
        nspins: Tuple[int, int], ) -> jnp.ndarray:
    """Combines arrays of parallel spins and antiparallel spins."""
    if len(nspins) != 2:
        raise ValueError('Separate spin channels has not been verified with spin sampling.')
    nsame_pairs = [nspin ** 2 for nspin in nspins]
    same_pair_partitions = network_blocks.array_partitions(nsame_pairs)
    up_up, down_down = jnp.split(parallel_spins, same_pair_partitions, axis=0)
    up_down, down_up = jnp.split(antiparallel_spins, 2, axis=0)
    trailing_dims = jnp.shape(parallel_spins)[1:]
    up = jnp.concatenate((
        up_up.reshape((nspins[0], nspins[0]) + trailing_dims),
        up_down.reshape((nspins[0], nspins[1]) + trailing_dims),
    ), axis=1, )
    down = jnp.concatenate((
        down_up.reshape((nspins[1], nspins[0]) + trailing_dims),
        down_down.reshape((nspins[1], nspins[1]) + trailing_dims),
    ), axis=1, )
    return jnp.concatenate((up, down), axis=0)


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


def make_ainet_features(
        natoms: int,
        nspins: Optional[Tuple[int, int]] = None,
        ndim: int = 3,
        rescale_inputs: bool = False, ) -> FeatureLayer:
    """Returns the init and apply functions for the standard features."""
    del nspins

    def init() -> Tuple[Tuple[int, int], Param]:
        return (natoms * (ndim + 1), ndim + 1), {}

    def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if rescale_inputs:
            log_r_ae = jnp.log(1 + r_ae)
            ae_features = jnp.concatenate((log_r_ae, ae * log_r_ae / r_ae), axis=2)
            log_r_ee = jnp.log(1 + r_ee)
            ee_features = jnp.concatenate((log_r_ee, ee * log_r_ee / r_ee), axis=2)
        else:
            ae_features = jnp.concatenate((r_ae, ae), axis=2)
            ee_features = jnp.concatenate((r_ee, ee), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features, ee_features

    return FeatureLayer(init=init, apply=apply)


def construct_symmetric_features(
        h_one: jnp.ndarray,
        h_two: jnp.ndarray,
        nspins: Tuple[int, int],
        h_aux: Optional[jnp.ndarray], ) -> jnp.ndarray:
    """Combines intermediate features from rank-one and -two streams."""
    spin_partitions = network_blocks.array_partitions(nspins)
    h_ones = jnp.split(h_one, spin_partitions, axis=0)
    h_twos = jnp.split(h_two, spin_partitions, axis=0)
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]
    features = [h_one] + g_one + g_two
    if h_aux is not None:
        features.append(h_aux)
    return jnp.concatenate(features, axis=1)


def make_schnet_convolution(nspins: Tuple[int, int], separate_spin_channels: bool) -> ...:
    """Returns init/apply pair for SchNet-style convolutions."""

    def init(key: chex.PRNGKey, dims_one: int, dims_two: int, embedding_dim: int) -> ParamTree:
        """Returns parameters for learned Schnet convolutions."""
        nchannels = 2 if separate_spin_channels else 1
        key_one, *key_two = jax.random.split(key, num=nchannels + 1)
        h_one_kernel = network_blocks.init_linear_layer(key_one, in_dim=dims_one, out_dim=embedding_dim,
                                                        include_bias=False)
        h_two_kernels = []
        for i in range(nchannels):
            h_two_kernels.append(
                network_blocks.init_linear_layer(
                    key_two[i],
                    in_dim=dims_two,
                    out_dim=embedding_dim,
                    include_bias=False, ))
        return {'single': h_one_kernel['w'],
                'double': [kernel['w'] for kernel in h_two_kernels], }

    def apply(params: ParamTree, h_one: jnp.ndarray, h_two: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """Applies the convolution B h_two . C h_one."""
        h_one_embedding = network_blocks.linear_layer(h_one, params['single'])
        h_two_embeddings = [network_blocks.linear_layer(h_two_channel, layer_param)
                            for h_two_channel, layer_param in zip(h_two, params['double'])]
        if separate_spin_channels:
            h_two_embedding = _combine_spin_pairs(h_two_embeddings[0], h_two_embeddings[1], nspins)
        else:
            h_two_embedding = h_two_embeddings[0]
        return h_one_embedding * h_two_embedding

    return init, apply


def make_schnet_electron_nuclear_convolution() -> ...:
    """Returns init/apply pair for SchNet-style convolutions for electrons-ions.

  See Gerard et al, arXiv:2205.09438.
  """

    def init(
            key: chex.PRNGKey,
            electron_nuclear_dim: int,
            nuclear_dim: int,
            embedding_dim: int,
    ) -> Param:
        key1, key2 = jax.random.split(key)
        return {
            'electron_ion_embedding': network_blocks.init_linear_layer(
                key1,
                in_dim=electron_nuclear_dim,
                out_dim=embedding_dim,
                include_bias=False,
            )['w'],
            'ion_embedding': network_blocks.init_linear_layer(
                key2, in_dim=nuclear_dim, out_dim=embedding_dim, include_bias=False
            )['w'],
        }

    def apply(
            params: Param, h_ion_nuc: jnp.ndarray, nuc_embedding: jnp.ndarray
    ) -> jnp.ndarray:
        ion_nuc_conv = (h_ion_nuc @ params['electron_ion_embedding']) * (
                nuc_embedding[None] @ params['ion_embedding']
        )
        return jnp.sum(ion_nuc_conv, axis=1)

    return init, apply


def make_fermi_net_layers(
        nspins: Tuple[int, int], natoms: int, options: AINetOptions
) -> Tuple[InitLayersAI, ApplyLayersAI]:
    """Creates the permutation-equivariant and interaction layers for FermiNet."""

    schnet_electron_init, schnet_electron_apply = make_schnet_convolution(
        nspins=nspins, separate_spin_channels=options.separate_spin_channels
    )
    schnet_electron_nuclear_init, schnet_electron_nuclear_apply = (
        make_schnet_electron_nuclear_convolution()
    )

    if all(
            len(hidden_dims) != len(options.hidden_dims[0])
            for hidden_dims in options.hidden_dims
    ):
        raise ValueError(
            'Each layer does not have the same number of streams: '
            f'{options.hidden_dims}'
        )

    if options.use_last_layer:
        num_convolutions = len(options.hidden_dims) + 1
    else:
        num_convolutions = len(options.hidden_dims)
    if (
            options.schnet_electron_electron_convolutions
            and len(options.schnet_electron_electron_convolutions) != num_convolutions
    ):
        raise ValueError(
            'Inconsistent number of layers for convolution and '
            'one- and two-electron streams. '
            f'{len(options.schnet_electron_electron_convolutions)=}, '
            f'expected {num_convolutions} layers.'
        )
    e_ion_options = (
        options.nuclear_embedding_dim,
        options.electron_nuclear_aux_dims,
        options.schnet_electron_nuclear_convolutions,
    )
    if any(e_ion_options) != all(e_ion_options):
        raise ValueError(
            'A subset of options set for electron-ion '
            'auxiliary stream: '
            f'{options.nuclear_embedding_dim=} '
            f'{options.electron_nuclear_aux_dims=} '
            f'{options.schnet_electron_nuclear_convolutions=}'
        )
    if (
            options.electron_nuclear_aux_dims
            and len(options.electron_nuclear_aux_dims) < num_convolutions - 1
    ):
        raise ValueError(
            'Too few layers in electron-nuclear auxiliary stream. '
            f'{options.electron_nuclear_aux_dims=}, '
            f'expected {num_convolutions - 1} layers.'
        )
    if (
            options.schnet_electron_nuclear_convolutions
            and len(options.schnet_electron_nuclear_convolutions) != num_convolutions
    ):
        raise ValueError(
            'Inconsistent number of layers for convolution and '
            'one- and two-electron streams. '
            f'{len(options.schnet_electron_nuclear_convolutions)=}, '
            f'expected {num_convolutions} layers.'
        )

    def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        """Returns tuple of output dimension from the final layer and parameters."""

        params = {}
        key, nuclear_key = jax.random.split(key, num=2)
        (num_one_features, num_two_features), params['input'] = (
            options.feature_layer.init()
        )
        if options.nuclear_embedding_dim:
            # Gerard et al project each nuclear charge to a separate vector.
            params['nuclear'] = network_blocks.init_linear_layer(
                nuclear_key,
                in_dim=1,
                out_dim=options.nuclear_embedding_dim,
                include_bias=True,
            )

        nchannels = len([nspin for nspin in nspins if nspin > 0])

        def nfeatures(out1, out2, aux):
            return (nchannels + 1) * out1 + nchannels * out2 + aux

        dims_one_in = num_one_features
        dims_two_in = num_two_features

        dims_e_aux_in = num_one_features // natoms

        key, subkey = jax.random.split(key)
        layers = []
        for i in range(len(options.hidden_dims)):
            layer_params = {}
            key, single_key, *double_keys, aux_key = jax.random.split(key, num=5)

            if options.schnet_electron_electron_convolutions:
                key, subkey = jax.random.split(key)
                layer_params['schnet'] = schnet_electron_init(
                    subkey,
                    dims_one=dims_one_in,
                    dims_two=dims_two_in,
                    embedding_dim=options.schnet_electron_electron_convolutions[i],
                )
                dims_two_embedding = options.schnet_electron_electron_convolutions[i]
            else:
                dims_two_embedding = dims_two_in
            if options.schnet_electron_nuclear_convolutions:
                key, subkey = jax.random.split(key)
                layer_params['schnet_nuclear'] = schnet_electron_nuclear_init(
                    subkey,
                    electron_nuclear_dim=dims_e_aux_in,
                    nuclear_dim=options.nuclear_embedding_dim,
                    embedding_dim=options.schnet_electron_nuclear_convolutions[i],
                )
                dims_aux = options.schnet_electron_nuclear_convolutions[i]
            else:
                dims_aux = 0

            dims_one_in = nfeatures(dims_one_in, dims_two_embedding, dims_aux)

            dims_one_out, dims_two_out = options.hidden_dims[i]
            layer_params['single'] = network_blocks.init_linear_layer(
                single_key,
                in_dim=dims_one_in,
                out_dim=dims_one_out,
                include_bias=True,
            )
            if i < len(options.hidden_dims) - 1 or options.use_last_layer:
                ndouble_channels = 2 if options.separate_spin_channels else 1
                layer_params['double'] = []
                for ichannel in range(ndouble_channels):
                    layer_params['double'].append(
                        network_blocks.init_linear_layer(
                            double_keys[ichannel],
                            in_dim=dims_two_in,
                            out_dim=dims_two_out,
                            include_bias=True,
                        )
                    )
                if not options.separate_spin_channels:
                    layer_params['double'] = layer_params['double'][0]
                if options.electron_nuclear_aux_dims:
                    layer_params['electron_ion'] = network_blocks.init_linear_layer(
                        aux_key,
                        in_dim=dims_e_aux_in,
                        out_dim=options.electron_nuclear_aux_dims[i],
                        include_bias=True,
                    )
                    dims_e_aux_in = options.electron_nuclear_aux_dims[i]

            layers.append(layer_params)
            dims_one_in = dims_one_out
            dims_two_in = dims_two_out

        if options.use_last_layer:
            layers.append({})
            if options.schnet_electron_electron_convolutions:
                key, subkey = jax.random.split(key)
                layers[-1]['schnet'] = schnet_electron_init(
                    subkey,
                    dims_one=dims_one_in,
                    dims_two=dims_two_in,
                    embedding_dim=options.schnet_electron_electron_convolutions[-1],
                )
                dims_two_in = options.schnet_electron_electron_convolutions[-1]
            if options.schnet_electron_nuclear_convolutions:
                key, subkey = jax.random.split(key)
                layers[-1]['schnet_nuclear'] = schnet_electron_nuclear_init(
                    subkey,
                    electron_nuclear_dim=dims_e_aux_in,
                    nuclear_dim=options.nuclear_embedding_dim,
                    embedding_dim=options.schnet_electron_nuclear_convolutions[-1],
                )
                dims_aux = options.schnet_electron_nuclear_convolutions[-1]
            else:
                dims_aux = 0
            output_dims = nfeatures(dims_one_in, dims_two_in, dims_aux)
        else:
            # Pass output of the one-electron stream straight to orbital shaping.
            output_dims = dims_one_in

        params['streams'] = layers

        return output_dims, params

    def electron_electron_convolution(
            params: ParamTree,
            h_one: jnp.ndarray,
            h_two: Tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        if options.schnet_electron_electron_convolutions:
            h_two_embedding = schnet_electron_apply(params['schnet'], h_one, h_two)
        elif options.separate_spin_channels:
            h_two_embedding = _combine_spin_pairs(h_two[0], h_two[1], nspins)
        else:
            h_two_embedding = h_two[0]
        return h_two_embedding

    def apply_layer(
            params: Mapping[str, ParamTree],
            h_one: jnp.ndarray,
            h_two: Tuple[jnp.ndarray, ...],
            h_elec_ion: Optional[jnp.ndarray],
            nuclear_embedding: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...], Optional[jnp.ndarray]]:
        if options.separate_spin_channels:
            assert len(h_two) == 2
        else:
            assert len(h_two) == 1

        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
        h_two_embedding = electron_electron_convolution(params, h_one, h_two)
        if options.schnet_electron_nuclear_convolutions:
            h_aux = schnet_electron_nuclear_apply(
                params['schnet_nuclear'], h_elec_ion, nuclear_embedding
            )
        else:
            h_aux = None
        h_one_in = construct_symmetric_features(
            h_one, h_two_embedding, nspins, h_aux=h_aux
        )

        h_one_next = jnp.tanh(
            network_blocks.linear_layer(h_one_in, **params['single'])
        )

        h_one = residual(h_one, h_one_next)

        if 'double' in params:
            if options.separate_spin_channels:
                params_double = params['double']
            else:
                params_double = [params['double']]
            h_two_next = [
                jnp.tanh(network_blocks.linear_layer(prev, **param))
                for prev, param in zip(h_two, params_double)
            ]
            h_two = tuple(residual(prev, new) for prev, new in zip(h_two, h_two_next))
        if h_elec_ion is not None and 'electron_ion' in params:
            h_elec_ion = network_blocks.linear_layer(
                h_elec_ion, **params['electron_ion']
            )

        return h_one, h_two, h_elec_ion

    def apply(
            params,
            *,
            ae: jnp.ndarray,
            r_ae: jnp.ndarray,
            ee: jnp.ndarray,
            r_ee: jnp.ndarray,
            spins: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the FermiNet interaction layers to a walker configuration. """
        del spins  # Unused.

        ae_features, ee_features = options.feature_layer.apply(
            ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
        )

        if options.electron_nuclear_aux_dims:
            h_elec_ion = jnp.reshape(ae_features, (ae_features.shape[0], natoms, -1))
        else:
            h_elec_ion = None

        h_one = ae_features

        if options.separate_spin_channels:
            h_two = _split_spin_pairs(ee_features, nspins)
        else:
            h_two = [ee_features]
        if options.nuclear_embedding_dim:
            nuclear_embedding = network_blocks.linear_layer(
                charges[:, None], **params['nuclear']
            )
        else:
            nuclear_embedding = None

        for i in range(len(options.hidden_dims)):
            h_one, h_two, h_elec_ion = apply_layer(
                params['streams'][i],
                h_one,
                h_two,
                h_elec_ion,
                nuclear_embedding,
            )

        if options.use_last_layer:
            last_layer = params['streams'][-1]
            h_two_embedding = electron_electron_convolution(last_layer, h_one, h_two)
            if options.schnet_electron_nuclear_convolutions:
                h_aux = schnet_electron_nuclear_apply(
                    last_layer['schnet_nuclear'], h_elec_ion, nuclear_embedding
                )
            else:
                h_aux = None
            h_to_orbitals = construct_symmetric_features(
                h_one, h_two_embedding, nspins, h_aux=h_aux
            )
        else:
            h_to_orbitals = h_one

        return h_to_orbitals

    return init, apply


def make_orbitals(
        nspins: Tuple[int, int],
        charges: jnp.ndarray,
        options: BaseNetworkOptions,
        equivariant_layers: Tuple[InitLayersAI, ApplyLayersAI],) -> ...:
    """Returns init, apply pair for orbitals."""

    equivariant_layers_init, equivariant_layers_apply = equivariant_layers

    def init(key: chex.PRNGKey) -> ParamTree:
        """Returns initial random parameters for creating orbitals."""
        key, subkey = jax.random.split(key)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

        active_spin_channels = [spin for spin in nspins if spin > 0]
        nchannels = len(active_spin_channels)
        if nchannels == 0:
            raise ValueError('No electrons present!')

        nspin_orbitals = []
        num_states = max(options.states, 1)
        for nspin in active_spin_channels:
            if options.full_det:
                norbitals = sum(nspins) * options.determinants * num_states

            else:
                norbitals = nspin * options.determinants * num_states
            if options.complex_output:
                norbitals *= 2
            nspin_orbitals.append(norbitals)

        # create envelope params
        natom = charges.shape[0]
        if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
            output_dims = dims_orbital_in
        elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
            if options.complex_output:
                output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
            else:
                output_dims = nspin_orbitals
        else:
            raise ValueError('Unknown envelope type')
        params['envelope'] = options.envelope.init(
            natom=natom, output_dims=output_dims, ndim=options.ndim
        )

        orbitals = []
        diffuse_coefficients = []
        for nspin_orbital in nspin_orbitals:
            key, subkey = jax.random.split(key)
            orbitals.append(network_blocks.init_linear_layer(
                subkey,
                in_dim=dims_orbital_in,
                out_dim=nspin_orbital,
                include_bias=options.bias_orbitals,))
            diffuse_coefficients.append(network_blocks.init_linear_layer(
                subkey,
                in_dim=dims_orbital_in,
                out_dim=nspin_orbital,
                include_bias=True))

        params['orbital'] = orbitals
        params['diffuse'] = diffuse_coefficients

        return params

    def apply(
            params,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,) -> Sequence[jnp.ndarray]:
        """Forward evaluation of the Fermionic Neural Network up to the orbitals."""
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)

        h_to_orbitals = equivariant_layers_apply(
            params['layers'],
            ae=ae,
            r_ae=r_ae,
            ee=ee,
            r_ee=r_ee,
            spins=spins,
            charges=charges,
        )

        if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
            envelope_factor = options.envelope.apply(
                ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
            )
            h_to_orbitals = envelope_factor * h_to_orbitals

        h_to_orbitals = jnp.split(
            h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
        )

        h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
        active_spin_channels = [spin for spin in nspins if spin > 0]
        active_spin_partitions = network_blocks.array_partitions(active_spin_channels)

        orbitals = [network_blocks.linear_layer(h, **p) for h, p in zip(h_to_orbitals, params['orbital'])]
        diffuse_part = [network_blocks.linear_layer(h, **p) for h, p in zip(h_to_orbitals, params['diffuse'])]

        if options.complex_output:
            orbitals = [orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals]
            diffuse_part = [diffuse[..., ::2] + 1.0j * diffuse[..., 1::2] for diffuse in diffuse_part]
        """we are going to rewrite the envelope part. Before we made some mistakes about this part. 21.1.2025."""
        if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
            ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
            r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
            r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
            for i in range(len(active_spin_channels)):
                orbitals[i] = orbitals[i] * options.envelope.apply(
                    ae=ae_channels[i],
                    r_ae=r_ae_channels[i],
                    r_ee=r_ee_channels[i],
                    **params['envelope'][i],
                )

        # Reshape into matrices.
        shapes = [(spin, -1, sum(nspins) if options.full_det else spin) for spin in active_spin_channels]
        orbitals = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)]
        orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
        diffuse_part = [jnp.reshape(diffuse, shape) for diffuse, shape in zip(diffuse_part, shapes)]
        diffuse_part = [jnp.transpose(diffuse, (1, 0, 2)) for diffuse in diffuse_part]
        if options.full_det:
            #orbitals = [jnp.concatenate(orbitals, axis=1)]
            orbitals = jnp.concatenate(orbitals, axis=1)
            diffuse_part = jnp.concatenate(diffuse_part, axis=1)

        diffuse_part = jnp.exp(-1 * diffuse_part) + 1
        orbitals = [orbitals * diffuse_part]

        return orbitals

    return init, apply


def make_AI_net(
        nspins: Tuple[int, int],
        charges: jnp.ndarray,
        *,
        ndim: int = 3,
        determinants: int = 1,
        states: int = 0,
        envelope: Optional[envelopes.Envelope] = None,
        feature_layer: Optional[FeatureLayer] = None,
        complex_output: bool = False,
        bias_orbitals: bool = False,
        full_det: bool = True,
        rescale_inputs: bool = False,
        hidden_dims: AILayers = ((4, 2), (4, 2), (4, 2)),
        use_last_layer: bool = False,
        separate_spin_channels: bool = False,
        schnet_electron_electron_convolutions: Tuple[int, ...] = tuple(),
        electron_nuclear_aux_dims: Tuple[int, ...] = tuple(),
        nuclear_embedding_dim: int = 0,
        schnet_electron_nuclear_convolutions: Tuple[int, ...] = tuple(),) -> Network:

    """Creates functions for initializing parameters and evaluating ferminet."""
    if sum([nspin for nspin in nspins if nspin > 0]) == 0:
        raise ValueError('No electrons present!')

    if not envelope:
        envelope = envelopes.make_isotropic_envelope()

    if not feature_layer:
        natoms = charges.shape[0]
        feature_layer = make_ainet_features(
            natoms, nspins, ndim=ndim, rescale_inputs=rescale_inputs
        )

    options = AINetOptions(
        ndim=ndim,
        determinants=determinants,
        states=states,
        rescale_inputs=rescale_inputs,
        envelope=envelope,
        feature_layer=feature_layer,
        complex_output=complex_output,
        bias_orbitals=bias_orbitals,
        full_det=full_det,
        hidden_dims=hidden_dims,
        separate_spin_channels=separate_spin_channels,
        schnet_electron_electron_convolutions=schnet_electron_electron_convolutions,
        electron_nuclear_aux_dims=electron_nuclear_aux_dims,
        nuclear_embedding_dim=nuclear_embedding_dim,
        schnet_electron_nuclear_convolutions=schnet_electron_nuclear_convolutions,
        use_last_layer=use_last_layer,
    )

    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
        if options.bias_orbitals:
            raise ValueError('Cannot bias orbitals w/STO envelope.')

    equivariant_layers = make_fermi_net_layers(nspins, charges.shape[0], options)

    orbitals_init, orbitals_apply = make_orbitals(
        nspins=nspins,
        charges=charges,
        options=options,
        equivariant_layers=equivariant_layers,
    )

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(subkey)

    def apply(
            params,
            pos: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward evaluation of the Fermionic Neural Network for a single datum."""

        orbitals = orbitals_apply(params, pos, spins, atoms, charges)
        return network_blocks.logdet_matmul(orbitals)

    return Network(options=options, init=init, apply=apply, orbitals=orbitals_apply)


feature_layer = make_ainet_features(
    natoms=2,
    nspins=(1, 1),
    ndim=3,)

network = make_AI_net(
        nspins=(1, 1),
        charges=jnp.array([1, 1]),
        ndim=3,
        determinants=1,
        feature_layer=feature_layer)

key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
params = network.init(subkey)
signed_network = network.apply
pos = jnp.array([1.1, 1.2, 1.3, 2.1, 2.2, 2.3])
atoms = jnp.array([[1, 1, 1], [2, 2, 2]])
charges = jnp.array([1, 1])
spins = jnp.array([1, -1])
output = network.apply(params, pos, spins, atoms, charges)
jax.debug.print("output:{}", output)