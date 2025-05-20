import enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
import chex
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
from GaussianNet.wavefunction import network_blocks
from GaussianNet.wavefunction import JastrowPade
from GaussianNet.wavefunction import envelopes

GaussianLayers = Tuple[Tuple[int, int], ...]
AngularLayers = Tuple[Tuple[int], ...]
PolyexponentLayers = Tuple[Tuple[int], ...]
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]

@chex.dataclass
class GaussianNetData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


class InitLayersGn(Protocol):

    def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        """"""


class ApplyLayersGn(Protocol):

    def __call__(self,
                 params: ParamTree,
                 ae: jnp.ndarray,
                 r_ae: jnp.ndarray,
                 ee: jnp.ndarray,
                 r_ee: jnp.ndarray,
                 spins: jnp.ndarray,
                 charges: jnp.ndarray,) -> jnp.ndarray:
        """"""


class FeatureInit(Protocol):

    def __call__(self,) -> Tuple[Tuple[int, int], Param]:
        """"""


class FeatureApply(Protocol):

    def __call__(self,
                 ae: jnp.ndarray,
                 r_ae: jnp.ndarray,
                 ee: jnp.ndarray,
                 r_ee: jnp.ndarray,
                 **params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """"""


@attr.s(auto_attribs=True)
class FeatureLayer:
    init: FeatureInit
    apply: FeatureApply


class InitGaussianNet(Protocol):

    def __call__(self, key: chex.PRNGKey) -> ParamTree:
        """"""


class GaussianNetLike(Protocol):

    def __call__(self,
                 params: ParamTree,
                 electrons: jnp.ndarray,
                 spins: jnp.ndarray,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """"""


class LogGaussianNetLike(Protocol):

    def __call__(self,
                 params: ParamTree,
                 electrons: jnp.ndarray,
                 spins: jnp.ndarray,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,) -> jnp.ndarray:
        """'"""


class OrbitalGnLike(Protocol):

    def __call__(self,
                 params: ParamTree,
                 pos: jnp.ndarray,
                 spins: jnp.ndarray,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,
                 ) -> Sequence[jnp.ndarray]:
        """"""


@attr.s(auto_attribs=True)
class Network:
    init: InitGaussianNet
    apply: GaussianNetLike
    orbitals: OrbitalGnLike


def cartesian_to_spherical(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    phi = jnp.arccos(z / r)
    theta = jnp.arctan(y, x) + jnp.pi
    return r, phi, theta


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


def y_0(theta, phi):
    return 1/2 * jnp.sqrt(1/jnp.pi)


def y_1(theta, phi):
    return jnp.array([1/2 * jnp.sqrt(3/(2 * jnp.pi)) * jnp.exp(-1 * 1.j * phi) * jnp.sin(theta),
                      1/2 * jnp.sqrt(3/jnp.pi) * jnp.cos(theta),
                      1/2 * jnp.sqrt(3/(2 * jnp.pi)) * jnp.exp(1.j * phi) * jnp.sin(theta)])


def y_2(theta, phi):
    return 1/4 * jnp.sqrt(15/(2 * jnp.pi)) * jnp.exp(-1 * 2 * 1.j * phi) * jnp.sin(theta)**2, \
           1/2 * jnp.sqrt(15/(2 * jnp.pi)) * jnp.exp(-1 * 1.j * phi) * jnp.sin(theta) * jnp.cos(theta), \
           1/4 * jnp.sqrt(5/jnp.pi)*(3 * jnp.cos(theta)**2 - 1), \
           -1/2 * jnp.sqrt(15/(2 * jnp.pi)) * jnp.exp(1.j * phi) * jnp.sin(theta) * jnp.cos(theta), \
           1/4 * jnp.sqrt(15/(2 * jnp.pi)) * jnp.exp(2 * 1.j * phi) * jnp.sin(theta)**2


def y_0(x: jnp.ndarray):
    """
    to be continued... 19.2.2025
    :param x: ae the cartesian  coordinate
            y: r_ae
    :return: d and f orbitals
    """
    return jnp.array([1 / 2 * jnp.sqrt(1 / jnp.pi)])


def y_sp(x: jnp.ndarray):
    """x: x/r, y/r, z/r"""
    return jnp.array([1 / 2 * jnp.sqrt(1 / jnp.pi),
                      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x[0],
                      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x[1],
                      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x[2]])


y_sp_parallel = jax.vmap(jax.vmap(y_sp, in_axes=0), in_axes=0)


def exp_feature(x: jnp.array):
    """
    :param x: r_ae
    :return:
    """
    return jnp.array([0.0051583 * jnp.exp(-1 * 13.073594 * x),
                      0.0603424 * jnp.exp(-1 * 6.541187 * x),
                      -0.1978471 * jnp.exp(-1 * 4.573411 * x),
                      -0.0810340 * jnp.exp(-1 * 1.637494 * x),
                      0.2321726 * jnp.exp(-1 * 0.819297 * x),
                      0.2914643 * jnp.exp(-1 * 0.409924 * x),
                      0.4336405 * jnp.exp(-1 * 0.231300 * x),
                      0.2131940 * jnp.exp(-1 * 0.102619 * x),
                      0.0049848 * jnp.exp(-1 * 0.051344 * x),
                      1.000000 * jnp.exp(-1 * 0.127852 * x),
                      0.0209076 * jnp.exp(-1 * 9.934169 * x),
                      0.0572698 * jnp.exp(-1 * 3.886955 * x),
                      0.1122682 * jnp.exp(-1 * 1.871016 * x),
                      0.2130082 * jnp.exp(-1 * 0.935757 * x),
                      0.2835815 * jnp.exp(-1 * 0.468003 * x),
                      0.3011207 * jnp.exp(-1 * 0.239473 * x),
                      0.2016934 * jnp.exp(-1 * 0.117063 * x),
                      0.0453575 * jnp.exp(-1 * 0.058547 * x),
                      0.0029775 * jnp.exp(-1 * 0.029281 * x),
                      1.000000 * jnp.exp(-1 * 0.149161 * x),
                      1.000000 * jnp.exp(-1 * 0.561160 * x)])


def y_2(x: jnp.ndarray):
    return jnp.array([1 / 2 * jnp.sqrt(15 / jnp.pi) * (x[0] * x[1]),
                      1 / 2 * jnp.sqrt(15 / jnp.pi) * (x[1] * x[2]),
                      1 / 4 * jnp.sqrt(5 / jnp.pi) * (3 * x[2]**2 - 1),
                      1 / 2 * jnp.sqrt(15 / jnp.pi) * (x[0] * x[2]),
                      1 / 4 * jnp.sqrt(15 / jnp.pi) * (x[0]**2 - x[1]**2)])


def make_gaussian_features(natoms: int, ndim: int = 3):

    def init() -> Tuple[Tuple[int, int], Param]:
        return (natoms * (ndim + 1), ndim + 1), {}

    def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ae_features = jnp.concatenate((r_ae, ae), axis=2)
        ee_features = jnp.concatenate((r_ee, ee), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features, ee_features
    return FeatureLayer(init=init, apply=apply)


def construct_symmetric_features(
        h_one: jnp.ndarray,
        h_two: jnp.ndarray,
        nspins: Tuple[int, int],
) -> jnp.ndarray:
    spin_partitions = network_blocks.array_partitions(nspins)
    h_ones = jnp.split(h_one, spin_partitions, axis=0)
    h_twos = jnp.split(h_two, spin_partitions, axis=0)
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]
    features = [h_one] + g_one + g_two
    return jnp.concatenate(features, axis=1)


def make_gaussian_net_layers(nspins: Tuple[int, int],
                             natoms: int,
                             nelectrons: int,
                             hidden_dims,
                             embedding_dim_eff_angular,
                             embedding_dim_poly_exponent,
                             feature_layer,):
    def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        params = {}
        key, angular_key, exponent_key = jax.random.split(key, num=3)
        (num_one_features, num_two_features), params['input'] = (feature_layer.init())
        """to be continued...8.5.2025."""
        nchannels = len([nspin for nspin in nspins if nspin > 0])

        def nfeatures(out1, out2):
            return (nchannels + 1) * out1 + nchannels * out2

        dims_one_in = num_one_features
        dims_two_in = num_two_features
        key, subkey = jax.random.split(key)
        layers = []

        for i in range(len(hidden_dims)):
            layer_params = {}
            key, single_key, *double_keys = jax.random.split(key, num=3)
            dims_one_in = nfeatures(dims_one_in, dims_two_in)
            dims_one_out, dims_two_out = hidden_dims[i]
            layer_params['single'] = network_blocks.init_linear_layer(
                single_key,
                in_dim=dims_one_in,
                out_dim=dims_one_out,
                include_bias=True,
            )

            if i < len(hidden_dims) - 1:
                ndouble_channels = 1
                layer_params['double'] = []
                for ichannel in range(ndouble_channels):
                    layer_params['double'].append(network_blocks.init_linear_layer(double_keys[ichannel],
                                                                                   in_dim=dims_two_in,
                                                                                   out_dim=dims_two_out,
                                                                                   include_bias=True,))
                    layer_params['double'] = layer_params['double'][0]

            layers.append(layer_params)
            dims_one_in = dims_one_out
            dims_two_in = dims_two_out
        output_dims = dims_one_in
        params['embedding_layer'] = layers
        return output_dims, params

    def apply_layer(
            params: Mapping[str, ParamTree],
            h_one: jnp.ndarray,
            h_two: Tuple[jnp.ndarray, ...],
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        assert len(h_two) == 1
        """to be continued... 9.5.2025"""
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y #the shape of x must be same with y
        h_two_embedding = h_two[0]
        h_one_in = construct_symmetric_features(h_one, h_two_embedding, nspins)
        h_one_next = jnp.tanh(network_blocks.linear_layer(h_one_in, **params['single']))
        h_one = residual(h_one, h_one_next)
        if 'double' in params:
            params_double = [params['double']]
            #jax.debug.print("params_double:{}", [params['double']])
            h_two_next = [jnp.tanh(network_blocks.linear_layer(prev, **param))
                          for prev, param in zip(h_two, params_double)]
            h_two = tuple(residual(prev, new) for prev, new in zip(h_two, h_two_next))

        return h_one, h_two

    def apply(params,
              ae: jnp.ndarray,
              r_ae: jnp.ndarray,
              ee: jnp.ndarray,
              r_ee: jnp.ndarray,
              charges: jnp.ndarray,):
        ae_features, ee_features = feature_layer.apply(
            ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
        )
        h_one = ae_features
        h_two = [ee_features]
        for i in range(len(hidden_dims)):
            h_one, h_two = apply_layer(params['embedding_layer'][i],
                                       h_one,
                                       h_two)
        h_to_orbitals = h_one
        return h_to_orbitals
    return init, apply


def make_embedding_angular_layers(nspins: Tuple[int, int],
                                  charges: jnp.ndarray,
                                  embedding_dim_eff_angular,
                                  equivariant_layers: Tuple[InitLayersGn, ApplyLayersGn],):
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey, subkey2 = jax.random.split(key, num=3)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

        layers_angular = []
        dims_one_in = 3 # the number of spherical harmonics
        for i in range(len(embedding_dim_eff_angular)):
            layer_params_angular = {}
            dims_one_out = embedding_dim_eff_angular[i]
            key, angular_key = jax.random.split(key, num=2)
            layer_params_angular['angular'] = network_blocks.init_linear_layer(
                angular_key,
                in_dim=dims_one_in,
                out_dim=dims_one_out,
                include_bias=True,
            )
            dims_one_in = dims_one_out
            layers_angular.append(layer_params_angular)
        params['angular_layer'] = layers_angular
        #jax.debug.print("params_angular_layer:{}", params['angular_layer'])

        return params

    def apply_layer(params: Mapping[str, ParamTree],
                    r_shift: jnp.ndarray):
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y  # the shape of x must be same with y
        r_shift_next = jnp.tanh(network_blocks.linear_layer(r_shift, **params['angular']))
        r_shift = residual(r_shift, r_shift_next)
        return r_shift

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray,):
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        y_one = ae / r_ae
        y_one = jnp.reshape(y_one, (6, 3))
        for i in range(len(embedding_dim_eff_angular)):
            y_one = apply_layer(params['angular_layer'][i], y_one)

        return y_one

    return init, apply


def make_exp_layers(embedding_dim_poly_exponent):

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey, subkey2 = jax.random.split(key, num=3)
        params = {}
        layers_exp = []
        dims_one_in = 21  # the number of exp features
        for i in range(len(embedding_dim_poly_exponent)):
            layer_params_exp = {}
            dims_one_out = embedding_dim_poly_exponent[i]
            key, exp_key = jax.random.split(key, num=2)
            layer_params_exp['exp'] = network_blocks.init_linear_layer(
                exp_key,
                in_dim=dims_one_in,
                out_dim=dims_one_out,
                include_bias=True,
            )
            dims_one_in = dims_one_out
            layers_exp.append(layer_params_exp)
        params['exp_layer'] = layers_exp
        return params

    def apply_layer(params: Mapping[str, ParamTree],
                    r_shift: jnp.ndarray):
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y  # the shape of x must be same with y
        r_shift_next = jnp.tanh(network_blocks.linear_layer(r_shift, **params['exp']))
        r_shift = residual(r_shift, r_shift_next)
        return r_shift

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray,):
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        #jax.debug.print("r_ae:{}", r_ae)
        exp_one = exp_feature(r_ae)
        #jax.debug.print("exp_one:{}", exp_one)
        #jax.debug.print("exp_one_shape:{}", exp_one.shape)
        exp_one = jnp.concatenate(exp_one, axis=1)
        #jax.debug.print("exp_one_shape:{}", exp_one.shape)
        #jax.debug.print("exp_one:{}", exp_one)
        exp_one = jnp.reshape(exp_one, (6, 21))
        for i in range(len(embedding_dim_poly_exponent)):
            exp_one = apply_layer(params['exp_layer'][i], exp_one)

        return exp_one

    return init, apply


def make_orbitals(nspins: Tuple[int, int],
                  charges: jnp.ndarray,
                  parallel_indices: jnp.array,
                  antiparallel_indices: jnp.array,
                  n_parallel: int,
                  n_antiparallel: int,
                  angular_init,
                  angular_apply,
                  exp_init,
                  exp_apply,
                  envelope,
                  equivariant_layers: Tuple[InitLayersGn, ApplyLayersGn],):
    """to be continued...11.5.2025."""
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    """the jastrow part needs to be done later.11.5.2025."""
    jastrow_ee_init, jastrow_ee_apply, jastrow_ae_init, jastrow_ae_apply = JastrowPade.get_jastrow(charges)


    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey, subsubkey, subsubsubkey = jax.random.split(key, num=4)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)
        params['angular'] = angular_init(subsubsubkey)
        params['exp'] = exp_init(key)
        active_spin_channels = [spin for spin in nspins if spin > 0]
        nchannels = len(active_spin_channels)
        nspin_orbitals = []
        for nspin in active_spin_channels:
            norbitals = sum(nspins) * 1 * 2
            nspin_orbitals.append(norbitals)

        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
        params['envelope'] = envelope.init(natom=1, output_dims=output_dims, ndim=3)
        #jax.debug.print("params_envelope:{}", params['envelope'])

        orbitals = []
        angular_map = []
        for nspin_orbital in nspin_orbitals:
            key, subkey, subsubkey = jax.random.split(key, num=3)
            orbitals.append(
                network_blocks.init_linear_layer(
                    subkey,
                    in_dim=dims_orbital_in,
                    out_dim=nspin_orbital,
                    include_bias=True
                )
            )
            angular_map.append(network_blocks.init_linear_layer(subsubkey,
                                                            in_dim=32,
                                                            out_dim=nspin_orbital,
                                                            include_bias=False))

        params['orbital'] = orbitals
        params['angular_map'] = angular_map
        params['jastrow_ee'] = jastrow_ee_init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
        params['jastrow_ae'] = jastrow_ae_init(nelectrons=6, natoms=1)
        return params

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray,):
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        h_to_orbitals = equivariant_layers_apply(params['layers'],
                                                 ae=ae,
                                                 r_ae=r_ae,
                                                 ee=ee,
                                                 r_ee=r_ee,
                                                 charges=charges)
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        angular_to_orbitals = angular_apply(params['angular'], pos, spins, atoms, charges)
        exp_to_orbitals = exp_apply(params['exp'], pos, spins, atoms, charges)
        """to be continued... 19.5.2025."""
        #jax.debug.print("exp_to_orbitals:{}", exp_to_orbitals)
        exp_to_orbitals = jnp.sum(exp_to_orbitals, axis=-1, keepdims=True)
        #jax.debug.print("exp_to_orbitals:{}", exp_to_orbitals)
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        h_to_orbitals = exp_to_orbitals * h_to_orbitals
        #jax.debug.print("angular_to_orbitals:{}", angular_to_orbitals)
        h_to_orbitals = jnp.split(h_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
        h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]

        angular_to_orbitals = jnp.split(angular_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
        angular_to_orbitals = [y for y, spin in zip(angular_to_orbitals, nspins) if spin > 0]

        angulars = [
            network_blocks.linear_layer(h, **p)
            for h, p in zip(angular_to_orbitals, params['angular_map'])
        ]
        #jax.debug.print("angular")
        orbitals = [
            network_blocks.linear_layer(h, **p)
            for h, p in zip(h_to_orbitals, params['orbital'])
        ]
        orbitals = [orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals]
        angulars = [angular[..., ::2] + 1.0j * angular[..., 1::2] for angular in angulars]
        #jax.debug.print("orbitals:{}", orbitals)
        #jax.debug.print("angulars:{}", angulars)

        orbitals_angular = []
        for i in range(len(orbitals)):
            orbitals_angular.append(orbitals[i] * angulars[i])
        """to be continued...16.5.2025"""
        """this line switch to the type with angular momentum functions."""

        active_spin_channels = [spin for spin in nspins if spin > 0]
        active_spin_partitions = network_blocks.array_partitions(active_spin_channels)
        ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
        r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
        r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
        for i in range(len(active_spin_channels)):
            orbitals_angular[i] = orbitals_angular[i] * envelope.apply(ae=ae_channels[i],
                                                                       r_ae=r_ae_channels[i],
                                                                       r_ee=r_ee_channels[i],
                                                                       **params['envelope'][i])

        #jax.debug.print("orbitals_angular:{}", orbitals_angular)
        shapes = [(spin, -1, sum(nspins)) for spin in active_spin_channels]
        orbitals_angular = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals_angular, shapes)]
        orbitals_angular = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals_angular]
        orbitals_angular = [jnp.concatenate(orbitals_angular, axis=1)]
        #jax.debug.print("orbitals_angular:{}", orbitals_angular)

        jastrow = jnp.exp(jastrow_ee_apply(r_ee=r_ee,
                                           parallel_indices=parallel_indices,
                                           antiparallel_indices=antiparallel_indices,
                                           params=params['jastrow_ee']) / 6)
        """to be continued... Jastrow 11.5.2025."""
        orbitals_angular_jastrow = [orbital * jastrow for orbital in orbitals_angular]

        return orbitals_angular_jastrow

    return init, apply



def make_gaussian_net(
        nspins: Tuple[int, int],
        charges: jnp.ndarray,
        parallel_indices: jnp.ndarray,
        antiparallel_indices: jnp.array,
        n_parallel: int,
        n_antiparallel: int,
        nelectrons: int = 6,
        natoms: int = 1,
        ndim: int = 3,
        determinants: int = 1,
        bias_orbitals: bool = False,
        full_det: bool = True,
        hidden_dims: GaussianLayers = ((32, 16), (32, 16), (32, 16), (32, 16)),
        embedding_dim_eff_angular: AngularLayers = ((32), (32), (32), (32)),# 3 is the number of coordinates.
        embedding_dim_poly_exponent: PolyexponentLayers = ((21), (21), (21), (21))):
    """The main function to create the many-body wave-function."""
    feature_layer = make_gaussian_features(natoms=natoms, ndim=ndim)
    """we only use isotropic function currently."""
    envelope = envelopes.make_isotropic_envelope()
    equivariant_layers = make_gaussian_net_layers(nspins=nspins,
                                                  natoms=natoms,
                                                  nelectrons=nelectrons,
                                                  hidden_dims=hidden_dims,
                                                  embedding_dim_eff_angular=embedding_dim_eff_angular,
                                                  embedding_dim_poly_exponent=embedding_dim_poly_exponent,
                                                  feature_layer=feature_layer)

    angular_init, angular_apply = make_embedding_angular_layers(nspins=nspins,
                                                                charges=charges,
                                                                embedding_dim_eff_angular=embedding_dim_eff_angular,
                                                                equivariant_layers=equivariant_layers)
    exp_init, exp_apply = make_exp_layers(embedding_dim_poly_exponent=embedding_dim_poly_exponent)

    orbitals_init, orbitals_apply = make_orbitals(nspins=nspins,
                                                  charges=charges,
                                                  parallel_indices=parallel_indices,
                                                  antiparallel_indices=antiparallel_indices,
                                                  n_parallel=n_parallel,
                                                  n_antiparallel=n_antiparallel,
                                                  angular_init=angular_init,
                                                  angular_apply=angular_apply,
                                                  exp_init=exp_init,
                                                  exp_apply=exp_apply,
                                                  envelope=envelope,
                                                  equivariant_layers=equivariant_layers)

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(key)

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray,) -> Tuple[jnp.ndarray, jnp.ndarray]:
        orbitals_with_angular = orbitals_apply(params, pos, spins, atoms, charges)
        result = network_blocks.logdet_matmul(orbitals_with_angular)
        return result

    return Network(init=init, apply=apply, orbitals=orbitals_apply)


'''
#from GaussianNet.main_train.train import init_electrons
from GaussianNet.tools.utils import system
from GaussianNet.wavefunction import spin_indices
import numpy as np
from absl import logging
from GaussianNet.tools.utils import utils

def walkers_update(f: GaussianNetLike,
                   params: ParamTree,
                   data: GaussianNetData,
                   key: chex.PRNGKey,
                   tstep: float,
                   ndim: int,
                   nelectrons: int,
                   batch_size: int, #this batch_size should be the number of walkers on each GPU
                   i=0):
    """params: batch_params.
    Something is wrong here. Probably it is due to the delay update of walkers. 8.4.2025. """
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    grad_f = jax.grad(logabs_f, argnums=1)

    def grad_f_closure(x):
        return grad_f(params, x, data.spins, data.atoms, data.charges)

    primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

    grad_phase = jax.grad(phase_f, argnums=1)

    def grad_phase_closure(x):
        return grad_phase(params, x, data.spins, data.atoms, data.charges)

    phase_primal, dgrad_phase = jax.linearize(
        grad_phase_closure, data.positions)

    O_old = primal + 1.j * phase_primal
    O_old = jnp.reshape(O_old, (nelectrons, ndim))
    x1 = data.positions
    x1 = jnp.reshape(x1, (nelectrons, ndim))
    x_new = jnp.zeros_like(x1)
    jax.debug.print("O_old:{}", O_old)
    jax.debug.print("x1:{}", x1)
    for i in range(len(x1)):
        key_inner, key_new_inner = jax.random.split(key)
        gauss = jnp.sqrt(tstep) * jax.random.normal(key=key_new_inner, shape=(jnp.shape(x1[i])))
        O_eff = O_old[i]
        jax.debug.print("O_eff:{}", O_eff)
        temp = O_eff + gauss + x1[i]
        jax.debug.print("temp:{}", temp)
        x2 = x1.at[i].set(temp)
        x_2_temp = jnp.reshape(x2, (-1))
        x_1_temp = jnp.reshape(x1, (-1))
        wave_x1_mag = logabs_f(params, x_1_temp, data.spins, data.atoms, data.charges)
        wave_x2_mag = logabs_f(params, x_2_temp, data.spins, data.atoms, data.charges)
        wave_x1_phase = phase_f(params, x_1_temp, data.spins, data.atoms, data.charges)
        wave_x2_phase = phase_f(params, x_2_temp, data.spins, data.atoms, data.charges)
        ratio = ((wave_x2_mag + 1.j * wave_x2_phase) / (wave_x1_mag + 1.j * wave_x1_phase)).real ** 2
        forward = jnp.sum(gauss ** 2)
        primal_x2, dgrad_f_x2 = jax.linearize(grad_f_closure, x_2_temp)
        phase_primal_x2, dgrad_phase_x2 = jax.linearize(grad_phase_closure, x_2_temp)

        O_new = primal_x2 + 1.j * phase_primal_x2
        O_new = jnp.reshape(O_new, (nelectrons, ndim))
        O_new_eff = O_new[i]
        backward = jnp.sum((gauss + O_eff + O_new_eff) ** 2)
        t_pro = jnp.exp(1 / (2 * tstep) * (forward - backward))
        ratio_total = jnp.abs(ratio) * t_pro
        #ratio_total = ratio_total * jnp.sign(ratio)
        rnd = jax.random.uniform(key, shape=ratio_total.shape, minval=0, maxval=1.0)
        cond = ratio_total > rnd
        jax.debug.print("cond:{}", cond)
        x_new = x_new.at[i].set(jnp.where(cond, x2[i], x1[i]))

    x_new = jnp.reshape(x_new, (-1))
    jax.debug.print("x_new:{}", x_new)
    data = GaussianNetData(**(dict(data) | {'positions': x_new}))
    new_key, new_key2 = jax.random.split(key)
    return data, new_key2

def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
    core_electrons: Mapping[str, int] = {},
    max_iter: int = 10000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    niter = 0
    total_electrons = sum(atom.charge - core_electrons.get(atom.symbol, 0)
                          for atom in molecule)
    if total_electrons != sum(electrons):
        if len(molecule) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:
        atomic_spin_configs = [
            (atom.element.nalpha - core_electrons.get(atom.symbol, 0) // 2,
             atom.element.nbeta - core_electrons.get(atom.symbol, 0) // 2)
            for atom in molecule
        ]
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while (
                tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons
                and niter < max_iter
        ):
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            atomic_spin_configs[i] = nbeta, nalpha
            niter += 1

    if tuple(sum(x) for x in zip(*atomic_spin_configs)) == electrons:
        # Assign each electron to an atom initially.
        electron_positions = []
        for i in range(2):
            for j in range(len(molecule)):
                atom_position = jnp.asarray(molecule[j].coords)
                electron_positions.append(
                    jnp.tile(atom_position, atomic_spin_configs[j][i]))
        electron_positions = jnp.concatenate(electron_positions)
    else:
        logging.warning(
            'Failed to find a valid initial electron configuration after %i'
            ' iterations. Initializing all electrons from a Gaussian distribution'
            ' centred on the origin. This might require increasing the number of'
            ' iterations used for pretraining and MCMC burn-in. Consider'
            ' implementing a custom initialisation.',
            niter,
        )
        electron_positions = jnp.zeros(shape=(3 * sum(electrons),))

    # Create a batch of configurations with a Gaussian distribution about each atom.
    key, subkey = jax.random.split(key)
    electron_positions += (
            jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
            * init_width
    )

    electron_spins = _assign_spin_configuration(
        electrons[0], electrons[1], batch_size
    )

    return electron_positions, electron_spins

seed = 23
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
subkey = jax.random.fold_in(subkey, jax.process_index())
pos, spins = init_electrons(
    subkey,
    [system.Atom('C', (0, 0, 0))],
    (3, 3),
    batch_size=1,
    init_width=0.01,
    core_electrons={},
)

atoms = jnp.array([[0.0, 0.0, 0.0]])
pos = pos[0]
spins = spins[0]
jax.debug.print("spins:{}", spins)
charges = jnp.array([0.0])
spins_test = jnp.array([[1., 1.,  1, - 1., - 1., -1]])
parallel_indices, antiparallel_indices, n_parallel, n_antiparallel =\
    spin_indices.jastrow_indices_ee(spins=spins_test,
                                    nelectrons=6)
jax.debug.print("n_parallel:{}", n_parallel)
jax.debug.print("n_antiparallel:{}", n_antiparallel)
network = make_gaussian_net(nspins=(3, 3),
                            charges=charges,
                            parallel_indices=parallel_indices,
                            antiparallel_indices=antiparallel_indices,
                            n_parallel=n_parallel,
                            n_antiparallel=n_antiparallel,)

params = network.init(subkey)
wavefunction_value = network.apply(params, pos, spins, atoms, charges)
jax.debug.print("wavefunction_value:{}", wavefunction_value)
signed_network = network.apply

data = GaussianNetData(
            positions=pos, spins=spins, atoms=atoms, charges=charges
        )
new_data, key = walkers_update(f=signed_network,
                               params=params,
                               data=data,
                               key=subkey,
                               tstep=0.01,
                               ndim=3,
                               nelectrons=6,
                               batch_size=1)
'''