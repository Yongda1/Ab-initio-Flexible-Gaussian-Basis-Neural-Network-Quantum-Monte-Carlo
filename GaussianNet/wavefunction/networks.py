import enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
import chex
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
from GaussianNet.wavefunction import network_blocks
from GaussianNet.wavefunction import JastrowPade

GaussianLayers = Tuple[Tuple[int, int], ...]
AngularLayers = Tuple[Tuple[int], ...]
PolyexponentLayers = Tuple[Tuple[int], ...]
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]


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
    theta = jnp.arctan2(y, x)
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
    '''
    x = ae[:, :, 0]
    y = ae[:, :, 1]
    z = ae[:, :, 2]
    r, phi, theta = cartesian_to_spherical(x, y, z)
    #jax.debug.print("phi:{}", phi)
    #jax.debug.print("theta:{}", theta)
    total_angular_different_atoms = []
    for i in range(len(theta)):
        angular_s = jax.scipy.special.sph_harm(m=jnp.array([0]), n=jnp.array([0]), theta=theta[i], phi=phi[i])
        angular_p = jax.scipy.special.sph_harm(m=jnp.array([-1, 0, 1]), n=jnp.array([1]), theta=theta[i], phi=phi[i])
        angular_d = jax.scipy.special.sph_harm(m=jnp.array([-2, -1, 0, 1, 2]), n=jnp.array([2]), theta=theta[i], phi=phi[i])
        angular_f = jax.scipy.special.sph_harm(m=jnp.array([-3, -2, -1, 0, 1, 2, 3]), n=jnp.array([3]), theta=theta[i], phi=phi[i])
        angular_total = jnp.concatenate([angular_s, angular_p, angular_d, angular_f])
        #jax.debug.print("angular_total:{}", angular_total)
        total_angular_different_atoms.append(angular_total)
    #jax.debug.print("total_angular_different_atoms:{}", total_angular_different_atoms)
    total_angular_different_atoms = jnp.array(total_angular_different_atoms)
    '''
    return ae, ee, r_ae, r_ee[..., None]


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
        #jax.debug.print("dims_orbital_in:{}", dims_orbital_in)
        params['angular_reformat'] = network_blocks.init_linear_layer(subkey2,
                                                                      in_dim=dims_orbital_in,
                                                                      out_dim=3,#this number is the dimension of the hidden layers.
                                                                      include_bias=False)
        #jax.debug.print("params_angular:{}", params['angular_reformat'])
        layers_angular = []
        dims_one_in = 3
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
        h_to_orbitals = equivariant_layers_apply(params['layers'],
                                                 ae=ae,
                                                 r_ae=r_ae,
                                                 ee=ee,
                                                 r_ee=r_ee,
                                                 charges=charges)
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        """here, something is wrong. """
        #r_shift = network_blocks.linear_layer(h_to_orbitals, **params['angular_reformat'])
        #r_shift = [jnp.dot(h, p) for h, p in zip(h_to_orbitals, params['angular_reformat']['w'])]
        r_shift = network_blocks.linear_layer(h_to_orbitals, w=params['angular_reformat']['w'])
        #jax.debug.print("r_shift:{}", r_shift)
        r_shift = jnp.array(r_shift)

        for i in range(len(embedding_dim_eff_angular)):
            r_shift = apply_layer(params['angular_layer'][i],
                                  r_shift)
            #jax.debug.print("r_shift:{}", r_shift)
        """to be continued... 9.5.2025"""
        #jax.debug.print("r_shift_out:{}", r_shift)
        #jax.debug.print("ae:{}", ae)
        ae = ae + jnp.reshape(r_shift, (6, 1, 3)) #we need do more to generalize it.
        #jax.debug.print("after_ae:{}", ae)
        x = ae[:, :, 0]
        y = ae[:, :, 1]
        z = ae[:, :, 2]
        r, phi, theta = cartesian_to_spherical(x, y, z)
        total_angular_different_atoms = []
        for i in range(len(theta)):
            angular_s = jax.scipy.special.sph_harm(m=jnp.array([0]),
                                                   n=jnp.array([0]),
                                                   theta=theta[i],
                                                   phi=phi[i])
            angular_p = jax.scipy.special.sph_harm(m=jnp.array([-1, 0, 1]),
                                                   n=jnp.array([1]),
                                                   theta=theta[i],
                                                   phi=phi[i])
            angular_d = jax.scipy.special.sph_harm(m=jnp.array([-2, -1, 0, 1, 2]),
                                                   n=jnp.array([2]),
                                                   theta=theta[i],
                                                   phi=phi[i])
            angular_f = jax.scipy.special.sph_harm(m=jnp.array([-3, -2, -1, 0, 1, 2, 3]),
                                                   n=jnp.array([3]),
                                                   theta=theta[i],
                                                   phi=phi[i])
            angular_total = jnp.concatenate([angular_s, angular_p, angular_d, angular_f])
            total_angular_different_atoms.append(angular_total)
        #jax.debug.print("total_angular_different_atoms:{}", total_angular_different_atoms)
        total_angular_different_atoms = jnp.sum(jnp.array(total_angular_different_atoms), axis=-1, keepdims=True)
        #jax.debug.print("total_angular_different_atoms:{}", total_angular_different_atoms)
        return total_angular_different_atoms

    return init, apply


def make_orbitals(nspins: Tuple[int, int],
                  charges: jnp.ndarray,
                  parallel_indices: jnp.array,
                  antiparallel_indices: jnp.array,
                  n_parallel: int,
                  n_antiparallel: int,
                  angular_init,
                  angular_apply,
                  equivariant_layers: Tuple[InitLayersGn, ApplyLayersGn],):
    """to be continued...11.5.2025."""
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    """the jastrow part needs to be done later.11.5.2025."""
    jastrow_ee_init, jastrow_ee_apply, jastrow_ae_init, jastrow_ae_apply = JastrowPade.get_jastrow(charges)


    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey, subsubkey = jax.random.split(key, num=3)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)
        params['angular'] = angular_init(subsubkey)
        active_spin_channels = [spin for spin in nspins if spin > 0]
        jax.debug.print("active_spin_channels:{}", active_spin_channels)
        #jax.debug.print("params:{}", params)
        nchannels = len(active_spin_channels)
        nspin_orbitals = []
        for nspin in active_spin_channels:
            norbitals = sum(nspins) * 1 * 2
            nspin_orbitals.append(norbitals)

        jax.debug.print("nspin_orbitals:{}", nspin_orbitals)

        orbitals = []
        for nspin_orbital in nspin_orbitals:
            key, subkey = jax.random.split(key)
            orbitals.append(
                network_blocks.init_linear_layer(
                    subkey,
                    in_dim=dims_orbital_in,
                    out_dim=nspin_orbital,
                    include_bias=True
                )
            )
        params['orbital'] = orbitals
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
        h_to_orbitals = jnp.split(h_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
        h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
        angular_value = angular_apply(params['angular'], pos, spins, atoms, charges)
        orbitals = [
            network_blocks.linear_layer(h, **p)
            for h, p in zip(h_to_orbitals, params['orbital'])
        ]
        orbitals = [
            orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals
        ]
        #jax.debug.print("orbitals:{}", orbitals)
        #jax.debug.print("angular_value:{}", angular_value)
        angular_value = jnp.reshape(angular_value, (2, 3, 1))
        #jax.debug.print("angular_value:{}", angular_value)
        #jax.debug.print("orbitals:{}", orbitals)
        orbitals_angular = [orbital * angular for orbital, angular in zip(orbitals, angular_value)]
        #jax.debug.print("orbitals_angular:{}", orbitals_angular)

        active_spin_channels = [spin for spin in nspins if spin > 0]
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
        jax.debug.print("jastrow:{}", jastrow)
        #jax.debug.print("orbitals_angular:{}", orbitals_angular)
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
        hidden_dims: GaussianLayers = ((32, 16), (32, 16), (32, 16)),
        embedding_dim_eff_angular: AngularLayers = ((16), (16), (3)),# 6 is the number of electrons correpsonding to rae
        embedding_dim_poly_exponent: PolyexponentLayers = ((16), (16), (16))):
    """The main function to create the many-body wave-function."""
    feature_layer = make_gaussian_features(natoms=natoms, ndim=ndim)
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

    orbitals_init, orbitals_apply = make_orbitals(nspins=nspins,
                                                  charges=charges,
                                                  parallel_indices=parallel_indices,
                                                  antiparallel_indices=antiparallel_indices,
                                                  n_parallel=n_parallel,
                                                  n_antiparallel=n_antiparallel,
                                                  angular_init=angular_init,
                                                  angular_apply=angular_apply,equivariant_layers=equivariant_layers)
    #params = orbitals_init(key=key)
    #output = orbitals_apply(params=params, pos=pos, spins=spins, atoms=atoms, charges=charges)

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


from AIQMC.main_train.train import init_electrons
from AIQMC.tools.utils import system
from AIQMC.wavefunction_new import spin_indices

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
parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = \
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