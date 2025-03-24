import enum
import functools
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
import chex
from AIQMCrelease3.wavefunction_Ynlm import network_blocks
from AIQMCrelease3.wavefunction_Ynlm import Jastrow
from AIQMCrelease3.wavefunction_Ynlm import envelope
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
    def __call__(self, key: chex.PRNGKey) -> Tuple[int, int, ParamTree]:
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

class LogAINetLike(Protocol):

    def __call__(
            self,
            params: ParamTree,
            electrons: jnp.ndarray,
            spins: jnp.ndarray,
            atoms: jnp.ndarray,
            charges: jnp.ndarray,
    ) -> jnp.ndarray:
        """Returns the log magnitude of the wavefunction."""

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
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


def make_ainet_features(natoms: int, ndim: int = 3, rescale_inputs: bool = False) -> FeatureLayer:

    def init() -> Tuple[Tuple[int, int], Param]:
        """ ndim + 1 means the dimension is 3 and the normalization of the cartesian coordinates, i.e.,r_ae."""
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
    h_ones = jnp.split(h_one, spin_partitions, axis=0)
    h_twos = jnp.split(h_two, spin_partitions, axis=0)
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]
    features = [h_one] + g_one + g_two
    return jnp.concatenate(features, axis=1)


def y_l_real(x: jnp.array):
    """
    :param x: y/r corresponds to Y_(1, -1)
    :param y: z/r corresponds to Y_(1, 0)
    :param z: x/r corresponds to Y_(1, 1)
    :return:
    """
    #jax.debug.print("x:{}", x)
    return jnp.array([1/2 * jnp.sqrt(1/jnp.pi),
                      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x[0],
                      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x[1],
                      jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x[2]])

def y_l_real_high(x: jnp.array, y: jnp.array):
    """
    to be continued... 19.2.2025
    :param x: ae the cartesian  coordinate
            y: r_ae
    :return: d and f orbitals
    """
    return jnp.array([1/2 * jnp.sqrt(15/jnp.pi) * (x[0] * x[1] / y**2),
                      1/2 * jnp.sqrt(15/jnp.pi) * (x[1] * x[2] / y**2),
                      1/4 * jnp.sqrt(5/jnp.pi) * ((3 * x[2]**2 - y**2) / y**2),
                      1/2 * jnp.sqrt(15/jnp.pi) * (x[0] * x[2] / y**2),
                      1/4 * jnp.sqrt(15/jnp.pi) * ((x[0]**2 - x[1]**2) / y**2),
                      1/4 * jnp.sqrt(35/(2 * jnp.pi) * ((x[1] * (3 * x[0]**2 - x[1]**2))/y**3)),
                      1/2 * jnp.sqrt(105/jnp.pi) * (x[0] * x[1] * x[2] / y**3),
                      1/4 * jnp.sqrt(21/(2 * jnp.pi)) * ((x[1] * (5 * x[2]**2 - y**2))/y**3),
                      1/4 * jnp.sqrt(7/jnp.pi) * ((5 * x[2]**3 - 3*x[2]*y**2)/y**3),
                      1/4 * jnp.sqrt(21/(2 * jnp.pi)) * ((x[0] * (5 * x[2]**2 - y**2))/y**3),
                      1/4 * jnp.sqrt(105/jnp.pi) * (((x[0]**2 - x[1]**2) * x[3])/y**3),
                      1/4 * jnp.sqrt(35/(2 * jnp.pi)) * ((x[0] * (x[0]**2 - 3*x[1]**2))/y**3)])


def make_ai_net_layers(nspins: Tuple[int, int],
                       nelectrons: int,
                       natoms: int,
                       hidden_dims,
                       hidden_dims_Ynlm,
                       feature_layer) -> Tuple[InitLayersAI, ApplyLayersAI]:

    def init(key: chex.PRNGKey) -> Tuple[int, int, ParamTree]:
        params = {}
        key, subkey = jax.random.split(key, num=2)
        (num_one_features, num_two_features), params['input'] = feature_layer.init()
        nchannels = len([nspin for nspin in nspins if nspin > 0])

        def nfeatures(out1, out2):
            return (nchannels + 1) * out1 + nchannels * out2

        dims_one_in = num_one_features
        dims_two_in = num_two_features
        key1, subkey1 = jax.random.split(key)
        layers = []
        layers_y = []
        dims_y_in = 4 * natoms  # 4 is the number of l orbitals, 2 is the number of atoms. s, p(3) = 4, r=(1, 2, 3)
        for i in range(len(hidden_dims)):
            layer_params = {}
            layer_params_y = {}
            key, single_key, single_y_key, *double_keys = jax.random.split(key1, num=5)
            dims_two_embedding = dims_two_in
            dims_one_in = nfeatures(dims_one_in, dims_two_embedding)
            #jax.debug.print("dims_one_in:{}", dims_one_in)
            dims_one_out, dims_two_out = hidden_dims[i] #((4,4), (4,4), (4,4))
            dims_y_out = hidden_dims_Ynlm[i] #((4,2), (4,2), (4,2))

            """someting is wrong about the dimension. solve it later. 18.2.2025."""
            layer_params['single'] = network_blocks.init_linear_layer(single_key,
                                                                      in_dim=dims_one_in,
                                                                      out_dim=dims_one_out,
                                                                      include_bias=True)
            layer_params_y['single_Ynlm'] = network_blocks.init_linear_layer(single_y_key,
                                                                             in_dim=dims_y_in,
                                                                             out_dim=dims_y_out,
                                                                             include_bias=True)

            if i < len(hidden_dims)-1:
                ndouble_channels = 1
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
                layer_params['double'] = layer_params['double'][0]
            
            layers.append(layer_params)
            layers_y.append(layer_params_y)
            dims_one_in = dims_one_out
            dims_two_in = dims_two_out
            dims_y_in = dims_y_out

        params['streams'] = layers
        params['streams_y'] = layers_y
        output_dims = dims_one_in
        output_dims_y = dims_y_in
        jax.debug.print("params:{}", params)
        return output_dims, output_dims_y, params

    def apply_layer(params: Mapping[str, ParamTree],
                    h_one: jnp.array,
                    h_two: Tuple[jnp.array, ...],
                    ) -> Tuple[jnp.array, Tuple[jnp.array, ...]]:
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
        h_two_embedding = h_two[0]
        h_one_in = construct_symmetric_features(h_one, h_two_embedding, nspins)
        h_one_next = jnp.tanh(network_blocks.linear_layer(h_one_in, **params['single']))
        h_one = residual(h_one, h_one_next)

        if 'double' in params:
            params_double = [params['double']]
            h_two_next = [jnp.tanh(network_blocks.linear_layer(prev, w=param['w'], b=param['b']))
                          for prev, param in zip(h_two, params_double)]
            h_two = tuple(residual(prev, new) for prev, new in zip(h_two, h_two_next))
        
        return h_one, h_two

    def apply_layer_y(params: Mapping[str, ParamTree],
                      y_one: jnp.array):
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
        y_one_next = jnp.tanh(network_blocks.linear_layer(y_one, **params['single_Ynlm']))
        y_one = residual(y_one, y_one_next)
        """to be contiuned... 19.2.2025."""
        return y_one

    def apply(params,
              ae: jnp.array,
              r_ae: jnp.array,
              ee: jnp.array,
              r_ee: jnp.array,):
        ae_features, ee_features = feature_layer.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        temp = ae / r_ae
        y_lm_s_p = jax.vmap(jax.vmap(y_l_real, in_axes=0), in_axes=0)(temp)
        y_lm_s_p = jnp.reshape(y_lm_s_p, (8, -1))
        y_one = y_lm_s_p
        for i in range(len(hidden_dims_Ynlm)):
            y_one = apply_layer_y(params['streams_y'][i], y_one)

        h_one = ae_features
        h_two = [ee_features]

        for i in range(len(hidden_dims)):
            #jax.debug.print("params['streams'][i]['single']:{}", params['streams'][i]['single'])
            h_one, h_two = apply_layer(params['streams'][i],
                                       h_one,
                                       h_two)

        h_to_orbitals = h_one
        y_to_orbitals = y_one
        return h_to_orbitals, y_to_orbitals
    return init, apply


def make_orbitals(nspins: Tuple[int, int],
                  nelectrons: int,
                  parallel_indices: jnp.array,
                  antiparallel_indices: jnp.array,
                  n_parallel: int,
                  n_antiparallel: int,
                  charges: jnp.array,
                  equivariant_layers: Tuple[InitLayersAI, ApplyLayersAI]) -> ...:
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    jastrow_ee_init, jastrow_ee_apply = Jastrow.get_jastrow()
    envelope_init, envelope_apply = envelope.make_isotropic_envelope()

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key)
        params = {}
        dims_orbital_in, dims_y_in, params['layers'] = equivariant_layers_init(subkey)
        active_spin_channels = [spin for spin in nspins if spin > 0]
        nchannels = len(active_spin_channels)
        nspin_orbitals = []
        for nspin in active_spin_channels:
            """* 2 means real part and imaginary part."""
            norbitals = sum(nspins) * 2
            nspin_orbitals.append(norbitals)
        #jax.debug.print("nspin_orbitals:{}", nspin_orbitals)
        natom = charges.shape[0]
        orbitals = []
        y_coefficients = []
        for nspin_orbital in nspin_orbitals:
            key, subkey = jax.random.split(key)
            orbitals.append(
                network_blocks.init_linear_layer(
                    subkey,
                    in_dim=dims_orbital_in,
                    out_dim=nspin_orbital,
                    include_bias=True,
                ))
        y_coefficients.append(network_blocks.init_linear_layer(
            subkey,
            in_dim=dims_y_in,
            out_dim=nelectrons,
            include_bias=False))
        #jax.debug.print("dims_orbital_in:{}", dims_orbital_in)
        params['orbitals'] = orbitals
        params['y'] = y_coefficients
        params['jastrow_ee'] = jastrow_ee_init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)

        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
        params['envelope'] = envelope_init(natom=natom, output_dims=output_dims, ndim=3)
        return params

    def apply(params,
              pos: jnp.array,
              spins: jnp.array,
              atoms: jnp.array,
              charges: jnp.array) -> Sequence[jnp.array]:
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        #jax.debug.print("r_ee:{}", r_ee)
        h_to_orbitals, y_to_orbitals = equivariant_layers_apply(params['layers'],
                                                                ae=ae,
                                                                r_ae=r_ae,
                                                                ee=ee,
                                                                r_ee=r_ee,)
        jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        active_spin_channels = [spin for spin in nspins if spin > 0]
        active_spin_partitions = network_blocks.array_partitions(active_spin_channels)

        h_to_orbitals = jnp.split(h_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
        h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
        orbitals = [network_blocks.linear_layer(h, **p) for h, p in zip(h_to_orbitals, params['orbitals'])]
        linear_y_coes = params['y'][0]['w']
        norm = jnp.linalg.norm(linear_y_coes, axis=-1, keepdims=True)
        linear_y_coes = linear_y_coes / norm
        y_orbitals = network_blocks.linear_layer(y_to_orbitals, linear_y_coes)
        orbitals = [orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals]
        """i just used the simplest envelope function. leave this problem to students. 
        I need focus on 2D ewald summation, i.e.PBC, 27.2.2025."""
        ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
        r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
        r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)

        for i in range(len(active_spin_channels)):
            orbitals[i] = orbitals[i] * envelope_apply(
                ae=ae_channels[i],
                r_ae=r_ae_channels[i],
                r_ee=r_ee_channels[i],
                **params['envelope'][i],
            )

        """we need check the following reshape. 21.2.2025."""
        shapes = [(spin, -1, sum(nspins)) for spin in active_spin_channels]
        orbitals = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)]
        orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
        orbitals = jnp.concatenate(orbitals, axis=1)
        orbitals = jnp.reshape(orbitals, (nelectrons, nelectrons))
        jax.debug.print("orbitals:{}", orbitals)
        jax.debug.print("y_orbitals:{}", y_orbitals)
        total_orbitals = orbitals * y_orbitals #element-wise
        jax.debug.print("total_orbitals:{}", total_orbitals)
        """something in Jastrow is wrong here 26.2.2025."""
        """The jastrow has some problems."""
        #jastrow = jnp.exp(jastrow_ee_apply(ee=ee,
        #                                   parallel_indices=parallel_indices,
        #                                   antiparallel_indices=antiparallel_indices,
        #                                   params=params['jastrow_ee']) / nelectrons)
        """to be continued... 21.2.2025."""
        #total_orbitals_jastrow = [total_orbitals * jastrow]
        total_orbitals_jastrow = [total_orbitals] # if 多行列式, 4 * 8 * 8
        return total_orbitals_jastrow

    return init, apply


def make_ai_net(nspins: Tuple[int, int],
                charges: jnp.array,
                parallel_indices: jnp.array,
                antiparallel_indices: jnp.array,
                n_parallel: int,
                n_antiparallel: int,
                ndim: int,
                natoms: int,
                nelectrons: int,
                determinants: int = 1,
                bias_orbitals: bool = True,
                rescale_inputs: bool = False,
                hidden_dims: AILayers = ((4, 4), (4, 4), (4, 4)),
                hidden_dims_Ynlm: AIYnlmLayers = ((6), (6), (6))):

    feature_layer = make_ainet_features(natoms, ndim=ndim, rescale_inputs=rescale_inputs)

    equivariant_layers = make_ai_net_layers(nspins, nelectrons, natoms, hidden_dims, hidden_dims_Ynlm, feature_layer)
    orbitals_init, orbitals_apply = make_orbitals(nspins=nspins, nelectrons=nelectrons,
                                                  charges=charges,
                                                  parallel_indices=parallel_indices,
                                                  antiparallel_indices=antiparallel_indices,
                                                  n_parallel=n_parallel,
                                                  n_antiparallel=n_antiparallel,
                                                  equivariant_layers=equivariant_layers)

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(subkey)

    def apply(params,
              pos: jnp.array,
              spins: jnp.array,
              atoms: jnp.array,
              charges: jnp.array,) -> Tuple[jnp.array, jnp.array]:
        orbitals = orbitals_apply(params, pos, spins, atoms, charges)
        return network_blocks.logdet_matmul(orbitals)

    return Network(init=init, apply=apply, orbitals=orbitals_apply)



from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.spin_indices import jastrow_indices_ee
atoms = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
charges = jnp.array([4.0, 4.0])
spins = jnp.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])

natoms = 2
ndim = 3
nelectrons = 8
nspins = (4, 4)
parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins, nelectrons=8)
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
network = make_ai_net(ndim=ndim,
                      nelectrons=nelectrons,
                      natoms=natoms,
                      nspins=nspins,
                      determinants=1,
                      charges=charges,
                      parallel_indices=parallel_indices,
                      antiparallel_indices=antiparallel_indices,
                      n_parallel=n_parallel,
                      n_antiparallel=n_antiparallel)

params = network.init(subkey)
pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                            electrons=spins,
                            batch_size=1, init_width=0.5)

pos = jnp.reshape(pos, (-1)) # 10 * 3 = 30
jax.debug.print("spin:{}", spins)
jax.debug.print("pos:{}", pos)
#ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
wavefunction_value = network.apply(params, pos, spins, atoms, charges)
jax.debug.print("wavefunction_value:{}", wavefunction_value)
