"""here, we take a test about the one atom calculation, Because the electron must have SO(3) symmetry.
We copy some setup from wavefunction_Ynlm. 13.4.2025."""
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

from jax import random
import numpy as np
import emlp.nn.flax as nn
from emlp.reps import T, V
from emlp.groups import SO

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
                 charges: jnp.array):
        """we should add the output type. 6.4.2025."""


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
    # jax.debug.print("x:{}", x)
    return jnp.array([1 / 2 * jnp.sqrt(1 / jnp.pi),
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
    # jax.debug.print("x:{}", x)
    # jax.debug.print("y:{}", y)
    # jax.debug.print("nan:{}", 1/4 * jnp.sqrt(35/(2 * jnp.pi)) * ((x[1] * (3 * x[0]**2 - x[1]**2))/y**3))
    # jax.debug.print("nan1:{}", x[1] * (3 * x[0]**2 - x[1]**2))
    # jax.debug.print("nan2:{}", y**3)
    # jax.debug.print("nan3:{}", (x[1] * (3 * x[0]**2 - x[1]**2))/y**3)
    return jnp.array([1 / 2 * jnp.sqrt(15 / jnp.pi) * (x[0] * x[1] / y ** 2),
                      1 / 2 * jnp.sqrt(15 / jnp.pi) * (x[1] * x[2] / y ** 2),
                      1 / 4 * jnp.sqrt(5 / jnp.pi) * ((3 * x[2] ** 2 - y ** 2) / y ** 2),
                      1 / 2 * jnp.sqrt(15 / jnp.pi) * (x[0] * x[2] / y ** 2),
                      1 / 4 * jnp.sqrt(15 / jnp.pi) * ((x[0] ** 2 - x[1] ** 2) / y ** 2),
                      1 / 4 * jnp.sqrt(35 / (2 * jnp.pi)) * ((x[1] * (3 * x[0] ** 2 - x[1] ** 2)) / y ** 3),
                      1 / 2 * jnp.sqrt(105 / jnp.pi) * (x[0] * x[1] * x[2] / y ** 3),
                      1 / 4 * jnp.sqrt(21 / (2 * jnp.pi)) * ((x[1] * (5 * x[2] ** 2 - y ** 2)) / y ** 3),
                      1 / 4 * jnp.sqrt(7 / jnp.pi) * ((5 * x[2] ** 3 - 3 * x[2] * y ** 2) / y ** 3),
                      1 / 4 * jnp.sqrt(21 / (2 * jnp.pi)) * ((x[0] * (5 * x[2] ** 2 - y ** 2)) / y ** 3),
                      1 / 4 * jnp.sqrt(105 / jnp.pi) * (((x[0] ** 2 - x[1] ** 2) * x[3]) / y ** 3),
                      1 / 4 * jnp.sqrt(35 / (2 * jnp.pi)) * ((x[0] * (x[0] ** 2 - 3 * x[1] ** 2)) / y ** 3)])


def make_ai_net_layers(model,
                       nspins: Tuple[int, int],
                       nelectrons: int,
                       natoms: int,
                       hidden_dims,
                       hidden_dims_Ynlm,
                       feature_layer,
                       ) -> Tuple[InitLayersAI, ApplyLayersAI]:
    def init(key: chex.PRNGKey) -> Tuple[int, int, ParamTree]:
        params = {}
        key, subkey = jax.random.split(key, num=2)
        (num_one_features, num_two_features), params['input'] = feature_layer.init()
        jax.debug.print("num_one_features:{}", num_one_features)
        x = jax.random.normal(key, shape=[24]) # we use the temporary number to [repin(G).size()])
        emlp_params = model.init(subkey, x)
        params['emlp_params'] = emlp_params
        return params

    def apply(params,
              ae: jnp.array,
              r_ae: jnp.array,
              ee: jnp.array,
              r_ee: jnp.array, ):
        ae_features, ee_features = feature_layer.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        h_one = ae_features
        jax.debug.print("h_one:{}", h_one)
        jax.debug.print("params:{}", params)
        h_one = jnp.reshape(h_one, -1)
        y = model.apply(params['emlp_params'], h_one)
        y = jnp.reshape(y, (nelectrons, -1))
        jax.debug.print("y:{}", y)
        return y

    return init, apply


def make_orbitals(model,
                  nspins: Tuple[int, int],
                  nelectrons: int,
                  parallel_indices: jnp.array,
                  antiparallel_indices: jnp.array,
                  spin_up_indices: jnp.array,
                  spin_down_indices: jnp.array,
                  n_parallel: int,
                  n_antiparallel: int,
                  charges: jnp.array,
                  equivariant_layers: Tuple[InitLayersAI, ApplyLayersAI]) -> ...:
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    jastrow_ee_init, jastrow_ee_apply, jastrow_ae_init, jastrow_ae_apply = Jastrow.get_jastrow(charges)
    envelope_init, envelope_apply = envelope.make_pp_like_envelope()

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key)
        params = {}
        params['layers'] = equivariant_layers_init(subkey)
        active_spin_channels = [spin for spin in nspins if spin > 0]
        nchannels = len(active_spin_channels)
        natom = charges.shape[0]
        nspin_orbitals = []
        for nspin in active_spin_channels:
            """* 2 means real part and imaginary part."""
            norbitals = sum(nspins) * 2
            nspin_orbitals.append(norbitals)

        orbitals = []
        for nspin_orbital in nspin_orbitals:
            key, subkey = jax.random.split(key)
            orbitals.append(
                network_blocks.init_linear_layer(
                    subkey,
                    in_dim=9,
                    out_dim=nspin_orbital,
                    include_bias=True,
                ))
        params['orbitals'] = orbitals
        params['jastrow_ee'] = jastrow_ee_init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
        params['jastrow_ae'] = jastrow_ae_init(nelectrons=nelectrons, natoms=natom)
        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
        params['envelope'] = envelope_init(natom=natom, nelectrons=nelectrons, ndim=3)
        return params

    def apply(params,
              pos: jnp.array,
              spins: jnp.array,
              atoms: jnp.array,
              charges: jnp.array) -> Sequence[jnp.array]:
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        h_to_orbitals  = equivariant_layers_apply(params['layers'],
                                                  ae=ae,
                                                  r_ae=r_ae,
                                                  ee=ee,
                                                  r_ee=r_ee, )
        jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        active_spin_channels = [spin for spin in nspins if spin > 0]
        h_up = h_to_orbitals[spin_up_indices]
        h_down = h_to_orbitals[spin_down_indices]
        h_to_orbitals_with_spin = [h_up, h_down]
        orbitals = [network_blocks.linear_layer(h, **p) for h, p in zip(h_to_orbitals_with_spin, params['orbitals'])]
        orbitals = [orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals]
        shapes = [(spin, -1, sum(nspins)) for spin in active_spin_channels]
        orbitals = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)]
        orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
        orbitals = jnp.concatenate(orbitals, axis=1)
        orbitals = jnp.reshape(orbitals, (nelectrons, nelectrons))
        jax.debug.print("orbitals:{}", orbitals)
        orbitals_with_envelope = []
        for i in range(nelectrons):
            test = envelope_apply(orbitals[i], r_ae[i], ae[i], charges, params['envelope'][i])
            orbitals_with_envelope.append(test)

        orbitals_with_envelope = jnp.array(orbitals_with_envelope)
        r_ee = jnp.reshape(r_ee, (nelectrons, nelectrons))
        """by removing the norm calculation in Jastrow, the 'Nan' problem is solved. 
        Actually, I dont understand why. Maybe it is some underlying bug in Jax."""
        jastrow_ee = jnp.exp(jastrow_ee_apply(r_ee=r_ee,
                                              parallel_indices=parallel_indices,
                                              antiparallel_indices=antiparallel_indices,
                                              params=params['jastrow_ee']) / nelectrons)
        # jax.debug.print("jastrow:{}", jastrow)
        # jax.debug.print("r_ae:{}", r_ae)
        """to be continued... 21.2.2025."""
        r_ae = jnp.reshape(r_ae, (nelectrons, -1))
        jastrow_ae = jnp.exp(jastrow_ae_apply(r_ae=r_ae, params=params['jastrow_ae']) / nelectrons)
        # jax.debug.print("jastrow_ae_ee:{}", jastrow_ae * jastrow_ee)
        """we first move to all electrons calculation today 6.4.2025."""
        orbitals_with_jastrow = orbitals_with_envelope * jastrow_ee * jastrow_ae
        total_orbitals_jastrow = [orbitals_with_jastrow]  # if 多行列式, 4 * 8 * 8

        return total_orbitals_jastrow

    return init, apply


def make_ai_net(nspins: Tuple[int, int],
                charges: jnp.array,
                parallel_indices: jnp.array,
                antiparallel_indices: jnp.array,
                spin_up_indices: jnp.array,
                spin_down_indices: jnp.array,
                n_parallel: int,
                n_antiparallel: int,
                ndim: int,
                natoms: int,
                nelectrons: int,
                determinants: int = 1,
                bias_orbitals: bool = True,
                rescale_inputs: bool = False,
                hidden_dims: AILayers = ((4, 4), (4, 4), (4, 4)),
                hidden_dims_Ynlm: AIYnlmLayers = ((6), (6), (6)),):
    feature_layer = make_ainet_features(natoms, ndim=ndim, rescale_inputs=rescale_inputs)
    repin = nelectrons * T(1) + nelectrons * T(0)
    repout = nelectrons * T(2)
    G = SO(3)
    test_output = repin(G).size()
    jax.debug.print("test_output:{}", test_output)
    #model = nn.EMLP(repin, repout, G)
    model = nn.EMLP(repin, repout, group=G, num_layers=3)
    #size_number = repin(G).size()
    #jax.debug.print("size_number:{}", size_number)
    equivariant_layers = make_ai_net_layers(model,
                                            nspins,
                                            nelectrons,
                                            natoms,
                                            hidden_dims,
                                            hidden_dims_Ynlm,
                                            feature_layer,
                                            )
    orbitals_init, orbitals_apply = make_orbitals(model,
                                                  nspins=nspins,
                                                  nelectrons=nelectrons,
                                                  charges=charges,
                                                  parallel_indices=parallel_indices,
                                                  antiparallel_indices=antiparallel_indices,
                                                  spin_up_indices=spin_up_indices,
                                                  spin_down_indices=spin_down_indices,
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
              charges: jnp.array, ) -> Tuple[jnp.array, jnp.array]:
        orbitals = orbitals_apply(params, pos, spins, atoms, charges)
        return network_blocks.logdet_matmul(orbitals)

    return Network(init=init, apply=apply, orbitals=orbitals_apply)


from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.spin_indices import jastrow_indices_ee
from AIQMCrelease3.spin_indices import spin_indices_h

atoms = jnp.array([[0.0, 0.0, 0.0]])
charges = jnp.array([6.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0,])
structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])

natoms = 1
ndim = 3
nelectrons = 6
nspins = (3, 3)
parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins, nelectrons=6)
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
spin_up_indices, spin_down_indices = spin_indices_h(spins)


network = make_ai_net(ndim=ndim,
                      nelectrons=nelectrons,
                      natoms=natoms,
                      nspins=nspins,
                      determinants=1,
                      charges=charges,
                      parallel_indices=parallel_indices,
                      antiparallel_indices=antiparallel_indices,
                      n_parallel=n_parallel,
                      n_antiparallel=n_antiparallel,
                      spin_up_indices=spin_up_indices,
                      spin_down_indices=spin_down_indices)


params = network.init(subkey)
pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                            electrons=spins,
                            batch_size=1, init_width=0.5)
jax.debug.print("params:{}", params)
pos = jnp.reshape(pos, (-1))  # 10 * 3 = 30
# jax.debug.print("spin:{}", spins)
# jax.debug.print("pos:{}", pos)
# ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
wavefunction_value = network.apply( params, pos, spins, atoms, charges)
jax.debug.print("wavefunction_value:{}", wavefunction_value)