import enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
import math
import chex
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
from GaussianNet.wavefunction import network_blocks
from GaussianNet.wavefunction import JastrowPade
from GaussianNet.wavefunction import envelopes
from GaussianNet.wavefunction import generate_g_uu
from GaussianNet.wavefunction import generate_g_ud
from GaussianNet.wavefunction.f_uu import c_uu
from GaussianNet.wavefunction.f_dd import c_dd
from GaussianNet.wavefunction.f_ud import f_s
from GaussianNet.wavefunction.f_ud import f_t

from pfapack import pfaffian as pf
from pfapack.ctypes import pfaffian as cpf
import numpy as np

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
                 charges: jnp.ndarray, ) -> jnp.ndarray:
        """"""


class FeatureInit(Protocol):

    def __call__(self, ) -> Tuple[Tuple[int, int], Param]:
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
                 charges: jnp.ndarray, ) -> jnp.ndarray:
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
                             feature_layer, ):
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
                                                                                   include_bias=True, ))
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
        residual = lambda x, y: (x + y) / jnp.sqrt(
            2.0) if x.shape == y.shape else y  # the shape of x must be same with y
        h_two_embedding = h_two[0]
        h_one_in = construct_symmetric_features(h_one, h_two_embedding, nspins)
        #jax.debug.print("h_one_in:{}", h_one_in)
        #jax.debug.print("params['single']:{}", params['single'])
        h_one_next = jnp.tanh(network_blocks.linear_layer(h_one_in, **params['single']))
        h_one = residual(h_one, h_one_next)
        if 'double' in params:
            params_double = [params['double']]
            h_two_next = [jnp.tanh(network_blocks.linear_layer(prev, **param))
                          for prev, param in zip(h_two, params_double)]
            h_two = tuple(residual(prev, new) for prev, new in zip(h_two, h_two_next))

        return h_one, h_two

    def apply(params,
              ae: jnp.ndarray,
              r_ae: jnp.ndarray,
              ee: jnp.ndarray,
              r_ee: jnp.ndarray,
              charges: jnp.ndarray, ):
        ae_features, ee_features = feature_layer.apply(
            ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
        )
        h_one = ae_features
        h_two = [ee_features]
        #jax.debug.print("h_one:{}", h_one)
        for i in range(len(hidden_dims)):
            h_one, h_two = apply_layer(params['embedding_layer'][i],
                                       h_one,
                                       h_two)
        h_to_orbitals = h_one
        return h_to_orbitals

    return init, apply


def pfaffian(orbitals: jnp.ndarray,):
    """we used this function to construct pfaffian wave function. However, the first step is to construct the pair orbitals.
        first, we need to generate the symmetry neural network c_(i,j)
        we need make a new approach to generate the pfaffian with arbitrary orbitals. 08.09.2025.
        """
    """to include the parameters of coe_uu, coe_dd, we need reconstruct the new class for pfaffian 11.09.2025."""
    jax.debug.print("orbitals:{}", orbitals)
    """first, we need split the determinants into four parts including G_uu, G_dd, G_ud, G_du.10.09.2025."""
    """first we try to solve the G_uu and G_dd part. Suppose that the number of electrons is even.10.09.2025."""
    n_spin_up = 3
    orbitals_uu = orbitals[0][0][0:n_spin_up, 0:n_spin_up]
    orbitals_dd = orbitals[0][0][n_spin_up:, n_spin_up:]
    orbitals_ud = orbitals[0][0][n_spin_up:, 0:n_spin_up]
    orbitals_du = orbitals[0][0][0:n_spin_up:, n_spin_up:]
    #jax.debug.print("orbitals_ud:{}", orbitals_ud)
    #jax.debug.print("orbitals_dd:{}", orbitals_dd)
    f_s, f_t = generate_g_ud.split_matrix( orbitals_determinant_uu=orbitals_uu,
                                               orbitals_determinant_dd=orbitals_dd,
                                               orbitals_determinant_ud=orbitals_ud,
                                               orbitals_determinant_du=orbitals_du)
    #jax.debug.print("f_s:{}", f_s)
    #jax.debug.print("f_t:{}", f_t)
    """then we need generate the coefficients function 12.09.2025."""
    det_value_uu = generate_g_uu.split_matrix(orbitals_determinant=orbitals_uu, n_spin=n_spin_up)
    #jax.debug.print("det_value_uu:{}", det_value_uu)
    det_value_dd = generate_g_uu.split_matrix(orbitals_determinant=orbitals_dd, n_spin=n_spin_up)
    #jax.debug.print("det_value_dd:{}", det_value_dd)
    return det_value_uu, det_value_dd, f_s, f_t


def make_orbitals(nspins: Tuple[int, int],
                  charges: jnp.ndarray,
                  parallel_indices: jnp.array,
                  antiparallel_indices: jnp.array,
                  n_parallel: int,
                  n_antiparallel: int,
                  n_determinants: int,
                  number_of_coefficients: int,
                  envelope,
                  equivariant_layers: Tuple[InitLayersGn, ApplyLayersGn], ):
    """to be continued...11.5.2025."""
    equivariant_layers_init, equivariant_layers_apply = equivariant_layers
    """the jastrow part needs to be done later.11.5.2025."""
    jastrow_ee_init, jastrow_ee_apply, jastrow_ae_init, jastrow_ae_apply = JastrowPade.get_jastrow(charges)
    """we add coe_uu here. 11.09.2025."""


    coe_init, coe_apply = c_uu.coefficients_layer(hidden_dims_coe=(4, 4, 4, number_of_coefficients))
    coe_init_dd, coe_apply_dd = c_dd.coefficients_layer(hidden_dims_coe=(4, 4, 4, number_of_coefficients))
    """we need rename the coefficient function to keep them consistent.12.09.2025."""
    coe_f_s_init, coe_f_s_apply = f_s.coefficients_layer(hidden_dims_coe=(4, 4, 4, 9))
    coe_f_t_init, coe_f_t_apply = f_t.coefficients_layer(hidden_dims_coe=(4, 4, 4, 9))

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey, subsubkey, subsubsubkey = jax.random.split(key, num=4)
        params = {}
        dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)
        active_spin_channels = [spin for spin in nspins if spin > 0]
        nchannels = len(active_spin_channels)
        nspin_orbitals = []
        """here, add more determinants. 25.5.2025."""
        for nspin in active_spin_channels:
            norbitals = sum(nspins) * n_determinants * 2
            nspin_orbitals.append(norbitals)

        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
        params['envelope'] = envelope.init(natom=1, output_dims=output_dims, ndim=3)

        orbitals = []
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

        params['orbital'] = orbitals
        params['jastrow_ee'] = jastrow_ee_init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
        params['jastrow_ae'] = jastrow_ae_init(nelectrons=6, natoms=1)
        """parameters for the coe function.11.09.2025."""
        output_dims, params['coe_uu'] = coe_init(subsubsubkey)
        output_dims, params['coe_dd'] = coe_init_dd(subsubsubkey)
        output_dims1, params['f_s'] = coe_f_s_init(subsubsubkey)
        output_dims2, params['f_t'] = coe_f_t_init(subsubsubkey)
        return params

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray, ):
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        h_to_orbitals = equivariant_layers_apply(params['layers'],
                                                 ae=ae,
                                                 r_ae=r_ae,
                                                 ee=ee,
                                                 r_ee=r_ee,
                                                 charges=charges)
        jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        h_to_orbitals = jnp.split(h_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
        h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        #jax.debug.print("orbital:{}", params['orbital'])
        #for h, p in zip(h_to_orbitals, params['orbital']):
            #jax.debug.print("h:{}", h)
            #jax.debug.print("p:{}", p)
            #value = network_blocks.linear_layer(h, **p)
            #jax.debug.print("value:{}", value)
        orbitals = [
            network_blocks.linear_layer(h, **p)
            for h, p in zip(h_to_orbitals, params['orbital'])
        ]
        jax.debug.print("orbitals:{}", orbitals)

        orbitals = [orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals]
        #jax.debug.print("orbitals_complex:{}", orbitals)

        orbitals_angular = orbitals

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

        shapes = [(spin, -1, sum(nspins)) for spin in active_spin_channels]
        orbitals_angular = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals_angular, shapes)]
        #jax.debug.print("orbitals_angular_before:{}", orbitals_angular)
        orbitals_angular = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals_angular]
        orbitals_angular = [jnp.concatenate(orbitals_angular, axis=1)]
        jax.debug.print("orbitals_angular:{}", orbitals_angular)
        """the determinant is |psi_1(r_1)  psi_2(r_1) psi_3(r_1) psi_4(r_1)|
                              |psi_1(r_2)  psi_2(r_2) psi_3(r_2) psi_4(r_2)|
                              |psi_1(r_3)  psi_2(r_3) psi_3(r_3) psi_4(r_3)|
                              |psi_1(r_4)  psi_2(r_4) psi_3(r_4) psi_4(r_4)|"""
        '''
        """the next step, we need construct the G_uu and G_dd 11.09.2025."""
        uu, dd, ud_s, ud_t = pfaffian(orbitals=jnp.array(orbitals_angular))
        r_ee_uu = jnp.reshape(r_ee, (6, -1))
        jax.debug.print("r_ee:{}", r_ee_uu)
        """what we need here is three r12, r13, r23, i.e., r_ee_uu_1[0], r_ee_uu_1[1]"""
        r_ees_parallel = jnp.array([r_ee_uu[parallel_indices[:, i][0], parallel_indices[:, i][1]] for i in range(6)])
        r_ees_parallel_uu = jnp.reshape(r_ees_parallel[0:3], (-1, 1))
        r_ees_parallel_uu = jnp.repeat(r_ees_parallel_uu, repeats=3, axis=1)

        coe_uu = jax.vmap(coe_apply, in_axes=(None, 0))(params['coe_uu'], r_ees_parallel_uu)
        jax.debug.print("r_ees_parallel:{}", r_ees_parallel)
        r_ees_parallel_dd = jnp.reshape(r_ees_parallel[3:], (-1, 1))
        r_ees_parallel_dd = jnp.repeat(r_ees_parallel_dd, repeats=3, axis=1)
        jax.debug.print("r_ees_parallel_dd:{}", r_ees_parallel_dd)
        coe_dd = jax.vmap(coe_apply_dd, in_axes=(None, 0))(params['coe_dd'], r_ees_parallel_dd)

        value_uu = jnp.sum(coe_uu * uu, axis=1)
        value_dd = jnp.sum(coe_dd * dd, axis=1)
        """so far, we got the up triangle part of matrix elements of G_uu. 11.09.2025.
           The G_dd part can be done in the same way. The problem is G_ud part. 12.09.2025."""
        r_ees_antiparallel = jnp.array([r_ee_uu[antiparallel_indices[:, i][0], antiparallel_indices[:, i][1]] for i in range(9)])
        #jax.debug.print("r_ees_antiparallel:{}", r_ees_antiparallel)
        r_ees_antiparallel = jnp.reshape(r_ees_antiparallel, (-1, 1))
        r_ees_antiparallel_ud = jnp.repeat(r_ees_antiparallel, repeats=9, axis=1)
        #jax.debug.print("r_ees_antiparallel_ud:{}", r_ees_antiparallel_ud)
        f_s_coe = jax.vmap(coe_f_s_apply, in_axes=(None, 0))(params['f_s'], r_ees_antiparallel_ud)
        f_t_coe = jax.vmap(coe_f_t_apply, in_axes=(None, 0))(params['f_t'], r_ees_antiparallel_ud)

        def return_array(ud_s: jnp.array):
            """here, we use a trick of changing the output axis to reshape the array.15.09/2025
            Just because the orbitals data used in the fs is int the wrong order as show in the slides page 16."""
            return ud_s

        return_array_vmap = jax.vmap(jax.vmap(return_array, in_axes=1, out_axes=0), in_axes=0, out_axes=0)
        f_s_orbitals = return_array_vmap(ud_s)
        f_t_orbitals = return_array_vmap(ud_t)
        f_s_coe = jnp.reshape(f_s_coe, f_s_orbitals.shape)
        f_t_coe = jnp.reshape(f_t_coe, f_t_orbitals.shape)
        f_s = jnp.sum(jnp.sum(f_s_orbitals * f_s_coe, axis=-1), axis=-1)
        f_t = jnp.sum(jnp.sum(f_t_orbitals * f_t_coe, axis=-1), axis=-1)
        #jax.debug.print("f_s: {}", f_s.shape)
        #jax.debug.print("f_t: {}", f_t.shape)
        value_f_ud = f_s + f_t
        #jax.debug.print("f_ud:{}", value_f_ud.shape)
        #jax.debug.print("uu:{}", value_uu.shape)
        """the shape of ud_s and ud_t need to be modified further. 12.09.2025."""
        """currently, we dont make the general method to construct pfaffian. Now, we just make it working for 6 electrons."""

        f_uu = jnp.zeros((9), dtype=jnp.complex64)
        f_dd = jnp.zeros((9), dtype=jnp.complex64)
        f_ud = jnp.zeros((3,3), dtype=jnp.complex64)
        #jax.debug.print("value_uu:{}", value_uu)
        #jax.debug.print("f_uu:{}", f_uu)
        f_uu = f_uu.at[1].set(value_uu[0])
        f_uu = f_uu.at[2].set(value_uu[1])
        f_uu = f_uu.at[5].set(value_uu[2])
        #jax.debug.print("f_uu:{}", f_uu)
        value_f_uu = jnp.reshape(f_uu, (3, 3))
        f_dd = f_dd.at[1].set(value_uu[0])
        f_dd = f_dd.at[2].set(value_uu[1])
        f_dd = f_dd.at[5].set(value_uu[2])
        value_f_dd = jnp.reshape(f_dd, (3, 3))
        jax.debug.print("f_dd:{}", value_f_dd)
        #jax.debug.print("f_uu:{}", value_f_uu)
        #jax.debug.print("f_ud:{}", value_f_ud)
        pfaffian_wavefunction_up_down = jnp.concatenate((f_ud, value_f_dd), axis=1)
        pfaffian_wavefunction_up_up = jnp.concatenate((value_f_uu, value_f_ud), axis=1)
        jax.debug.print("pfaffian_wavefunction_up_down:{}", pfaffian_wavefunction_up_down)
        jax.debug.print("pfaffian_wavefunction_up_up:{}", pfaffian_wavefunction_up_up)
        pfaffian_up = jnp.concatenate((pfaffian_wavefunction_up_up, pfaffian_wavefunction_up_down), axis=0)
        jax.debug.print("pfaffian_up:{}", pfaffian_up)
        return pfaffian_up
        '''
        """
        jastrow = jnp.exp(jastrow_ee_apply(r_ee=r_ee,
                                           parallel_indices=parallel_indices,
                                           antiparallel_indices=antiparallel_indices,
                                           params=params['jastrow_ee']) / 6)
        orbitals_angular_jastrow = [orbital * jastrow for orbital in orbitals_angular]"""
        return orbitals_angular


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
        #hidden_dims_coe = (16, 16, 16, 16),
        number_of_coefficients: int = 1,
        ):
    """The main function to create the many-body wave-function."""
    feature_layer = make_gaussian_features(natoms=natoms, ndim=ndim)
    """we only use isotropic function currently."""
    envelope = envelopes.make_isotropic_envelope()
    equivariant_layers = make_gaussian_net_layers(nspins=nspins,
                                                  natoms=natoms,
                                                  nelectrons=nelectrons,
                                                  hidden_dims=hidden_dims,
                                                  feature_layer=feature_layer)

    orbitals_init, orbitals_apply = make_orbitals(nspins=nspins,
                                                  charges=charges,
                                                  parallel_indices=parallel_indices,
                                                  antiparallel_indices=antiparallel_indices,
                                                  n_parallel=n_parallel,
                                                  n_antiparallel=n_antiparallel,
                                                  n_determinants=1,
                                                  number_of_coefficients=number_of_coefficients,
                                                  envelope=envelope,
                                                  equivariant_layers=equivariant_layers)



    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(key)

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray, ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        orbitals_with_angular = orbitals_apply(params, pos, spins, atoms, charges)
        jax.debug.print("orbitals_with_angular:{}", orbitals_with_angular)
        """here, we test the pfaffian function.10.09.2025."""
        #output = pfaffian(orbitals_with_angular)
        result = network_blocks.logdet_matmul(orbitals_with_angular)
        #pfaffian_wavefunction = orbitals_apply(params, pos, spins, atoms, charges)
        #jax.debug.print("pfaffian_wavefunction:{}", pfaffian_wavefunction)
        #jax.debug.print("-pfaffian_wavefunction:{}", -pfaffian_wavefunction)
        #pfaffian_wavefunction =pfaffian_wavefunction - jnp.transpose(pfaffian_wavefunction)
        """this is only working for the real number.15.09.2025. we need find a way to solve this problem.
        the python version is not working well but the cpython version is working. 16.09.2025."""
        #result = pf.pfaffian(pfaffian_wavefunction)
        #jax.debug.print("type_pfaffian_wavefunction:{}", type(pfaffian_wavefunction))
        """here, we have a type warning. It does not matter. 16.09.2025."""
        #result = cpf(matrix= pfaffian_wavefunction, uplo='U')
        """here,we need notice the output of pfaffian is just a complex number. While we calculate the determinant, we are
        using the log to calculate the value of wave function. To match the format, we rewrite the result to [[[result]]].
        Tomorrow, we test it. We also notice the method orbitals_apply. in case somewhere we used it. 16.09.2025.
        It is running now. We need check out if our codes are running smoothly 16.09.2025..
        """
        #result = network_blocks.slogdet(jnp.array([[[result]]]))
        return result

    return Network(init=init, apply=apply, orbitals=orbitals_apply)


from GaussianNet.wavefunction import spin_indices


seed = 23
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
subkey = jax.random.fold_in(subkey, jax.process_index())

atoms = jnp.array([[0.0, 0.0, 0.0]])
#pos = jnp.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
pos = jnp.array([2, 2, 2, 1, 1, 1, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
charges = jnp.array([0.0])
spins_test = jnp.array([[1., 1., 1., - 1., - 1., -1.]])
spins = spins_test
parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = spin_indices.jastrow_indices_ee(spins=spins_test,
                                    nelectrons=6)
#jax.debug.print("parallel_indices:{}", parallel_indices)
#jax.debug.print("antiparallel_indices:{}", antiparallel_indices)
#jax.debug.print("n_parallel:{}", n_parallel)
#jax.debug.print("n_antiparallel:{}", n_antiparallel)

n_spin_up = 3
number_coe = int(math.factorial(n_spin_up) / (math.factorial(2) * math.factorial(n_spin_up - 2)))
#jax.debug.print("number_coe:{}", number_coe)
network = make_gaussian_net(nspins=(3, 3),
                            charges=charges,
                            parallel_indices=parallel_indices,
                            antiparallel_indices=antiparallel_indices,
                            n_parallel=n_parallel,
                            n_antiparallel=n_antiparallel,
                            number_of_coefficients=number_coe,)

params = network.init(subkey)
wavefunction_value = network.apply(params, pos, spins, atoms, charges)
jax.debug.print("wavefunction_value:{}", wavefunction_value)
