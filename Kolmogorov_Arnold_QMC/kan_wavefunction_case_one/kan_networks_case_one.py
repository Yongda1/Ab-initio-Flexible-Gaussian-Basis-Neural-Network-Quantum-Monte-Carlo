#from jaxkan.KAN import KAN
import jax.numpy as jnp
import jax
import chex
import itertools
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import kan_networks_blocks_case_one as kan_networks_blocks
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import chebyshev_blocks as chebyshev_blocks
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import kan_envelopes_case_one_general as kan_envelopes
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import chebyshev_envelopes
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.JastrowPade import make_pade_ee_jastrow
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import simple_envelope
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import normal_network_blocks

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]

@chex.dataclass
class KANetsData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


def construct_input_features(
        pos: jnp.ndarray,
        atoms: jnp.ndarray,
        ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs inputs to Fermi Net from raw electron and atomic positions.
    For KANets, we need do the normalization for the input layer to be [-1, 1].
    currently the optimization is not stable. We consider to reconstruct it as FermiNet did before?
    They used mean value to stabilize the networks."""
    #jax.debug.print("atoms:{}", atoms)
    assert atoms.shape[1] == ndim
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]

def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
  """Returns the indices for splitting an array into separate partitions.

  Args:
    sizes: size of each of N partitions. The dimension of the array along
    the relevant axis is assumed to be sum(sizes).

  Returns:
    sequence of indices (length len(sizes)-1) at which an array should be split
    to give the desired partitions.
  """
  return list(itertools.accumulate(sizes))[:-1]

def construct_symmetric_features(
        h_one: jnp.ndarray,
        h_two: jnp.ndarray,
        nspins: Tuple[int, int],
) -> jnp.ndarray:
    spin_partitions = array_partitions(nspins)
    h_ones = jnp.split(h_one, spin_partitions, axis=0)
    h_twos = jnp.split(h_two, spin_partitions, axis=0)
    g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]
    features = [h_one] + g_one + g_two
    return jnp.concatenate(features, axis=1)


def make_kan_features(natoms: int, ndim: int = 3):
    def init() -> Tuple[Tuple[int, int], Param]:
        return (natoms * (ndim +1), ndim + 1), {}

    def apply(ae, r_ae) -> jnp.ndarray:
        ae_features = jnp.concatenate((r_ae, ae), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features
    return init, apply



def make_kan_net_layers(layer_dims: jnp.ndarray,
                        g: jnp.ndarray,
                        k: jnp.ndarray,
                        chebyshev: bool = False,
                        spline: bool = False,
                        add_residual: bool = False,
                        add_bias: bool = True,
                        external_weights: bool = True,
                        ):
    """

    :param layer_dims: the number of nodes each layer.
    :param g: the number of grid for spline basis functions.
    :param k: the order of the spline or chebyshev basis functions.
    :param chebyshev: turn on chebyshev basis functions or not.
    :return: one vector from equivalent layers.
    """
    def init(key: chex.PRNGKey):
        """here, we initialize the parameters of KANets wave function. 9.10.2025."""
        params = {}
        """we also need initialize the gird for each layer."""
        layers = []
        for i in range(len(layer_dims)-1):
            """because the number of nodes in KANets is controlled by layer_dims=jnp.ndarray([3, 4, 5, 6]). Therefore, the """
            layer_params = {}
            dimension_in = int(layer_dims[i])
            dimension_out = int(layer_dims[i+1])
            if chebyshev:
                """choose chebyshev basis functions or not."""
                layer_params['single'] = chebyshev_blocks.init_chebyshev(key=key,
                                                                         n_in=dimension_in,
                                                                         n_out=dimension_out,
                                                                         d=int(k[i]),
                                                                         add_residual=add_residual,
                                                                         add_bias=add_bias,
                                                                         external_weights=external_weights)
            elif spline:
                layer_params['single'] = kan_networks_blocks.init_ka_layer(key=key,
                                                      n_in=dimension_in,
                                                      n_out=dimension_out,
                                                      g=int(g[i]),
                                                      k=int(k[i]),
                                                      add_residual=add_residual,
                                                      add_bias=add_bias,
                                                      external_weights=external_weights)
            layers.append(layer_params)
            #dimension_in = int(layer_dims[i+1])

        params['embedding_layer'] = layers
        output_dims = int(layer_dims[-1])
        return params, output_dims

    '''
    def apply_layer(params: Mapping[str, ParamTree],
                    h_one: jnp.ndarray,
                    n_in: int,
                    n_out: int,
                    g_each_layer: int,
                    k_each_layer: int,
                    grid_range: jnp.ndarray,
                    ):
        """
        :param params:
        :param h_one: input vector for each layer.
        :param n_in:
        :param n_out:
        :param g_each_layer:
        :param k_each_layer:
        :param grid_range: no grid range for chebyshev basis functions.
        :return:
        we need residual connection. It is important for the stable opt.
        """
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
        if chebyshev:
            h_one_next = chebyshev_blocks.forward_each_layer(x=h_one,
                                                             n_in=n_in,
                                                             n_out=n_out,
                                                             d=k_each_layer,
                                                             c_basis = params['c_basis'],
                                                             c_ext = params['c_ext'],
                                                             bias = params['bias'],
                                                             c_res = params['c_res'])
        elif spline:
            h_one_next = kan_networks_blocks.forward_each_layer(x=h_one,
                                                                n_in=n_in,
                                                                n_out=n_out,
                                                                g=g_each_layer,
                                                                k=k_each_layer,
                                                                grid_range=grid_range,
                                                                c_basis = params['c_basis'],
                                                                c_spl = params['c_spl'],
                                                                bias = params['bias'],
                                                                c_res = params['c_res'])
        h_one_next = residual(h_one, h_one_next)
        return h_one_next'''


    def apply_layer(params: Mapping[str, ParamTree],
                    h_one: jnp.ndarray,
                    n_in: int,
                    n_out: int,
                    g_each_layer: int,
                    k_each_layer: int,
                    grid_range: jnp.ndarray,
                    ):
        """
        :param params:
        :param h_one: input vector for each layer.
        :param n_in:
        :param n_out:
        :param g_each_layer:
        :param k_each_layer:
        :param grid_range: no grid range for chebyshev basis functions.
        :return:
        we need residual connection. It is important for the stable opt.
        """
        residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
        if chebyshev:
            h_one_next = chebyshev_blocks.forward_each_layer(x=h_one,
                                                             n_in=n_in,
                                                             n_out=n_out,
                                                             d=k_each_layer,
                                                             c_basis = params['c_basis'],
                                                             c_ext = params['c_ext'],
                                                             bias = params['bias'],
                                                             c_res = params['c_res'])
        elif spline:
            h_one_next = kan_networks_blocks.forward_each_layer(x=h_one,
                                                                n_in=n_in,
                                                                n_out=n_out,
                                                                g=g_each_layer,
                                                                k=k_each_layer,
                                                                grid_range=grid_range,
                                                                c_basis = params['c_basis'],
                                                                c_spl = params['c_spl'],
                                                                bias = params['bias'],
                                                                c_res = params['c_res'])
        h_one_next = residual(h_one, h_one_next)
        return h_one_next


    def apply(params,
              input_vector: jnp.ndarray,
              grid_range: jnp.ndarray,):
        h_one = input_vector
        for i in range(len(layer_dims)-1):
            #jax.debug.print("h_one:{}", h_one)
            h_one = apply_layer(
                                params = params['embedding_layer'][i]['single'],
                                h_one = h_one,
                                n_in = int(layer_dims[i]),
                                n_out = int(layer_dims[i+1]),
                                g_each_layer = int(g[i]),
                                k_each_layer = int(k[i]),
                                grid_range=grid_range[i],)

        return h_one

    return init, apply




def make_orbitals(nspins: Tuple[int, int],
                  charges: jnp.ndarray,
                  grid_range: jnp.ndarray,
                  nelectrons: int,
                  nfeatures: int,
                  n_parallel: int,
                  n_antiparallel: int,
                  parallel_indices: jnp.ndarray,
                  antiparallel_indices: jnp.ndarray,
                  equivariant_layers_init,
                  equivariant_layers_apply,
                  jastrow_ee_init,
                  jastrow_ee_apply,
                  g_envelope: int,
                  k_envelope: int,
                  grid_range_envelope: jnp.ndarray,
                  chebyshev: bool = False,
                  spline: bool = False,
                  simple: bool = True,):
    #equivariant_layers_init, equivariant_layers_apply = equivariant_layers()
    simple_envelope_init, simple_envelope_apply = simple_envelope.make_isotropic_envelope()


    def init(key: chex.PRNGKey) -> ParamTree:
        params = {}
        key, subkey, key_map, key_envelope, key_orbitals= jax.random.split(key, num=5)
        """we finished the parameters initialization of equivariant layers."""
        #params['layers'], output_dims = equivariant_layers_init(subkey)
        params['layers'], dims_orbital_in = equivariant_layers_init(subkey)
        """this parameters is not necessary to be a square matrix."""
        #params['map_h_to_orbitals'] = jax.random.normal(key_map, (nelectrons, output_dims))
        #params['envelopes'] = jax.random.normal(key_envelope, (3, 1, 1))

        active_spin_channels = [spin for spin in nspins if spin > 0]
        nchannels = len(active_spin_channels)
        nspin_orbitals = []
        """here, add more determinants. 25.5.2025."""
        for nspin in active_spin_channels:
            norbitals = sum(nspins) * 1 * 2 #1 is the number of determinants
            nspin_orbitals.append(norbitals)

        output_dims = [nspin_orbital // 2 for nspin_orbital in nspin_orbitals]
        orbitals = []
        for nspin_orbital in nspin_orbitals:
            key, subkey, subsubkey = jax.random.split(key, num=3)
            orbitals.append(
                normal_network_blocks.init_linear_layer(
                                                        subkey,
                                                        in_dim=dims_orbital_in,
                                                        out_dim=nspin_orbital,
                                                        include_bias=True
                )
            )

        params['orbital'] = orbitals


        """please be same with the apply function. I will reformat it into cfg file."""
        if chebyshev:
            params['orbitals'] = chebyshev_envelopes.init_chebyshev(key=key_envelope, n_in=nelectrons, n_out=nelectrons, d=k_envelope,)
        elif spline:
            params['orbitals'] = kan_envelopes.init_ka_layer(key=key_orbitals, n_in=nelectrons, n_out=nelectrons, g=g_envelope, k=k_envelope)
        elif simple:
            #params['orbitals'] = jax.random.normal(key_orbitals, (nelectrons, 3))
            #params['orbitals'] = jax.random.normal(key_orbitals, (nelectrons, 1))
            params['envelope'] = simple_envelope_init(natom=1, output_dims=output_dims, ndim=3)
        params['jastrow_ee'] = jastrow_ee_init(n_parallel=n_parallel, n_antiparallel=n_antiparallel)
        #jax.debug.print("params['jastrow_ee']:{}", params['jastrow_ee'])

        return params

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray) -> jnp.ndarray:
        """To construct the determinant, we follow suc a rule. r1 r2 r3 -> orbital1 to get, orbital1(r1), orbital1(2), orbital1(3).
        Therefore, the shape of coe_eff and coe_eff_second should be like
                            orbital1(r1), orbital1(r2), orbital1(r3)
                            orbital2(r1), orbital2(r2), orbital2(r3)
                            orbital3(r1), orbital3(r2), orbital3(r3)
        """
        #jax.debug.print("pos:{}", pos)
        #jax.debug.print("atoms:{}", atoms)
        ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
        #ae = ae/r_ae
        """we construct input layer here.23.10.2025."""
        h_one = jnp.concatenate((r_ae, ae), axis=2).reshape(nelectrons, -1)
        #ee_features = jnp.concatenate((r_ee, ee), axis=2)
        #jax.debug.print("ee_features:{}", ee_features)
        #h_two = ee_features
        #jax.debug.print("input:{}", input)
        #jax.debug.print("h_two:{}", h_two)

        #h_test = construct_symmetric_features(h_one, h_two, nspins)
        """ignore this line, it is not applied currently in our nets.3.12.2025."""
        #jax.debug.print("h_test:{}", h_test)
        """we need do more for this part to make h_test to be the input vector."""
        """to be finished...21.10.2025."""
        """we need think more about the orbitals construction."""
        h_to_orbitals = equivariant_layers_apply(params['layers'], h_one, grid_range=grid_range)
        h_to_orbitals = jnp.split(h_to_orbitals, array_partitions(nspins), axis=0)
        h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        orbitals = [
            normal_network_blocks.linear_layer(h, **p)
            for h, p in zip(h_to_orbitals, params['orbital'])
        ]
        #jax.debug.print("orbitals:{}", orbitals)

        orbitals = [orbital[..., ::2] + 1.0j * orbital[..., 1::2] for orbital in orbitals]
        #jax.debug.print("orbitals_complex:{}", orbitals)

        orbitals_angular = orbitals
        #jax.debug.print("orbitals_angular:{}", orbitals_angular)
        #jax.debug.print("r_ae:{}", r_ae)
        shape = r_ae.shape
        active_spin_channels = [spin for spin in nspins if spin > 0]
        active_spin_partitions = array_partitions(active_spin_channels)
        ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
        r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
        r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)

        r_ae = jnp.reshape(r_ae, (1, nelectrons))

        if simple:
            #jax.debug.print("active_spin_channels:{}", active_spin_channels)
            for i in range(len(active_spin_channels)):
                orbitals_angular[i] = orbitals_angular[i] * simple_envelope_apply(ae=ae_channels[i],
                                                                                  r_ae=r_ae_channels[i],
                                                                                  r_ee=r_ee_channels[i],
                                                                                  **params['envelope'][i],)
        elif chebyshev:
            envelope_chebyshev = chebyshev_envelopes.forward_each_layer(x=r_ae,
                                                                        n_in=nelectrons,
                                                                        n_out=nelectrons,
                                                                        d=k_envelope,
                                                                        c_basis=params['orbitals']['c_basis'],
                                                                        c_ext=params['orbitals']['c_ext'],
                                                                        bias=params['orbitals']['bias'],
                                                                        c_res=params['orbitals']['c_res'])

            envelope_chebyshev = jnp.reshape(envelope_chebyshev, shape)
            envelope_chebyshev = jnp.split(envelope_chebyshev, active_spin_partitions, axis=0)
            #jax.debug.print("envelope_chebyshev:{}",  envelope_chebyshev)
            for i in range(len(active_spin_channels)):
                orbitals_angular[i] = orbitals_angular[i] * envelope_chebyshev[i]
        elif spline:
            envelope_spline = kan_envelopes.forward_each_layer(x=r_ae,
                                                               n_in=nelectrons,
                                                               n_out=nelectrons,
                                                               g=g_envelope,
                                                               k=k_envelope,
                                                               grid_range=grid_range_envelope,
                                                               c_basis=params['orbitals']['c_basis'],
                                                               c_spl=params['orbitals']['c_spl'],
                                                               bias=params['orbitals']['bias'],
                                                               c_res=params['orbitals']['c_res'])
            envelope_spline = jnp.reshape(envelope_spline, shape)
            envelope_spline = jnp.split(envelope_spline, active_spin_partitions, axis=0)
            for i in range(len(active_spin_channels)):
                orbitals_angular[i] = orbitals_angular[i] * envelope_spline[i]

        #jax.debug.print("orbitals_angular_second:{}", orbitals_angular)
        shapes = [(spin, -1, sum(nspins)) for spin in active_spin_channels]
        orbitals_angular = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals_angular, shapes)]
        #jax.debug.print("orbitals_angular_before:{}", orbitals_angular)
        orbitals_angular = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals_angular]
        orbitals_angular = [jnp.concatenate(orbitals_angular, axis=1)]
        #jax.debug.print("orbitals_angular:{}", orbitals_angular)
        '''
        #h_to_orbitals = jnp.expand_dims(h_to_orbitals, 1)
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        #coe_eff = jnp.sum(h_to_orbitals * params['map_h_to_orbitals'], axis=-1)
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        h_to_orbitals = jnp.reshape(h_to_orbitals, (nelectrons, 1, -1))
        #jax.debug.print("h_to_orbitals:{}", h_to_orbitals)
        #coe_eff = [jnp.dot(h, p) for h, p in zip(h_to_orbitals, params['map_h_to_orbitals'])]
        #jax.debug.print("coe_eff:{}", coe_eff)
        #jax.debug.print("params['map_h_to_orbitals']:{}", params['map_h_to_orbitals'])
        coe_eff = h_to_orbitals * params['map_h_to_orbitals']
        #jax.debug.print("coe_eff:{}", coe_eff)
        coe_eff = jnp.sum(coe_eff, axis=-1)
        #jax.debug.print("coe_eff:{}", coe_eff)
        #jax.debug.print("r_ae:{}", r_ae)
        """for case one, we need """
        #jax.debug.print("ae:{}", ae)
        #r_ae = jnp.tile(r_ae, (nelectrons,)).reshape(nelectrons, nelectrons)
        #r_eff = r_ae + coe_eff # not necessary
        r_eff = coe_eff
        #jax.debug.print("r_eff:{}", r_eff)
        #jax.debug.print("r_ae:{}", r_ae)
        #jax.debug.print("r_eff:{}", r_eff)
        """do not forget the parameters for the envelope functions. Something is wrong."""
        if chebyshev:
            orbitals_spline_determinant = chebyshev_envelopes.forward_each_layer(x=r_eff,
                                                                                 n_in=nelectrons,
                                                                                 n_out=nelectrons,
                                                                                 d=k_envelope,
                                                                                 c_basis = params['orbitals']['c_basis'],
                                                                                 c_ext = params['orbitals']['c_ext'],
                                                                                 bias = params['orbitals']['bias'],
                                                                                 c_res = params['orbitals']['c_res'])
        elif spline:
            orbitals_spline_determinant = kan_envelopes.forward_each_layer(x=r_eff,
                                                                           n_in=nelectrons,
                                                                           n_out=nelectrons,
                                                                           g=g_envelope,
                                                                           k=k_envelope,
                                                                           grid_range=grid_range_envelope,
                                                                           c_basis =  params['orbitals']['c_basis'],
                                                                           c_spl =  params['orbitals']['c_spl'],
                                                                           bias =  params['orbitals']['bias'],
                                                                           c_res =  params['orbitals']['c_res'])
        else:
            """28.11.2025, it is proved that the format of envelope function has a large impact on the calculation result.
            We need do more on the envelope function. Consider to read some coefficients from HF."""
            jax.debug.print("params['orbitals']:{}", params['orbitals'])
            ##ae = jnp.reshape(ae, (nelectrons, 3))
            ##r_ae = jnp.reshape(r_ae, (nelectrons, -1))
            #jax.debug.print("r_ae:{}", r_ae)
            #jax.debug.print("params['orbitals']:{}", params['orbitals'])
            #envelope_exp = jnp.exp(-1 * jnp.abs(jnp.sum(params['orbitals']*ae, axis=-1)))
            ##envelope_exp = jnp.exp(-1 * jnp.abs(params['orbitals'] * r_ae))
            #jax.debug.print("r_eff:{}", r_eff)
            #jax.debug.print("envelope_exp:{}", envelope_exp)
            ##envelope_exp = jnp.reshape(envelope_exp, (-1, nelectrons))
            #jax.debug.print("envelope_exp:{}", envelope_exp)
            ##orbitals_spline_determinant = envelope_exp[None, ...] * r_eff
            #jax.debug.print("orbitals_spline_determinant:{}", orbitals_spline_determinant)
            orbitals_spline_determinant = r_eff
        #jax.debug.print("r_ee:{}", r_ee)
        """the shape of orbitals_spline_determinant should be like, 19.11.2025.
        |psi_1(r1), psi_2(r1), psi_3(r1), psi_4(r1), psi_5(r1), psi6(r1)|
        |psi_1(r2), psi_2(r2), psi_3(r2), psi_4(r2), psi_5(r2), psi6(r2)|
        ...
        |psi_1(r6), psi_2(r6), psi_3(r6), psi_4(r6), psi_5(r6), psi6(r6)|
        the normalization constant is ignored. because it can be absorbed into the neural network."""
        #r_ee = jnp.reshape(r_ee, (nelectrons, nelectrons))
        #jax.debug.print("r_ee:{}", r_ee)
        """
        jastrow = jnp.exp(jastrow_ee_apply(r_ee=r_ee,
                                           params=params['jastrow_ee'],
                                           parallel_indices=parallel_indices,
                                           antiparallel_indices=antiparallel_indices,)/nelectrons)
        """
        #jax.debug.print("orbitals_spline_determinant:{}", orbitals_spline_determinant)
        #return orbitals_spline_determinant * jastrow
        return orbitals_spline_determinant'''
        return orbitals_angular
    return init, apply


def make_kan_net(nspins: Tuple[int, int],
                 charges: jnp.ndarray,
                 nelectrons: int,
                 nfeatures: int,
                 n_parallel: int,
                 n_antiparallel: int,
                 parallel_indices: jnp.array,
                 antiparallel_indices: jnp.array,
                 layer_dims: jnp.ndarray,
                 g: jnp.ndarray,
                 k: jnp.ndarray,
                 grid_range: jnp.ndarray,
                 g_envelope: int,
                 k_envelope: int,
                 grid_range_envelope: jnp.ndarray,
                 natoms: int,
                 ndims: int = 3,
                 chebyshev: bool = True,
                 spline: bool = False,
                 add_residual: bool = False,
                 add_bias: bool = True,
                 external_weights: bool = True,
                 envelope_chebyshev: bool = False,
                 envelope_spline: bool = False,
                 envelope_simple: bool = True,
                 ):
    """
    nspins: the spin configuration.
    nelectrons: number of electrons.
    natoms: number of atoms.
    nfeatures: it is the number of features, it should be (number of atoms) * 4 for each electron.
    layer_dims: it is an array. [m_1, m_2, m_3, ..., m_n], m_1 must be same with nfeatures. m_n must be same with nelectrons, i.e., the number of orbitals.
    make_kan_net_layers is the equivariant layer based on Kolmogorov-Arnold Networks.
    Currently, it is only working for single atom. But no limit for electrons.
    ndims: the number of dimensions.
    g: the grid number on each layer. We allow different layer uses different grids.
    k: the oder of spline functions on each layer. We allow different layer uses different order of spline functions.
    chebyshev: whether to use chebyshev orbitals. If it is true, the grid information should be deleted. however, the order of degree should be kept.
    """
    #feature_layer = make_kan_features(natoms=natoms, ndim=ndims)
    """ to be continued... we need add the module about chebyshev polynomials. 18.11.2025."""
    """we need change the way to turn on the different envelope functions.1.12.2025 """
    kan_equivariant_layers_init, kan_equivariant_layers_apply = make_kan_net_layers(layer_dims=layer_dims,
                                                                                    g=g,
                                                                                    k=k,
                                                                                    chebyshev=chebyshev,
                                                                                    spline=spline,
                                                                                    add_residual=add_residual,
                                                                                    add_bias=add_bias,
                                                                                    external_weights=external_weights,
                                                                                    )
    jastrow_ee_init, jastrow_ee_apply = make_pade_ee_jastrow()
    orbitals_init, orbitals_apply = make_orbitals(nspins=nspins,
                                                  charges=charges,
                                                  grid_range=grid_range,
                                                  nelectrons=nelectrons,
                                                  nfeatures=nfeatures,
                                                  n_parallel=n_parallel,
                                                  n_antiparallel=n_antiparallel,
                                                  parallel_indices=parallel_indices,
                                                  antiparallel_indices=antiparallel_indices,
                                                  equivariant_layers_init=kan_equivariant_layers_init,
                                                  equivariant_layers_apply=kan_equivariant_layers_apply,
                                                  jastrow_ee_init=jastrow_ee_init,
                                                  jastrow_ee_apply=jastrow_ee_apply,
                                                  g_envelope=g_envelope,
                                                  k_envelope=k_envelope,
                                                  grid_range_envelope=grid_range_envelope,
                                                  chebyshev=envelope_chebyshev,
                                                  spline=envelope_spline,
                                                  simple=envelope_simple,)

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(key)

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray,):
        determinant = orbitals_apply(params, pos, spins, atoms, charges)
        #jax.debug.print("determinant:{}", determinant)
        #sign, logdet = jnp.linalg.slogdet(determinant)
        sign, logdet = normal_network_blocks.logdet_matmul(determinant)
        """we only consider single determinant.23.10.2025."""
        return sign, logdet

    def orbitals(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray, ):
        determinant = orbitals_apply(params, pos, spins, atoms, charges)
        return determinant

    return init, apply, orbitals


