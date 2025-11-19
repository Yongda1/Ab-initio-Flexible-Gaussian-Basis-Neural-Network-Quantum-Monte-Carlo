#from jaxkan.KAN import KAN
import jax.numpy as jnp
import jax
import chex
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import kan_networks_blocks_case_one as kan_networks_blocks
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import chebyshev_blocks as chebyshev_blocks
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import kan_envelopes_case_one_general as kan_envelopes
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one import chebyshev_envelopes
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.JastrowPade import make_pade_ee_jastrow


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
    """Constructs inputs to Fermi Net from raw electron and atomic positions."""
    #jax.debug.print("atoms:{}", atoms)
    assert atoms.shape[1] == ndim
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


def make_kan_features(natoms: int, ndim: int = 3):
    def init() -> Tuple[Tuple[int], Param]:
        return (natoms * (ndim +1),), {}

    def apply(ae, r_ae) -> jnp.ndarray:
        ae_features = jnp.concatenate((r_ae, ae), axis=2)
        ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
        return ae_features
    return init, apply



def make_kan_net_layers(layer_dims: jnp.ndarray,
                        g: jnp.ndarray,
                        k: jnp.ndarray,
                        chebyshev: bool = False,):
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
                                                                         add_residual=True,
                                                                         add_bias=True,
                                                                         external_weights=True)
            else:
                layer_params['single'] = kan_networks_blocks.init_ka_layer(key=key,
                                                      n_in=dimension_in,
                                                      n_out=dimension_out,
                                                      g=int(g[i]),
                                                      k=int(k[i]),
                                                      add_residual=True,
                                                      add_bias=True,
                                                      external_weights=True)
            layers.append(layer_params)
            #dimension_in = int(layer_dims[i+1])

        params['embedding_layer'] = layers
        output_dims = int(layer_dims[-1])
        return params, output_dims


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
        """
        if chebyshev:
            h_one_next = chebyshev_blocks.forward_each_layer(x=h_one,
                                                             n_in=n_in,
                                                             n_out=n_out,
                                                             d=k_each_layer,
                                                             c_basis = params['c_basis'],
                                                             c_ext = params['c_ext'],
                                                             bias = params['bias'],
                                                             c_res = params['c_res'])
        else:
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
                  chebyshev: bool = False,):
    #equivariant_layers_init, equivariant_layers_apply = equivariant_layers()


    def init(key: chex.PRNGKey) -> ParamTree:
        params = {}
        key, subkey, key_map, key_envelope, key_orbitals= jax.random.split(key, num=5)
        """we finished the parameters initialization of equivariant layers."""
        params['layers'], output_dims = equivariant_layers_init(subkey)
        """this parameters is not necessary to be a square matrix."""
        params['map_h_to_orbitals'] = jax.random.normal(key_map, (nelectrons, output_dims))
        #params['envelopes'] = jax.random.normal(key_envelope, (3, 1, 1))
        """please be same with the apply function. I will reformat it into cfg file."""
        if chebyshev:
            params['orbitals'] = chebyshev_envelopes.init_chebyshev(key=key_envelope, n_in=nelectrons, n_out=nelectrons, d=k_envelope,)
        else:
            params['orbitals'] = kan_envelopes.init_ka_layer(key=key_orbitals, n_in=nelectrons, n_out=nelectrons, g=g_envelope, k=k_envelope)
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
        #jax.debug.print("ae:{}", ae)
        #jax.debug.print("r_ae: {}", r_ae)
        #nfeatures = 4
        #nelectrons = 6
        """we construct input layer here.23.10.2025."""
        input_layer = jnp.concatenate((r_ae, ae), axis=2).reshape(nelectrons, -1)
        #jax.debug.print("input:{}", input)
        """to be finished...21.10.2025."""
        """we need think more about the orbitals construction."""
        h_to_orbitals = equivariant_layers_apply(params['layers'], input_layer, grid_range=grid_range)
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
        #r_ae = jnp.tile(r_ae, (nelectrons,)).reshape(nelectrons, nelectrons)

        #r_eff = r_ae + coe_eff # not necessary
        r_eff = coe_eff
        #jax.debug.print("r_ae:{}", r_ae)
        jax.debug.print("r_eff:{}", r_eff)
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
        else:
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
        #jax.debug.print("r_ee:{}", r_ee)
        """the shape of orbitals_spline_determinant should be like,
        |psi_1(r1), psi_2(r1), psi_3(r1), psi_4(r1), psi_5(r1), psi(r1)|
        |psi_1(r2), psi_2(r2), psi_3(r2), psi_4(r2), psi_5(r2), psi(r2)|
        ...
        |psi_1(r6), psi_2(r6), psi_3(r6), psi_4(r6), psi_5(r6), psi(r6)|"""
        r_ee = jnp.reshape(r_ee, (nelectrons, nelectrons))
        #jax.debug.print("r_ee:{}", r_ee)
        jastrow = jnp.exp(jastrow_ee_apply(r_ee=r_ee,
                                           params=params['jastrow_ee'],
                                           parallel_indices=parallel_indices,
                                           antiparallel_indices=antiparallel_indices,)/nelectrons)
        jax.debug.print("orbitals_spline_determinant:{}", orbitals_spline_determinant)
        return orbitals_spline_determinant * jastrow
    return init, apply


def make_kan_net(nspins: Tuple[int, int],
                 charges: jnp.ndarray,
                 nelectrons: int,
                 nfeatures: int,
                 n_parallel: int,
                 n_antiparallel: int,
                 parallel_indices: jnp.array,
                 antiparallel_indices: jnp.array,
                 layer_dims : jnp.ndarray,
                 g: jnp.ndarray,
                 k: jnp.ndarray,
                 grid_range: jnp.ndarray,
                 g_envelope: int,
                 k_envelope: int,
                 grid_range_envelope: jnp.ndarray,
                 natoms: int,
                 ndims: int=3,
                 chebyshev: bool = False,
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
    kan_equivariant_layers_init, kan_equivariant_layers_apply = make_kan_net_layers(layer_dims=layer_dims,
                                                                                    g=g,
                                                                                    k=k,
                                                                                    chebyshev=chebyshev,)
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
                                                  chebyshev=chebyshev)

    def init(key: chex.PRNGKey) -> ParamTree:
        key, subkey = jax.random.split(key, num=2)
        return orbitals_init(key)

    def apply(params,
              pos: jnp.ndarray,
              spins: jnp.ndarray,
              atoms: jnp.ndarray,
              charges: jnp.ndarray,):
        determinant = orbitals_apply(params, pos, spins, atoms, charges)
        sign, logdet = jnp.linalg.slogdet(determinant)
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


