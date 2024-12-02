"""Evaluates the pseudopotential Hamiltonian on a wavefunction. 04.09.2024."""

from typing import Sequence

import chex
import jax
import jax.numpy as jnp
import kfac_jax
from AIQMCbatch3adm import main_adam
from AIQMCbatch3adm import nn


signed_network, data, params, log_network = main_adam.main()
jax.debug.print("data:{}", data)
ndim = 3
#ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
#jax.debug.print("ae:{}", ae)

rn_local = jnp.array([1, 3, 2])
rn_non_local = jnp.array([2])
local_coefficient = jnp.array([4.00000, 57.74008, -25.81955])
nonlocal_coefficient = jnp.array([52.13345])
local_exponent = jnp.array([14.43502, 8.39889, 7.38188])
nonlocal_exponent = jnp.array([7.76079])


"""for many atoms test, we reconstruct the list of coefficients and exponents. 
For example, we have two C atoms.
"""
rn_local_general = jnp.array([[1, 3, 2], [1, 3, 2]])
rn_non_local_general = jnp.array([[2], [2]])
local_coefficient_general = jnp.array([[4.00000, 57.74008, -25.81955], [4.00000, 57.74008, -25.81955]])
nonlocal_coefficient_general = jnp.array([[52.13345], [52.13345]])
local_exponent_general = jnp.array([[14.43502, 8.39889, 7.38188], [14.43502, 8.39889, 7.38188]])
nonlocal_exponent_general = jnp.array([[7.76079], [7.76079]])




def get_v_l(data: nn.AINetData, rn_local: jnp.array, local_coefficient: jnp.array, local_exponent: jnp.array,):
    """calculate the local part of pseudopotential energy.
    we need make the method be general. It means that we could have many atoms which may enlarge the dimension of the coefficient and exponent array."""
    nelectron = 4
    natoms = 2
    rn_local = rn_local - 2
    ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1)
    local_part1 = -1 * data.charges/r_ae
    r_ae = jnp.reshape(r_ae, (nelectron, natoms, 1))
    
    def exp_single(r_ae: jnp.array, local_exponent: jnp.array, rn_local: jnp.array, local_coefficient: jnp.array):
        return local_coefficient * r_ae**rn_local * jnp.exp(-local_exponent * jnp.square(r_ae))

    local_part2_parllel = jax.vmap(exp_single, in_axes=(0, None, None, None), out_axes=0)
    local_energy_part2 = local_part2_parllel(r_ae, local_exponent, rn_local, local_coefficient)
    local_energy_part2 = jnp.sum(local_energy_part2, axis=-1)
    total_local_energy = local_part1 + local_energy_part2
    return total_local_energy


#get_v_l_parallel = jax.pmap(jax.vmap(get_v_l,
#                                    in_axes=(nn.AINetData(positions=0, atoms=0, charges=0), None, None, None,)),
#                           in_axes=(0, None, None, None,))
"""if we have more hosts, we can duplicate the arrays to multi devices."""
#output = get_v_l_parallel(data, rn_local_general, local_coefficient_general, local_exponent_general)


def get_non_v_l(data: nn.AINetData, rn_non_local: jnp.array, non_local_coefficient: jnp.array, non_local_exponent: jnp.array):
    nelectron = 4
    natoms = 2
    ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1)
    r_ae = jnp.reshape(r_ae, (nelectron, natoms, 1))

    def exp_non_single(r_ae: jnp.array, rn_non_local: jnp.array, non_local_coefficient: jnp.array, non_local_exponent: jnp.array):
        return non_local_coefficient * r_ae ** rn_non_local * jnp.exp(-non_local_exponent * jnp.square(r_ae))

    non_local_parallel = jax.vmap(exp_non_single, in_axes=(0, None, None, None), out_axes=0)
    non_local_output = non_local_parallel(r_ae,  non_local_exponent, rn_non_local, non_local_coefficient)
    #jax.debug.print("non_local_output:{}", non_local_output)
    return non_local_output


'''
get_non_v_l_parallel = jax.pmap(jax.vmap(get_non_v_l,
                                    in_axes=(nn.AINetData(positions=0, atoms=0, charges=0), None, None, None,)),
                           in_axes=(0, None, None, None,))
output1 = get_non_v_l_parallel(data, rn_non_local_general, nonlocal_coefficient_general, nonlocal_exponent_general)
'''

def generate_quadrature_grids():
    """generate quadrature grids from Mitas, Shirley, and Ceperley."""
    """Generate in Cartesian grids for octahedral symmetry.
    We are not going to give more options for users, so just default 50 integration points."""
    octpts = jnp.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T
    #jax.debug.print("octpts:{}", octpts)
    nonzero_count = jnp.count_nonzero(octpts, axis=1)
    #jax.debug.print("nonzero_count:{}", nonzero_count)
    OA = octpts[nonzero_count == 1]
    OB = octpts[nonzero_count == 2] / jnp.sqrt(2)
    OC = octpts[nonzero_count == 3] / jnp.sqrt(3)
    #jax.debug.print("OA:{}", OA)
    #jax.debug.print("OB:{}", OB)
    #jax.debug.print("OC:{}", OC)
    d1 = OC * jnp.sqrt(3 / 11)
    #jax.debug.print("d1:{}", d1)
    OD1 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0], (1, -1)), jnp.reshape(d1[:, 1], (1, -1)), jnp.reshape(d1[:, 2] * 3, (1, -1))), axis=0))
    OD2 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0], (1, -1)), jnp.reshape(d1[:, 1] * 3, (1, -1)), jnp.reshape(d1[:, 2], (1, -1))), axis=0))
    OD3 = jnp.transpose(jnp.concatenate((jnp.reshape(d1[:, 0] * 3, (1, -1)), jnp.reshape(d1[:, 1], (1, -1)), jnp.reshape(d1[:, 2], (1, -1))), axis=0))
    #jax.debug.print("OD1:{}", OD1)
    #jax.debug.print("OD2:{}", OD2)
    #jax.debug.print("OD3:{}", OD3)
    OD = jnp.concatenate((OD1, OD2, OD3), axis=0)
    #jax.debug.print("OD:{}", OD)
    #coordinates = jnp.stack((OA, OB, OC, OD), axis=1)
    weights = jnp.array([[4/315], [64/2835], [27/1280], [14641/725760]])
    return OA, OB, OC, OD, weights


#output2 = generate_quadrature_grids()
#jax.debug.print("output2:{}", output2)

def get_rot(batch_size: int, key: chex.PRNGKey):
    """actually, here, we generate the normal rotation matrix to """
    key, subkey = jax.random.split(key)
    """here, we dont use random.Rotation. Because this function is not working currently."""
    rot = jax.random.orthogonal(key=key, n=3, shape=(batch_size,))
    #jax.debug.print("rot:{}", rot)
    OA, OB, OC, OD, weights = generate_quadrature_grids()
    """actually, I dont understand how to use jnp.einsum, but currently it is working."""
    Points_OA = jnp.einsum('jkl,ik->jil', rot, OA,)
    Points_OB = jnp.einsum('jkl,ik->jil', rot, OB,)
    Points_OC = jnp.einsum('jkl,ik->jil', rot, OC,)
    Points_OD = jnp.einsum('jkl,ik->jil', rot, OD,)
    #jax.debug.print("Points_OD:{}", Points_OD)
    return Points_OA, Points_OB, Points_OC, Points_OD, weights


def P_l_0(x):
    """we should be aware of judgement. now, we need rewrite this part to make it run efficiently on GPU.
    l = 0"""
    return jnp.ones(x.shape)


def P_l_1(x):
    return x


def P_l_2(x):
    return 0.5 * (3 * x * x - 1)


def P_l_3(x):
    return 0.5 * (5 * x * x * x - 3 * x)


def P_l_4(x):
    return 0.125 * (35 * x * x * x * x - 30 * x * x + 3)


l_list = jnp.array([0, 1])
key = jax.random.PRNGKey(1)
#sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
#sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
Points_OA, Points_OB, Points_OC, Points_OD, weights = get_rot(4, key)


def get_P_l(data: nn.AINetData, params: nn.ParamTree, Points:jnp.array, weights: float, l_list: jnp.array):
    nelectron = 4
    natoms = 2
    ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1)
    r_ae = jnp.reshape(r_ae, (nelectron, natoms, 1))

    def rot_coords_single(r_ae: jnp.array, Points: jnp.array):
        return r_ae * Points

    rot_coords_parallel = jax.vmap(jax.vmap(rot_coords_single, in_axes=(0, None), out_axes=0), in_axes=(0, None), out_axes=0)
    roted_coords = rot_coords_parallel(r_ae, Points)

    def calculate_cos_theta_single(ae: jnp.array, roted_coords: jnp.array):
        return jnp.sum(ae * roted_coords, axis=-1)/(jnp.linalg.norm(ae) * jnp.linalg.norm(roted_coords))

    calculate_cos_theta_parallel = jax.vmap(jax.vmap(calculate_cos_theta_single, in_axes=(0, 0), out_axes=0), in_axes=(0, 0))
    cos_theta = calculate_cos_theta_parallel(ae, roted_coords)
    jax.debug.print("cos_theta:{}", cos_theta)
    """then we need calculate the value of the wavefunction."""
    order = jnp.arange(0, nelectron, step=1)
    x1 = data.positions
    x2 = jnp.reshape(x1, (nelectron, ndim))

    def return_arrays(x2: jnp.array, roted_coords: jnp.array, order1: jnp.array):
        temp = x2.at[order1].set(roted_coords)
        temp = jnp.reshape(temp, (-1))
        return temp

    return_arrays_parallel = jax.vmap(jax.vmap(jax.vmap(return_arrays,
                                                        in_axes=(None, 0, None), out_axes=0),
                                               in_axes=(None, 0, None), out_axes=0),
                                      in_axes=(None, 0, 0))

    denominator = log_network(params, x1, data.atoms, data.charges)

    roted_configurations = return_arrays_parallel(x2, roted_coords, order)
    #jax.debug.print("roted_configurations:{}", roted_configurations)
    batch_lognetwork = jax.vmap(jax.vmap(jax.vmap(log_network,
                                                  in_axes=(None, 0, None, None), out_axes=0),
                                in_axes=(None, 0, None, None), out_axes=0),
                                in_axes=(None, 0, None, None), out_axes=0)
    roted_wavefunciton_value = batch_lognetwork(params, roted_configurations, data.atoms, data.charges)
    #jax.debug.print("roted_wavefunction_value:{}", roted_wavefunciton_value)
    ratios = roted_wavefunciton_value/denominator
    jax.debug.print("ratios:{}", ratios)
    """the following part is not general. We need think about the situation like CO2 or SiO2. 2.12.2024."""
    return cos_theta, ratios





get_P_l_parallel = jax.pmap(jax.vmap(get_P_l,
                                    in_axes=(nn.AINetData(positions=0, atoms=0, charges=0), None, 0,  None, None)),
                           in_axes=(0, 0, None, None, None))
output2 = get_P_l_parallel(data, params, Points_OA, weights[0], l_list)