"""Evaluates the pseudopotential Hamiltonian on a wavefunction. 04.09.2024."""

from typing import Sequence

import chex
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

r_ae = jnp.array([[[0.21824889, 0.3565338], [0.1946077, 0.32006422], [0.4780831,  0.138754], [0.41992992, 0.19055614]],
                 [[0.16530964, 0.29526055], [0.15191387, 0.22501956], [0.3564806, 0.05262673], [0.45009968, 0.16455044]],
                 [[0.35471296, 0.65752304], [0.08244702, 0.36039594], [0.48147705, 0.13537169], [0.1520589, 0.22781217]],
                 [[0.08920264, 0.26871547], [0.20597123, 0.25272587], [0.23355496, 0.22838382], [0.32041857, 0.20322587]]])

ae=jnp.array([[[[-0.05045887, -0.05971689,  0.06463739], [-0.25045887, -0.2597169,  -0.13536263]],
     [[ 0.07916525, -0.05145506,  0.09662296], [-0.12083475, -0.25145507, -0.10337704]],
     [[ 0.07315412, 0.01910421, 0.0914843 ], [-0.12684588, -0.18089579, -0.1085157]],
     [[ 0.21733882, 0.19526899, 0.16549167], [ 0.01733881, -0.00473101, -0.03450833]]],
    [[[ 0.0466635, 0.01833163, -0.10274688], [-0.1533365,  -0.18166837, -0.3027469 ]],
     [[-0.12950271, -0.06703701, 0.10728339], [-0.3295027, -0.267037,   -0.09271661]],
     [[ 0.24344115, 0.26252785,  0.21634525], [ 0.04344115, 0.06252785,  0.01634525]],
     [[ 0.08412638, 0.14403898, 0.22974347], [-0.11587363, -0.05596103, 0.02974346]]],
    [[[-0.01945793, 0.21137373, -0.01259473], [-0.21945794, 0.01137373, -0.21259473]],
     [[ 0.14928178, -0.14417866, 0.04273015], [-0.05071822, -0.34417868, -0.15726987]],
     [[ 0.32278678, 0.1137078,   0.31117824], [ 0.12278678, -0.08629221,  0.11117823]],
     [[ 0.17437936, 0.19614023,  0.19370084], [-0.02562064, -0.00385977, -0.00629917]]],
    [[[-0.11490166, -0.15279403, -0.11633499], [-0.31490165, -0.35279405, -0.316335]],
     [[ 0.09555221, -0.11339962,  0.12491734], [-0.10444779, -0.3133996,  -0.07508266]],
     [[ 0.18371007,  0.3944462,   0.14370795], [-0.01628993,  0.19444619, -0.05629206]],
     [[ 0.28625673,  0.31620038,  0.12505463], [0.08625673,  0.11620037, -0.07494538]]]])



rn_local = jnp.array([1, 3, 2])
rn_non_local = jnp.array([2])
local_coefficient = jnp.array([4.00000, 57.74008, -25.81955])
nonlocal_coefficient = jnp.array([52.13345])
local_exponent = jnp.array([14.43502, 8.39889, 7.38188])
nonlocal_exponent = jnp.array([7.76079])


def get_v_l(r_ae: jnp.array, rn_local: jnp.array,
                           local_coefficient: jnp.array, local_exponent: jnp.array,
                           symbols: Sequence[str], batch_size: int):
    """here, we are not going to write general codes. Currently, we only implement the C atom.
    We can make this function better later. 06.09.2024."""
    jax.debug.print("r_ae:{}", r_ae)
    rn_local = rn_local - 2
    #jax.debug.print("rn_local:{}", rn_local)
    """here, we need match the shape of rn_local with r_ae, coefficient, exponent.05.09.2024."""
    rn_local = jnp.repeat(jnp.reshape(rn_local, (1, -1)), batch_size, axis=0)
    #jax.debug.print("rn_local:{}", rn_local)
    rn_local = jnp.reshape(rn_local, (batch_size, 1, -1))
    #jax.debug.print("rn_local:{}", rn_local)
    rn_local = jnp.repeat(rn_local, 4, axis=1)
    #jax.debug.print("rn_local:{}", rn_local)
    rn_local = jnp.reshape(rn_local, (batch_size, 4, 1, -1))
    #jax.debug.print("rn_local:{}", rn_local)
    r_ae = jnp.reshape(r_ae, (batch_size, 4, 2, 1))
    #jax.debug.print("r_ae:{}", r_ae)
    #first_part = r_ae^rn_local
    first_part = jnp.power(r_ae, rn_local)
    #jax.debug.print("first_part:{}", first_part)
    #jax.debug.print("exponent:{}", local_exponent)
    local_exponent = jnp.repeat(jnp.reshape(local_exponent, (1, -1)), batch_size, axis=0)
    local_exponent = jnp.reshape(local_exponent, (batch_size, 1, -1))
    local_exponent = jnp.repeat(local_exponent, 4, axis=1) # 4 is the number of electrons.
    local_exponent = jnp.reshape(local_exponent, (batch_size, 4, 1, -1))
    #local_exponent = jnp.repeat(local_exponent, 2, axis=1) # 2 is the number of atoms.
    #local_exponent = jnp.reshape(local_exponent, jnp.shape(first_part))
    #jax.debug.print("exponent:{}", local_exponent)
    second_part = jnp.exp(-1 * local_exponent * jnp.square(r_ae))
    #jax.debug.print("second_part:{}", second_part)
    #jax.debug.print("coefficient:{}", local_coefficient)
    local_coefficient = jnp.repeat(jnp.reshape(local_coefficient, (1, -1)), batch_size, axis=0)
    local_coefficient = jnp.reshape(local_coefficient, (batch_size, 1, -1))
    local_coefficient = jnp.repeat(local_coefficient, 4, axis=1)  # 4 is the number of electrons. 2 is the number of atmos.
    local_coefficient = jnp.reshape(local_coefficient, (batch_size, 4, 1, -1))
    local_coefficient = jnp.repeat(local_coefficient, 2, axis=1)
    local_coefficient = jnp.reshape(local_coefficient, jnp.shape(first_part))
    #jax.debug.print("local_coefficient:{}", local_coefficient)
    local_part_total = local_coefficient * first_part * second_part
    #jax.debug.print("local_part_total:{}", local_part_total)
    local_part_pp_energy = jnp.sum(local_part_total, axis=-1) + -1 * 1/jnp.reshape(r_ae, (batch_size, 4, 2)) * 2 # here, 2 is Z_eff.
    #jax.debug.print("output:{}", local_part_pp_energy)
    return local_part_pp_energy


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
    #jax.debug.print("OA:{}", OA)
    #OA = jnp.reshape(OA, (6, 1, 3))
    #jax.debug.print("OA:{}", OA)
    #rot = jnp.reshape(rot, (1, batch_size, 3, 3))
    #jax.debug.print("rot:{}", rot)
    """actually, I dont understand how to use jnp.einsum, but currently it is working."""
    Points_OA = jnp.einsum('jkl,ik->jil', rot, OA,)
    Points_OB = jnp.einsum('jkl,ik->jil', rot, OB,)
    Points_OC = jnp.einsum('jkl,ik->jil', rot, OC, )
    Points_OD = jnp.einsum('jkl,ik->jil', rot, OD, )
    #jax.debug.print("Points_OD:{}", Points_OD)
    return Points_OA, Points_OB, Points_OC, Points_OD, weights





#Points_OA, Points_OB, Points_OC, Points_OD, weights = get_rot(4, key=key)
#jax.debug.print("output3:{}", output3)

def P_l(x, l):
    """we should be aware of judgement."""
    if l == 0:
        return np.ones(x.shape)
    elif l == 1:
        return x
    elif l == 2:
        return 0.5 * (3 * x * x - 1)
    elif l == 3:
        return 0.5 * (5 * x * x * x - 3 * x)
    elif l == 4:
        return 0.125 * (35 * x * x * x * x - 30 * x * x + 3)
    else:
        raise NotImplementedError(f"Legendre functions for l>4 not implemented {l}")

def get_P_l(r_ae: jnp.array, batch_size: int, key: chex.PRNGKey):
    """We need think more about this part. 06.09.2024.
    Here, we need generate the 50 coordinates of integration points.11.09.2024."""
    Points_OA, Points_OB, Points_OC, Points_OD, weights = get_rot(batch_size, key=key)
    jax.debug.print("r_ae:{}", r_ae)
    #jax.debug.print("Points_OA:{}", Points_OA)
    """then, we need generate first 6 points of the OA array from the radius r_ea.
    we should be aware that the Points_OA is the array including the points on the normal sphere. So, if we 
    need all the cartesian coordinates of these points, we just need multiply the coordinates by the radius."""
    Points_OA = jnp.reshape(Points_OA, (batch_size, 1, 1, 6, 3))
    jax.debug.print("Points_OA:{}", Points_OA)
    """we need match the shape of r_ae and Points_OA to generate these points.11.09.2024.
    we need a long time to solve this problem."""
    r_ae = jnp.reshape(r_ae, (batch_size, 4, 2, 1))# 4 is the number of electrons. 2 is the number of atoms.
    jax.debug.print("r_ae:{}", r_ae)

key = jax.random.PRNGKey(seed=1)
output4 = get_P_l(r_ae=r_ae, batch_size=4, key=key)


def get_v_nonlocal(ae: jnp.array, rn_non_local: jnp.array, non_local_coefficient: jnp.array, non_local_exponent: jnp.array):
    """evaluate the nonlocal part pp energy. 06.09.2024."""
    jax.debug.print("ae:{}", ae)
    jax.debug.print("rn_non_local:{}", rn_non_local)
    jax.debug.print("non_local_coefficient:{}", non_local_coefficient)
    jax.debug.print("non_local_exponent:{}", non_local_exponent)
    

#output = get_v_l(r_ae=r_ae, rn_local=rn_local, local_coefficient=local_coefficient,
#                local_exponent=local_exponent, symbols=['C', 'C'], batch_size=4)

#outpu1 = get_v_nonlocal(ae=ae, rn_non_local=rn_non_local, non_local_coefficient=nonlocal_coefficient, non_local_exponent=nonlocal_exponent)


def ecp_ea(r_ae: jnp.array, batch_size: int, charges: jnp.array, symbols: Sequence[str], quad_degree: int=4, ecp: str='ccecp', complex_output: bool=True,):
    """we contiune this tomorrow. 04.09.2024.
    here, we use the method in Pyqmc to do pseduopotential calculation. Later, we can use grid to discrete the pseudopotential."""
    output = get_v_l(r_ae, symbols)