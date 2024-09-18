"""Evaluates the pseudopotential Hamiltonian on a wavefunction. 04.09.2024."""

from typing import Sequence

import chex
import jax
from jax import lax
import jax.numpy as jnp
from AIQMC import main
from AIQMC import nn

signednetwork, data, batchparams, batchphase, batchnetwork = main.main()
print("data.positions", data.positions)

r_ae = jnp.array([[[0.21824889, 0.3565338], [0.1946077, 0.32006422], [0.4780831,  0.138754], [0.41992992, 0.19055614]],
                 [[0.16530964, 0.29526055], [0.15191387, 0.22501956], [0.3564806, 0.05262673], [0.45009968, 0.16455044]],
                 [[0.35471296, 0.65752304], [0.08244702, 0.36039594], [0.48147705, 0.13537169], [0.1520589, 0.22781217]],
                 [[0.08920264, 0.26871547], [0.20597123, 0.25272587], [0.23355496, 0.22838382], [0.32041857, 0.20322587]]])

ae = jnp.array([[[[-0.05045887, -0.05971689,  0.06463739], [-0.25045887, -0.2597169,  -0.13536263]],
     [[ 0.07916525, -0.05145506,  0.09662296], [-0.12083475, -0.25145507, -0.10337704]],
     [[ 0.07315412, 0.01910421, 0.0914843 ], [-0.12684588, -0.18089579, -0.1085157]],
     [[ 0.21733882, 0.19526899, 0.16549167], [ 0.01733881, -0.00473101, -0.03450833]]],
    [[[ 0.0466635, 0.01833163, -0.10274688], [-0.1533365,  -0.18166837, -0.3027469]],
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


def get_summation_legrend(r_ae: jnp.array, Points: jnp.array, number_points: int,
                          batch_size: int, nelectrons: int, natoms: int, weights: jnp.array,):
    r_ae_O = jnp.reshape(jnp.repeat(jnp.reshape(r_ae, (batch_size, -1, 1)), number_points, axis=-1), (-1, number_points, 1))  # 12 is the number of OB points.
    Points_O = jnp.reshape(jnp.repeat(Points, nelectrons * natoms, axis=0), (-1, number_points, 3))
    r_rot_coord_O = r_ae_O * Points_O
    r_rot_coord_O = jnp.reshape(r_rot_coord_O, (batch_size, nelectrons, natoms, number_points, 3))
    #jax.debug.print("r_rot_coord_0:{}", r_rot_coord_O)
    cos_theta_O = r_rot_coord_O[:, :, :, :, 0] / jnp.reshape(r_ae_O, (batch_size, nelectrons, natoms, number_points))
    # jax.debug.print("cos_theta_OB:{}", cos_theta_OB)
    l_list_O = jnp.repeat(jnp.reshape(jnp.repeat(l_list, batch_size, axis=0), (batch_size, -1, 1)), natoms * number_points, axis=-1)
    l_list_O = jnp.reshape(l_list_O, (jnp.shape(cos_theta_O)))
    P_l_value_O = (2 * l_list_O + 1) * P_l_0(cos_theta_O) * weights
    #jax.debug.print("P_l_value_OC:{}", P_l_value_OC)
    return P_l_value_O


def get_P_l(batch_network: nn.AINetLike, batch_phase: nn.LogAINetLike, batchparams: nn.ParamTree, data: nn.AINetData, r_ae: jnp.array, batch_size: int, nelectrons: int, natoms: int, key: chex.PRNGKey, l_list: jnp.array):
    """We need think more about this part. 06.09.2024.
    Here, we need generate the 50 coordinates of integration points.11.09.2024."""
    Points_OA, Points_OB, Points_OC, Points_OD, weights = get_rot(batch_size, key=key)
    """then, we need generate first 6 points of the OA array from the radius r_ea.
    we should be aware that the Points_OA is the array including the points on the normal sphere. So, if we 
    need all the cartesian coordinates of these points, we just need multiply the coordinates by the radius."""
    """we need match the shape of r_ae and Points_OA to generate these points.11.09.2024.
    we need a long time to solve this problem.
    12.09.2024, for convenience, we only show how to do OA points summation."""
    r_ae_OA = jnp.reshape(r_ae, (batch_size, -1, 1))
    r_ae_OA = jnp.repeat(r_ae_OA, 6, axis=-1) # 6 is the number of points in OA.
    r_ae_OA = jnp.reshape(r_ae_OA, (-1, 6, 1))
    #jax.debug.print("r_ae_OA:{}", r_ae_OA)
    Points_OA = jnp.repeat(Points_OA, nelectrons*natoms, axis=0)
    Points_OA = jnp.reshape(Points_OA, (-1, 6, 3))
    r_rot_coord_OA = r_ae_OA * Points_OA
    r_rot_coord_OA = jnp.reshape(r_rot_coord_OA, (batch_size, nelectrons, natoms, 6, 3))
    #jax.debug.print("r_rot_coord_OA:{}", r_rot_coord_OA)
    #jax.debug.print("data.atoms:{}", data.atoms)
    coord_atoms = jnp.reshape(data.atoms, (batch_size, natoms, 3))
    coord_atoms = jnp.repeat(coord_atoms, 6, axis=1)
    coord_atoms = jnp.repeat(coord_atoms, nelectrons, axis=0)
    coord_atoms = jnp.reshape(coord_atoms, (batch_size, nelectrons, natoms, 6, 3))
    #jax.debug.print("coord_atoms:{}", coord_atoms)
    pos_integration_points = coord_atoms + r_rot_coord_OA
    jax.debug.print("pos_integration_points:{}", pos_integration_points)
    jax.debug.print("data.positions:{}", data.positions)
    x_denominator = jnp.reshape(data.positions, (batch_size, nelectrons, 3))
    jax.debug.print("x_denominator:{}", x_denominator)
    """here, we outline the problems explicitly. For OA points, every atom needs 6 integration points. This means, for the system with two atoms,
    we need 12 integration points to evaluate the non-local part for one electron. Therefore, we need replace the coordinates of the electron by the coordiantes of 
    12 integration points. However, if we really just replace the electron coordinate 12 times, this must cost a lot. And the codes cannot run efficiently.
    13.09.2024."""
    """14.09.2024, we have to compare the computational cost between all electron calculation and pseudopotential calculation.
    And if we want to make this method be general, we need think more about the input and output parameters.!!!"""
    """currently, we first finished the 6 points part, i.e. OA points.
    Here, we dont get ideas to make the run to be parallel on GPU. Currently, we only solve it by using loops. We know it is slow. We will improve it later."""
    #jax.debug.print("x_denominator:{}", x_denominator[:, 0])
    #jax.debug.print("the shape of pos_integration_points:{}", jnp.shape(pos_integration_points))
    #jax.debug.print("pos_integration_points:{}", pos_integration_points[:, :, 0])
    """problem is larger than we are thinking. we need make a plan about how to do it. 18.09.2024."""
    jax.debug.print("pos:{}", data.positions)
    #batch_network_value_wavefunction = jax.vmap(f, in_axes=(None, 1, 1, 1), out_axes=0)
    ratio_denominator_real_part = batch_network(batchparams, data.positions, data.atoms, data.charges)
    ratio_denominator_img_part = batchphase(batchparams, data.positions, data.atoms, data.charges)
    jax.debug.print("ratio_denominator_real:{}", ratio_denominator_real_part)
    jax.debug.print("ratio_denominator_img:{}", ratio_denominator_img_part)
    ratio_denominator = ratio_denominator_img_part * jnp.exp(ratio_denominator_real_part)
    """the following part is tempoary plan."""
    ratio = []
    for i in range(len(pos_integration_points)):#batch dimension
        for j in range(len(pos_integration_points[i])): #electrons dimension
            for k in range(len(pos_integration_points[i][j])): #atoms dimension
                print("----------------")
                for m in range(len(pos_integration_points[i][j][k])):
                    #jax.debug.print("pos_integration_points:{}", pos_integration_points[i][j][k][m])
                    x_numerator = x_denominator.at[i, j].set(pos_integration_points[i][j][k][m])
                    #jax.debug.print("x_numerator:{}", x_numerator)
                    x_numerator = jnp.reshape(x_numerator, (1, batch_size, -1))
                    #jax.debug.print("x_numerator:{}", x_numerator)
                    ratio_numerator_real = batch_network(batchparams, x_numerator, data.atoms, data.charges)
                    ratio_numerator_img = batchphase(batchparams, x_numerator, data.atoms, data.charges)
                    #jax.debug.print("ratio_numerator_real:{}", ratio_numerator_real)
                    #jax.debug.print("ratio_numerator_img:{}", ratio_numerator_img)
                    ratio_numerator = ratio_numerator_img*jnp.exp(ratio_numerator_real)
                    ratio.append(ratio_numerator/ratio_denominator)
                    #ratio = jnp.array(ratio_numerator)/jnp.array(ratio_denominator)
                    #jax.debug.print("ratio:{}", ratio)


    ratio = jnp.reshape(jnp.array(ratio), (batch_size, nelectrons, natoms, 6, batch_size)) - (1+ 1.j * 0)#this line could has problems.
    jax.debug.print("ratio:{}", ratio)
    ratio = jnp.sum(jnp.sum(ratio, axis=-1), axis=-1)
    jax.debug.print("ratio:{}", ratio)
    """to be continued. It is a complex number. 18.09.2024."""


    cos_theta_OA = r_rot_coord_OA[:, :, :, :, 0]/jnp.reshape(r_ae_OA, (batch_size, nelectrons, natoms, 6))
    l_list_OA = jnp.repeat(jnp.reshape(jnp.repeat(l_list, batch_size, axis=0), (batch_size, -1, 1)), natoms*6, axis=-1)
    l_list_OA = jnp.reshape(l_list_OA, (jnp.shape(cos_theta_OA)))
    #jax.debug.print("l_list:{}", l_list)
    P_l_value_OA = (2*l_list_OA + 1) * P_l_0(cos_theta_OA) * weights[0]
    jax.debug.print("P_l_value_OA:{}", P_l_value_OA)
    value_OA = jnp.sum(P_l_value_OA, axis=-1) * ratio
    jax.debug.print("value_OA:{}", value_OA)
    P_l_value_OB = get_summation_legrend(r_ae=r_ae, Points=Points_OB, number_points=12, batch_size=4, nelectrons=4, natoms=2, weights=weights[1])
    P_l_value_OC = get_summation_legrend(r_ae=r_ae, Points=Points_OC, number_points=8, batch_size=4, nelectrons=4, natoms=2, weights=weights[2])
    P_l_value_OD = get_summation_legrend(r_ae=r_ae, Points=Points_OD, number_points=24, batch_size=4, nelectrons=4, natoms=2, weights=weights[3])
    #jax.debug.print("P_l_value_0D:{}", P_l_value_OD)
    P_l_value_total = jnp.sum(P_l_value_OA, axis=-1) + jnp.sum(P_l_value_OB, axis=-1) + jnp.sum(P_l_value_OC, axis=-1) + jnp.sum(P_l_value_OD, axis=-1)
    jax.debug.print("P_l_value_total:{}", P_l_value_total)
    return P_l_value_total




key = jax.random.PRNGKey(seed=1)
l_list = jnp.array([[0, 0, 0, 0]])
output4 = get_P_l(batch_network=batchnetwork, batch_phase=batchphase, batchparams=batchparams, data=data, r_ae=r_ae,
                  batch_size=4, nelectrons=4, natoms=2, key=key, l_list=l_list)


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