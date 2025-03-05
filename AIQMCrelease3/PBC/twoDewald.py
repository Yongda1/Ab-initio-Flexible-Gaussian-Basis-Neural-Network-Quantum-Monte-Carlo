"""create the function for two dimensional ewald summation."""
import jax
import jax.numpy as jnp
import numpy as np
import itertools
from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.wavefunction_Ynlm.nn import construct_input_features

natoms = 2
ndim = 3
nelectrons = 8
nspins = (4, 4)
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
atoms = jnp.array([[0.0, 0.0, 0.0], [2/3, 1/3, 0.0]])
charges = jnp.array([4.0, 4.0])
spins = jnp.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
lattice = jnp.array([[0.5 * jnp.sqrt(3), 0.5, 0],
                     [0.5 * jnp.sqrt(3), -0.5, 0],
                     [0, 0, 10]])

#jax.debug.print("lattice:{}", jnp.linalg.norm(lattice, axis=-1))
pos, spins = init_electrons(subkey, structure=lattice, atoms=atoms, charges=charges,
                            electrons=spins,
                            batch_size=1, init_width=0.5)
pos = jnp.reshape(pos, (-1))

ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=3)
#jax.debug.print("ae:{}", ae)
#jax.debug.print("r_ae:{}", r_ae)
recvec = jnp.linalg.inv(lattice).T
cell_area = jnp.linalg.det(lattice[:2, :2])
jax.debug.print("area:{}", lattice[:2, :2])
jax.debug.print("lattice:{}", lattice)
jax.debug.print("recvec:{}", recvec)


def lattice_displacements(lattice: jnp.array, nlatvec: int = 1):
    space = jnp.repeat(jnp.array([jnp.arange(-nlatvec, nlatvec + 1)]), 2, axis=0)
    XYZ = jnp.meshgrid(*space, indexing='ij')
    xyz = jnp.stack(XYZ, axis=-1).reshape((-1, 2))
    z_zeros = jnp.zeros((xyz.shape[0], 1))
    xyz = jnp.concatenate([xyz, z_zeros], axis=1)

    def l_d_multi(xyz_inner: jnp.array, lattice: jnp.array):
        return xyz_inner * lattice

    l_d_multi_parallel = jax.vmap(jax.vmap(l_d_multi, in_axes=(0, 0)), in_axes=(0, None))
    l_d = l_d_multi_parallel(xyz, lattice)
    return l_d


l_d = lattice_displacements(lattice)


def set_alpha(alpha_scaling: float = 5.0):
    smallest_height = jnp.amin(1/jnp.linalg.norm(recvec[:2, :2], axis=1))
    alpha = alpha_scaling / smallest_height
    return alpha


alpha = set_alpha(alpha_scaling=5.0)


def real_rij(ae: jnp.array, l_d: jnp.array, alpha: jnp.array):
    """calculate the real part."""
    ae = jnp.reshape(ae, (nelectrons, natoms, 3, 1))

    def rij_plus_m(aeinner: jnp.array, l_d_inner: jnp.array):
        return aeinner + l_d_inner

    rij_plus_m_parallel = jax.vmap(jax.vmap(jax.vmap(rij_plus_m, in_axes=(0, None)), in_axes=(0, None)), in_axes=(None, 0))
    rij_m = rij_plus_m_parallel(ae, l_d)
    rij_m = jnp.linalg.norm(jnp.linalg.norm(rij_m, axis=-1), axis=-1)
    rij_m = jax.scipy.special.erfc(alpha * rij_m) / rij_m
    return rij_m


rij_m = real_rij(ae, l_d, alpha)


def set_ewald_e_ion(rij_m: jnp.array):
    #ion_ion_charge_ij = jnp.triu(charges[None, ...] * charges[..., None], k=1)
    
    def charges_rij_m(rij_m_inner: jnp.array, charges: jnp.array):
        return rij_m_inner * (-1) * charges
    
    charges_rij_m_parallel = jax.vmap(jax.vmap(charges_rij_m, in_axes=(0, None)), in_axes=(0, None))
    e_ion_real_cross = charges_rij_m_parallel(rij_m, charges)
    return jnp.sum(e_ion_real_cross)


#output = set_ewald_e_ion(rij_m)


def generate_positive_gpoints(gmax: int):
    gXpos = jnp.mgrid[1: gmax + 1, -gmax: gmax + 1, 0:1].reshape(3, -1)
    gX0Ypos = jnp.mgrid[0:1, 1: gmax + 1, 0:1].reshape(3, -1)
    gpts = jnp.concatenate([gXpos, gX0Ypos], axis=1)
    gpoints = jnp.einsum("ji,jk->ik", gpts, recvec * 2 * jnp.pi)
    return gpoints



#gpoints = generate_positive_gpoints(gmax=200)
#jax.debug.print("gpoints:{}", gpoints)


def set_gpoints(gmax: int, tol: float = 1e-10):
    candidate_gpoints = generate_positive_gpoints(gmax)
    jax.debug.print("candidate_gpoints:{}", candidate_gpoints.shape)
    gnorm = jnp.linalg.norm(candidate_gpoints, axis=-1)
    #gweight = jnp.pi * jax.scipy.special.erfc(gnorm/(2 * alpha)) * 2
    #gweight /= cell_area * gnorm
    #jax.debug.print("gweight:{}", gweight)
    return gnorm


gnorm = set_gpoints(200)


def ewald_recip_weight(r: jnp.array):
    zij = r[:, :, 2]

    def w1_w2(zij_inner: jnp.array, gnorm_inner: jnp.array):
        return jnp.exp(zij_inner * gnorm_inner)*jax.scipy.special.erfc(alpha * zij + gnorm_inner/(2 * alpha)) +\
               jnp.exp(-1 * zij * gnorm_inner)*jax.scipy.special.erfc(-1 * alpha * zij + gnorm_inner/(2 * alpha))

    w1_w2_parallel = jax.vmap(w1_w2, in_axes=(None, 0))
    recip_weight = w1_w2_parallel(zij, gnorm)
    return recip_weight


output = ewald_recip_weight(ae)


def ewald_recip_weight_charge(r: jnp.array):
    zij = r[:, :, 2]
    w1 = zij * jax.scipy.special.erfc(alpha * zij)
    w2 = 1/(alpha * jnp.sqrt(jnp.pi)) * jnp.exp(-1 * alpha**2 * zij**2)
    return w1 + w2

w12 = ewald_recip_weight_charge(ae)
jax.debug.print("w12:{}", w12.shape)
"""to be continued... 5.3.2025."""
