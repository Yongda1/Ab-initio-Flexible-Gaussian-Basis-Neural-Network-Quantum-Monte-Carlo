"""create the function for two dimensional ewald summation."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
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

def generate_positive_gpoints(gmax: int):
    gXpos = jnp.mgrid[1: gmax + 1, -gmax: gmax + 1, 0:1].reshape(3, -1)
    gX0Ypos = jnp.mgrid[0:1, 1: gmax + 1, 0:1].reshape(3, -1)
    gpts = jnp.concatenate([gXpos, gX0Ypos], axis=1)
    #jax.debug.print("gpts:{}", gpts)
    gpoints = jnp.einsum("ji,jk->ik", gpts, recvec * 2 * jnp.pi)
    return gpoints



gpoints = generate_positive_gpoints(gmax=5)
#jax.debug.print("gpoints:{}", gpoints)


def ewald_recip_weight(r: jnp.array):
    zij = r[:, :, 2]

    def w1_w2(zij_inner: jnp.array, gnorm_inner: jnp.array):
        """here,we got a problem about overflows."""
        w1 = jnp.exp(zij_inner * gnorm_inner) * jax.scipy.special.erfc(alpha * zij_inner + gnorm_inner/(2 * alpha))
        w2 = jnp.exp(-1 * zij_inner * gnorm_inner)*jax.scipy.special.erfc(-1 * alpha * zij_inner + gnorm_inner/(2 * alpha))
        return w1 + w2

    w1_w2_parallel = jax.vmap(w1_w2, in_axes=(None, 0))
    recip_weight = w1_w2_parallel(zij, gnorm)
    return recip_weight


#recip_weights = ewald_recip_weight(ae)
#jax.debug.print("recip_weights:{}", recip_weights.shape)


def ewald_recip_weight_charge(r: jnp.array):
    zij = r[:, :, 2]
    w1 = zij * jax.scipy.special.erf(alpha * zij)
    w2 = 1/(alpha * jnp.sqrt(jnp.pi)) * jnp.exp(-1 * alpha**2 * zij**2)
    return (w1 + w2) * jnp.pi/cell_area

#w12 = ewald_recip_weight_charge(ae)
#jax.debug.print("w12:{}", w12.shape)
"""to be continued... 5.3.2025."""


def ewald_self(sum_charge_squared: jnp.array):
    sqrt_pi = jnp.sqrt(jnp.pi)
    real_part = -1 * alpha / sqrt_pi
    #recip_part =
    charge_part = -sqrt_pi / (cell_area * alpha)
    return (real_part + charge_part) * sum_charge_squared



def set_gpoints(gmax: int, tol: float = 1e-10):
    candidate_gpoints = generate_positive_gpoints(gmax)
    #jax.debug.print("candidate_gpoints:{}", candidate_gpoints)
    gnorm = jnp.linalg.norm(candidate_gpoints, axis=-1)
    #gweight = jnp.pi * jax.scipy.special.erfc(gnorm/(2 * alpha)) * 2
    #gweight /= cell_area * gnorm
    #jax.debug.print("gweight:{}", gweight)
    return gnorm


gnorm = set_gpoints(5)

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
    
    def charges_rij_m(rij_m_inner: jnp.array, charges: jnp.array):
        return rij_m_inner * (-1) * charges
    
    charges_rij_m_parallel = jax.vmap(jax.vmap(charges_rij_m, in_axes=(0, None)), in_axes=(0, None))
    e_ion_real_cross = charges_rij_m_parallel(rij_m, charges)

    def h_rij(gpoints_inner: jnp.array, r_inner: jnp.array):
        return jnp.dot(gpoints_inner, r_inner)

    h_rij_parallel = jax.vmap(jax.vmap(jax.vmap(h_rij, in_axes=(None, 0)), in_axes=(None, 0)), in_axes=(0, None))
    g_dot_r = h_rij_parallel(gpoints, ae)
    g_weight = ewald_recip_weight(ae)
    temp = jnp.cos(g_dot_r)
    
    def devided(g_dot_r_inner: jnp.array, gnorm_inner: jnp.array):
        return g_dot_r_inner / gnorm_inner

    devided_parallel = jax.vmap(devided, in_axes=(0, 0))
    e_ion_recip = devided_parallel(temp, gnorm) * g_weight * (jnp.pi / cell_area)

    def multiply_charges(e_ion_recip_inner: jnp.array, charges_inner: jnp.array):
        return e_ion_recip_inner * -2 * charges_inner

    multiply_charges_parallel = jax.vmap(jax.vmap(multiply_charges, in_axes=(0, None)), in_axes=(0, None))
    e_ion_recip = multiply_charges_parallel(e_ion_recip, charges)

    weight = ewald_recip_weight_charge(ae)

    def multiply_charges_c(weight_inner: jnp.array, charges_inner: jnp.array):
        return weight_inner * -2 * charges_inner

    multiply_charges_c_parallel = jax.vmap(multiply_charges_c, in_axes=(0, None))
    e_ion_charge = multiply_charges_c_parallel(weight, charges)
    #jax.debug.print("e_ion_reall_cross:{}", e_ion_real_cross)
    #jax.debug.print("e_ion_recip:{}", e_ion_recip)
    #jax.debug.print("e_ion_charge:{}", e_ion_charge)
    """here, we have one problem about sign to be checked."""
    return jnp.sum(e_ion_real_cross) + jnp.sum(e_ion_recip) - jnp.sum(e_ion_charge)


energy_e_ion = set_ewald_e_ion(rij_m)
jax.debug.print("energy_e_ion:{}", energy_e_ion)


def real_rij_ee(ee: jnp.array, l_d: jnp.array, alpha: jnp.array):
    """calculate the real part."""

    def rij_plus_m_ee(ee_inner: jnp.array, l_d_inner: jnp.array):
        #jax.debug.print("ee_inner:{}", ee_inner)
        #jax.debug.print("l_d_inner:{}", l_d_inner)
        #jax.debug.print("ee_inner+l_d_inner:{}", ee_inner + l_d_inner)
        return ee_inner + l_d_inner

    rij_plus_m_ee_parallel = jax.vmap(rij_plus_m_ee, in_axes=(0, None))
    rij_ee_m = rij_plus_m_ee_parallel(ee, l_d)
    rij_ee_m = jnp.linalg.norm(jnp.linalg.norm(rij_ee_m, axis=-1), axis=-1)
    rij_ee_m = jax.scipy.special.erfc(alpha * rij_ee_m) / rij_ee_m
    return rij_ee_m

def ewald_recip_ee_weight(r: jnp.array):
    zij = r[:, 2]
    #jax.debug.print("zij:{}", zij)

    def w1_w2_ee(zij_inner: jnp.array, gnorm_inner: jnp.array):
        """here,we got a problem about overflows."""
        w1 = jnp.exp(zij_inner * gnorm_inner) * jax.scipy.special.erfc(alpha * zij_inner + gnorm_inner/(2 * alpha))
        w2 = jnp.exp(-1 * zij_inner * gnorm_inner)*jax.scipy.special.erfc(-1 * alpha * zij_inner + gnorm_inner/(2 * alpha))
        return w1 + w2

    w1_w2_parallel = jax.vmap(w1_w2_ee, in_axes=(None, 0))
    recip_weight = w1_w2_parallel(zij, gnorm)
    return recip_weight


def ewald_recip_ee_weight_charge(r: jnp.array):
    zij = r[:, 2]
    w1 = zij * jax.scipy.special.erfc(alpha * zij)
    w2 = 1/(alpha * jnp.sqrt(jnp.pi)) * jnp.exp(-1 * alpha**2 * zij**2)
    return (w1 + w2) * jnp.pi/cell_area

def set_ewald_e_e():
    """calculate the ewald summation for electrons-electrons interaction."""
    up_triangle_indices = jnp.triu_indices_from(r_ee[..., 0], k=1)
    ee_temp = ee[up_triangle_indices]
    ee_temp = jnp.reshape(ee_temp, (-1, 3, 1))
    #jax.debug.print("ee_temp:{}", ee_temp)
    rij_m_ee = real_rij_ee(ee_temp, l_d, alpha)
    e_e_real_cross = 1 * rij_m_ee

    def h_rij_ee(gpoints_inner: jnp.array, r_inner: jnp.array):
        return jnp.dot(gpoints_inner, r_inner)

    h_rij_parallel = jax.vmap(jax.vmap(h_rij_ee, in_axes=(None, 0)), in_axes=(0, None))
    ee_temp_2 = jnp.reshape(ee_temp, (-1, 3))
    g_dot_r = h_rij_parallel(gpoints, ee_temp_2)
    g_dot_r = jnp.cos(g_dot_r)
    #jax.debug.print("g_dot_r:{}", g_dot_r.shape)
    g_ee_weight = ewald_recip_ee_weight(ee_temp_2)

    def devided(g_dot_r_inner: jnp.array, gnorm_inner: jnp.array):
        return g_dot_r_inner / gnorm_inner

    devided_parallel = jax.vmap(devided, in_axes=(0, 0))
    #jax.debug.print("g_ee_weight:{}", g_ee_weight.shape)
    e_e_recip = 2 * devided_parallel(g_dot_r, gnorm) * g_ee_weight * (jnp.pi / cell_area)
    weight_ee = ewald_recip_ee_weight_charge(ee_temp_2)
    e_e_charge = 2 * weight_ee
    return jnp.sum(e_e_real_cross) + jnp.sum(e_e_recip) + jnp.sum(e_e_charge)


output = set_ewald_e_e()
jax.debug.print("output:{}", output)



"""we need test the many ions example."""
def set_ewald_ion_ion():
    """calculate the ewald summation for ion-ion interactions."""
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    r_aa_indices = jnp.triu_indices_from(r_aa, k=1)
    aa = atoms[None, ...] - atoms[:, None]
    aa = aa[r_aa_indices]
    jax.debug.print("aa:{}", aa)
    ion_charges = charges[None, ...] * charges[..., None]
    ion_charges_indices = jnp.triu_indices_from(charges[None, ...] * charges[..., None], k=1)
    ion_charges = ion_charges[ion_charges_indices]
    jax.debug.print("ion_charges:{}", ion_charges)
    ion_ion_temp = jnp.reshape(aa, (-1, 3, 1))
    rij_m_ion_ion = real_rij_ee(ion_ion_temp, l_d, alpha)

    def charges_rij_m(rij_m_inner: jnp.array, charges: jnp.array):
        return rij_m_inner * charges

    charges_rij_m_parallel = jax.vmap(charges_rij_m, in_axes=(0, None))
    ion_ion_real_cross = charges_rij_m_parallel(rij_m_ion_ion, ion_charges)
    jax.debug.print("ion_ion_real_cross:{}", ion_ion_real_cross)

    def h_rij_ion_ion(gpoints_inner: jnp.array, r_inner: jnp.array):
        return jnp.dot(gpoints_inner, r_inner)

    h_rij_parallel = jax.vmap(jax.vmap(h_rij_ion_ion, in_axes=(None, 0)), in_axes=(0, None))
    ion_ion_temp_2 = jnp.reshape(ion_ion_temp, (-1, 3))
    jax.debug.print("ion_ion_temp2:{}", ion_ion_temp_2)
    g_dot_r = h_rij_parallel(gpoints, ion_ion_temp_2)
    g_dot_r = jnp.cos(g_dot_r)
    jax.debug.print("g_dot_r:{}", g_dot_r.shape)
    g_ion_ion_weight = ewald_recip_ee_weight(ion_ion_temp_2)
    #jax.debug.print("g_ion_ion_weight:{}", g_ion_ion_weight)

    def devided(g_dot_r_inner: jnp.array, gnorm_inner: jnp.array):
        return g_dot_r_inner / gnorm_inner

    devided_parallel = jax.vmap(devided, in_axes=(0, 0))
    ion_ion_recip = devided_parallel(g_dot_r, gnorm) * g_ion_ion_weight * (jnp.pi / cell_area)
    #jax.debug.print("ion_ion_recip:{}", ion_ion_recip)

    def multiply_charges(ion_ion_recip_inner: jnp.array, charges_inner: jnp.array):
        return ion_ion_recip_inner * charges_inner

    multiply_charges_parallel = jax.vmap(multiply_charges, in_axes=(0, None))
    ion_ion_recip = 2 * multiply_charges_parallel(ion_ion_recip, ion_charges)

    weight_ion_ion = ewald_recip_ee_weight_charge(ion_ion_temp_2)
    ion_ion_charges = 2 * weight_ion_ion * ion_charges

    return jnp.sum(ion_ion_real_cross) + jnp.sum(ion_ion_recip) - jnp.sum(ion_ion_charges)



test_output = set_ewald_ion_ion()
jax.debug.print("test_output:{}", test_output)

