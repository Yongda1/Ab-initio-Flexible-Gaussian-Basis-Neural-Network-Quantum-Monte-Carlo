import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from AIQMCrelease3.initial_electrons_positions.init import init_electrons
from AIQMCrelease3.wavefunction_Ynlm.nn import construct_input_features


def set_ewald_sum(natoms: int,
                  ndim: int,
                  nelectrons: int,
                  npsins,
                  lattice: jnp.array,
                  charges: jnp.array,
                  alpha_scaling: float = 5.0,
                  gmax: int = 5,
                  ):
    """2D ewald summation includes electron-electron, electrons-ion, ion-ion interactions. and each one interaction has
    three parts, real_cross part, recip_part, charges_part."""
    recvec = jnp.linalg.inv(lattice).T
    cell_area = jnp.linalg.det(lattice[:2, :2])

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

    def set_alpha(alpha_scaling: float = 5.0):
        smallest_height = jnp.amin(1 / jnp.linalg.norm(recvec[:2, :2], axis=1))
        alpha = alpha_scaling / smallest_height
        return alpha

    def generate_positive_gpoints(gmax: int):
        gXpos = jnp.mgrid[1: gmax + 1, -gmax: gmax + 1, 0:1].reshape(3, -1)
        gX0Ypos = jnp.mgrid[0:1, 1: gmax + 1, 0:1].reshape(3, -1)
        gpts = jnp.concatenate([gXpos, gX0Ypos], axis=1)
        gpoints = jnp.einsum("ji,jk->ik", gpts, recvec * 2 * jnp.pi)
        return gpoints

    def set_gpoints(gmax: int, tol: float = 1e-10):
        candidate_gpoints = generate_positive_gpoints(gmax)
        # jax.debug.print("candidate_gpoints:{}", candidate_gpoints)
        gnorm = jnp.linalg.norm(candidate_gpoints, axis=-1)
        # gweight = jnp.pi * jax.scipy.special.erfc(gnorm/(2 * alpha)) * 2
        # gweight /= cell_area * gnorm
        # jax.debug.print("gweight:{}", gweight)
        return gnorm

    def real_rij(ae: jnp.array, l_d: jnp.array, alpha: jnp.array):

        ae = jnp.reshape(ae, (nelectrons, natoms, 3, 1))

        def rij_plus_m(aeinner: jnp.array, l_d_inner: jnp.array):
            return aeinner + l_d_inner

        rij_plus_m_parallel = jax.vmap(jax.vmap(jax.vmap(rij_plus_m, in_axes=(0, None)), in_axes=(0, None)),
                                       in_axes=(None, 0))
        rij_m = rij_plus_m_parallel(ae, l_d)
        rij_m = jnp.linalg.norm(jnp.linalg.norm(rij_m, axis=-1), axis=-1)
        rij_m = jax.scipy.special.erfc(alpha * rij_m) / rij_m
        return rij_m

    def ewald_recip_weight(r: jnp.array, alpha: jnp.array, gnorm: jnp.array):
        zij = r[:, :, 2]

        def w1_w2(zij_inner: jnp.array, gnorm_inner: jnp.array):
            """here,we got a problem about overflows."""
            w1 = jnp.exp(zij_inner * gnorm_inner) * jax.scipy.special.erfc(
                alpha * zij_inner + gnorm_inner / (2 * alpha))
            w2 = jnp.exp(-1 * zij_inner * gnorm_inner) * jax.scipy.special.erfc(
                -1 * alpha * zij_inner + gnorm_inner / (2 * alpha))
            return w1 + w2

        w1_w2_parallel = jax.vmap(w1_w2, in_axes=(None, 0))
        recip_weight = w1_w2_parallel(zij, gnorm)
        return recip_weight

    # recip_weights = ewald_recip_weight(ae)
    # jax.debug.print("recip_weights:{}", recip_weights.shape)

    def ewald_recip_weight_charge(r: jnp.array, alpha: jnp.array):
        zij = r[:, :, 2]
        w1 = zij * jax.scipy.special.erf(alpha * zij)
        w2 = 1 / (alpha * jnp.sqrt(jnp.pi)) * jnp.exp(-1 * alpha ** 2 * zij ** 2)
        return (w1 + w2) * jnp.pi / cell_area

    def ewald_sum(ae: jnp.array, ee: jnp.array,):
        l_d = lattice_displacements(lattice)
        alpha = set_alpha(alpha_scaling=alpha_scaling)
        gpoints = generate_positive_gpoints(gmax=gmax)
        gnorm = set_gpoints(gmax=gmax)
        rij_m = real_rij(ae, l_d, alpha)
        """currently, we dont move this inner function somewhere for debugging easily."""
        def charges_rij_m(rij_m_inner: jnp.array, charges_inner: jnp.array):
            return rij_m_inner * (-1) * charges_inner

        charges_rij_m_parallel = jax.vmap(jax.vmap(charges_rij_m, in_axes=(0, None)), in_axes=(0, None))
        e_ion_real_cross = charges_rij_m_parallel(rij_m, charges)

        def h_rij(gpoints_inner: jnp.array, r_inner: jnp.array):
            return jnp.dot(gpoints_inner, r_inner)

        h_rij_parallel = jax.vmap(jax.vmap(jax.vmap(h_rij, in_axes=(None, 0)), in_axes=(None, 0)), in_axes=(0, None))
        g_dot_r = h_rij_parallel(gpoints, ae)
        g_weight = ewald_recip_weight(ae, alpha, gnorm)
        temp = jnp.cos(g_dot_r)

        def devided(g_dot_r_inner: jnp.array, gnorm_inner: jnp.array):
            return g_dot_r_inner / gnorm_inner

        devided_parallel = jax.vmap(devided, in_axes=(0, 0))
        e_ion_recip = devided_parallel(temp, gnorm) * g_weight * (jnp.pi / cell_area)

        def multiply_charges(e_ion_recip_inner: jnp.array, charges_inner: jnp.array):
            return e_ion_recip_inner * -2 * charges_inner

        multiply_charges_parallel = jax.vmap(jax.vmap(multiply_charges, in_axes=(0, None)), in_axes=(0, None))
        e_ion_recip = multiply_charges_parallel(e_ion_recip, charges)

        weight = ewald_recip_weight_charge(ae, alpha)

        def multiply_charges_c(weight_inner: jnp.array, charges_inner: jnp.array):
            return weight_inner * -2 * charges_inner

        multiply_charges_c_parallel = jax.vmap(multiply_charges_c, in_axes=(0, None))
        e_ion_charge = multiply_charges_c_parallel(weight, charges)
        energy_e_ion = jnp.sum(e_ion_real_cross) + jnp.sum(e_ion_recip) - jnp.sum(e_ion_charge)
        """to be continued..."""




        return energy_e_ion

    return ewald_sum