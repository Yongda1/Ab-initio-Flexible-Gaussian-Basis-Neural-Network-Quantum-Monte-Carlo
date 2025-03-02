"""this module returns the electron indices for jastrow calculation."""
import jax
import jax.numpy as jnp
import numpy as np


def jastrow_indices_ee(spins: jnp.array, nelectrons: int):
    temp = jnp.reshape(spins, (nelectrons, 1))
    spins = jnp.reshape(spins, (1, nelectrons))
    spins_total = spins * temp
    spins_total_uptriangle = jnp.triu(spins_total, k=1)
    sample = jnp.zeros_like(a=spins_total_uptriangle)
    parallel = jnp.where(spins_total_uptriangle > sample, spins_total_uptriangle, sample)
    antiparallel = jnp.where(spins_total_uptriangle < sample, spins_total_uptriangle, sample)
    parallel_indices = jnp.nonzero(parallel)
    antiparallel_indices = jnp.nonzero(antiparallel)
    parallel_indices = jnp.array(parallel_indices)
    antiparallel_indices = jnp.array(antiparallel_indices)
    n_parallel = len(parallel_indices[0])
    n_antiparallel = len(antiparallel_indices[0])
    return parallel_indices, antiparallel_indices, n_parallel, n_antiparallel


def jastrow_indices_ae(charges_jastrow: jnp.array, natoms: int):
    charges_jastrow = np.array(charges_jastrow)
    charges_indices_jastrow = np.arange(natoms)
    atom_jastrow_indices = []
    charged_jastrow_needed = []
    for i in range(len(charges_indices_jastrow)):
        temp = np.repeat(charges_indices_jastrow[i], charges_jastrow[i])
        temp1 = np.repeat(charges_jastrow[i], charges_jastrow[i])
        atom_jastrow_indices.append(temp)
        charged_jastrow_needed.append(temp1)

    atom_jastrow_indices = np.hstack(np.array(atom_jastrow_indices))
    charged_jastrow_needed = np.hstack(np.array(charged_jastrow_needed))
    return atom_jastrow_indices, charged_jastrow_needed


#spins = jnp.array([1, -1, 1, -1, 1, -1, 1, -1.0])
#parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins, nelectrons=8)
#jax.debug.print("parallel_indices:{}", parallel_indices)
#jax.debug.print("antiparallel_indices:{}", antiparallel_indices)