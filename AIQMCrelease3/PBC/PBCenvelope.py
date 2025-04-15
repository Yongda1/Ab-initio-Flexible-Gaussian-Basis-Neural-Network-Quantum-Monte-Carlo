"""here, I will create the envelope function for PBC calculation.
By following the Bloch theorem, we create the envelope function like the following line:
\psi_k(r) = e^(ik*r) * u_k(r)
u(r+R) = u(r)
To statisfy the Bloch theorem, we have to set the specific following format. To make it work, we have to introduce
ae and ee with different g vector. 
Here, we have to use the format of periodic Gaussian basis because the application of angular momentum functions.
This means that the input of the wave function must have multi rae and ee as input. Anyway, we first need k points.
"""
import itertools
from typing import Mapping, Optional, Sequence, Tuple, Union
import jax.numpy as jnp
import numpy as np
import jax


structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
supercell = jnp.array([[4, 0, 0], [0, 4, 0], [0, 0, 1]])

def make_kpoints(
    lattice: Union[np.array, jnp.array],
    supercell: jnp.array
) -> jnp.array:
  """only generate homogenous kpoints."""
  recvec = jnp.linalg.inv(lattice).T
  #jax.debug.print("rec_lattice:{}", recvec)
  def supercell_cal(supercell_inner: jnp.array, lattice_inner: jnp.array):
      return supercell_inner * lattice_inner

  supercell_cal_parallel = jax.vmap(jax.vmap(supercell_cal, in_axes=(0, None)), in_axes=(0, 0))
  supercell_real = supercell_cal_parallel(supercell, lattice)
  supercell_real = jnp.sum(supercell_real, axis=1)
  recvec_supercell = jnp.linalg.inv(supercell_real).T
  #jax.debug.print("recvec_supercell:{}", recvec_supercell)
  #jax.debug.print("supercell_real:{}", supercell_real)
  kpoints_mesh = jnp.diagonal(recvec / recvec_supercell)
  #jax.debug.print("kpoints_mesh:{}", kpoints_mesh)
  kpoints = jnp.mgrid[0:int(kpoints_mesh[0]), 0:int(kpoints_mesh[1]), 0:1].reshape(3, -1).T
  #jax.debug.print("1/kpoints_mesh:{}", 1/kpoints_mesh)
  kpoints = kpoints * (1/kpoints_mesh)
  return kpoints


output = make_kpoints(structure, supercell=supercell)
#jax.debug.print("kpoints:{}", output)
#jax.debug.print("output:{}", len(output))
"""we also need reconstruct the feature layer with g vectors. This means the input must have a set of input r_ae and r_ee 
with different g vectors. We make a test here. 15.4.2025."""

def construct_input_features(
        pos: jnp.ndarray,
        atoms: jnp.ndarray,
        g_max: int,
        lattice: jnp.array,
        ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """this means the input is batched. 15.4.2025. This means that we still need the envelope function for molecular to
    be sure that the basis will converge fast with large g_vectors. Then, we need introduce another envelope function
    for PBC."""
    assert atoms.shape[1] == ndim
    g_grid = jnp.mgrid[-g_max:g_max, -g_max:g_max, 0:1].reshape(3, -1).T
    jax.debug.print("g_grid:{}", g_grid)
    jax.debug.print("lattice:{}", lattice)

    def g_vectors_cal(g_grid_inner: jnp.array, lattice_inner: jnp.array):
        return g_grid_inner * lattice_inner

    g_vectors_cal_parallel = jax.vmap(jax.vmap(g_vectors_cal, in_axes=(0, 0)), in_axes=(0, None))
    g_vectors = g_vectors_cal_parallel(g_grid, lattice)
    g_vectors = jnp.sum(g_vectors, axis=-1)
    jax.debug.print("g_vectors:{}", g_vectors)
    jax.debug.print("g_vectors:{}", len(g_vectors))

    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    jax.debug.print("ae:{}", ae.shape)
    jax.debug.print("ee:{}", ee.shape)
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    jax.debug.print("r_ae.shape:{}", r_ae.shape)
    jax.debug.print("r_ee:{}", r_ee.shape)
    
    def r_ae_g_vectors(r_ae_inner: jnp.array, g_vectors_inner: jnp.array):
        return r_ae_inner + g_vectors_inner

    r_ae_g_vectors_parallel = jax.vmap(r_ae_g_vectors, in_axes=(None, 0))
    ae_gvectors = r_ae_g_vectors_parallel(ae, g_vectors)
    ee_gvectors = r_ae_g_vectors_parallel(ee, g_vectors)
    #jax.debug.print("r_ae_g_vectors:{}", ae_gvectors)
    jax.debug.print("r_ae_g_vectors_shape:{}", ae_gvectors.shape)
    #jax.debug.print("ee_gvectors:{}", ee_gvectors.shape)
    #jax.debug.print("ee_gvectors:{}", ee_gvectors)
    r_ae_g_vectors = jnp.linalg.norm(ae_gvectors, axis=-1, keepdims=True)
    r_ee_g_vectors = jnp.linalg.norm(ee_gvectors, axis=-1, keepdims=False)
    jax.debug.print("r_ae_g_vectors:{}", r_ae_g_vectors.shape)
    jax.debug.print("r_ee_g_vectors:{}", r_ee_g_vectors.shape)
    return ae_gvectors, ee_gvectors, r_ae_g_vectors, r_ee_g_vectors[..., None]
'''
from AIQMCrelease3.initial_electrons_positions.init import init_electrons

atoms = jnp.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
charges = jnp.array([6.0, 6.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])

natoms = 2
ndim = 3
nelectrons = 12
nspins = (6, 6)
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
pos, spins = init_electrons(subkey, structure=structure, atoms=atoms, charges=charges,
                            electrons=spins,
                            batch_size=1, init_width=0.5)
pos = jnp.reshape(pos, (-1)) # 10 * 3 = 30
ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, 2, structure)
jax.debug.print("ae:{}", ae.shape)
jax.debug.print("ee:{}", ee.shape)
jax.debug.print("r_ae:{}", r_ae.shape)
jax.debug.print("r_ae:{}", r_ae)
jax.debug.print("r_ee:{}", r_ee.shape)
'''