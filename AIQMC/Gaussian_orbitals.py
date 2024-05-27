"""This module can process ccecp basis file."""
import numpy as np
from scipy import special
import jax.numpy as jnp

basis = open("C.cc-pVDZ.nwchem", "r")
ae = np.array([1, 1, 1])
r_ae = np.array(np.sqrt(3))
#because we are not sure what ae and r_ae format are, we do not apply vmap or pmap here. Just show a case how to construct it.
#In this way, r_ae could be a jnp.ndarray but not float64.
def primitive_Gaussian_basis(ae: np.ndarray, r_ae: np.float64, exponents: np.float64, coefficients: np.float64, l: int, m: int):
    return coefficients * (r_ae**l) * np.exp(-1 * exponents * r_ae**2) * \
           special.sph_harm(l, m, np.arctan(ae[1]/ae[0]), np.arccos(ae[2]/r_ae))

coe = []
for line in basis.readlines():
    coe.append(line.split())

s_1_coe = np.array(np.array(coe[1:10]))
s_2_coe = np.array(np.array(coe[11]))
p_1_coe = np.array(np.array(coe[13:22]))
p_2_coe = np.array(np.array(coe[23]))
d_1_coe = np.array(np.array(coe[25]))

for i in s_1_coe:
    value = primitive_Gaussian_basis(ae, r_ae, float(i[0]), float(i[1]), l=0, m=0)