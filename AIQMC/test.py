"""test"""
import functools
import jax.numpy as jnp
import numpy as np


def slogdet(x):
    """computes sign and log of determinants of matrices
    different from Ferminet, we only consider the determinants which have more rows and columns."""
    sign, logabsdet = jnp.linalg.slogdet(x)
    return sign, logabsdet

xs = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]],[[3, 0], [0, 3]], [[3, 0], [0, 3]]])

for x in xs:
    print(slogdet(x))

a = [slogdet(x) for x in xs]
print(a)
phase_in, logabsdet = functools.reduce(lambda a, b: (a[0] * b[0], a[1] + b[1]), [slogdet(x) for x in xs], (1, 0))
print(phase_in)
print(logabsdet)