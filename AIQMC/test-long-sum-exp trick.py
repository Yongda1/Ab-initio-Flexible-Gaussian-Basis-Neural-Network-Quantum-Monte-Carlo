import functools
import itertools
from typing import MutableMapping, Optional, Sequence, Tuple
#import chex. something is wrong from this library chex.
import jax
import jax.numpy as jnp

xs = [jnp.array([[2.0+1.j * 1, 1.0], [1.0, 1.0]]), jnp.array([[3.0, 1.0], [1.0, 1.0]]), jnp.array([[1.0, 2.0], [1.0, 1.0]])]
w = jnp.array([1.0, 2.0, 3.0])

def slogdet(x):
    """computes sign and log of determinants of matrices
    different from Ferminet, we only consider the determinants which have more rows and columns."""
    sign, logabsdet = jnp.linalg.slogdet(x)
    return sign, logabsdet

def logdet_matmul(xs: Sequence[jnp.ndarray], w: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """return sum_i w_i D_i
    today, we need solve this problem."""
    #the following line has some problems. The return value of functool.reduce is a number but not a list. Fine, we
    #solve this problem later.
    print("xs", xs)
    print("w", w)
    phase_in = []
    logabsdet = []
    for x in xs:
        print("x", x)
        sign, logabs = slogdet(x)
        phase_in.append(sign)
        logabsdet.append(logabs)
    print("phase_in", phase_in)
    print("logabsdet", jnp.exp(jnp.array(logabsdet)))

    maxlogabsdet = jnp.max(jnp.array(logabsdet))
    print("maxlogabsdet", maxlogabsdet)
    det = jnp.array(phase_in) * jnp.exp(jnp.array(logabsdet) - maxlogabsdet)
    print("det", det)
    result = jnp.dot(det, w)
    print("result", result)
    phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogabsdet
    print("log_out", log_out)
    return phase_out, log_out


phase_out, log_out = logdet_matmul(xs=xs, w=w)

"""what about complex number"""
