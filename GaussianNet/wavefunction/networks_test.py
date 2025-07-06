from pfapack import pfaffian as pf
import enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import attr
import chex
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
import kfac_jax


@chex.dataclass
class PfaffianNetData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


def construct_input_features(
        pos: jnp.ndarray,
        atoms: jnp.ndarray,
        ndim: int = 3 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert atoms.shape[1] == ndim
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None],axis=-1) * (1.0 - jnp.eye(n)))
    #jax.debug.print("r_ee_rest:{}", r_ee[0][1:])
    #jax.debug.print("r_ee:{}", r_ee)
    r_ee_indices = jnp.triu_indices_from(r_ee, k=1)
    #jax.debug.print("r_ee_indices:{}", r_ee_indices)
    #r_ee = r_ee[0][1:]
    r_ee = r_ee[r_ee_indices]
    n1 = r_ee.shape[0]
    jax.debug.print("n1:{}", n1)
    r_ee_input = r_ee[0: int(n1/2)]
    r_ee_triplet = r_ee_input**2 * jnp.exp((-1 * r_ee_input - r_ee_input**2))
    r_ee_singlet = r_ee_input ** 2 * jnp.exp((-1 * r_ee_input - r_ee_input ** 2))
    return ae, r_ae, r_ee_triplet, r_ee_singlet



atoms = jnp.array([[0.0, 0.0, 0.0]])
pos = jnp.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,])
ae, r_ae, r_ee_triplet, r_ee_singlet = construct_input_features(pos, atoms, 3)
jax.debug.print("ae:{}", ae)
jax.debug.print("r_ae:{}", r_ae)
jax.debug.print("r_ee_triplet:{}", r_ee_triplet)
jax.debug.print("r_ee_singlet:{}", r_ee_singlet)
