"""VMC"""
import jax.numpy as jnp
from typing import Optional, Tuple, Union

def main(atoms: jnp.array,
         charges: jnp.array,
         spins: jnp.array,
         tstep: float,
         nelectrons: int,
         nsteps: int,
         natoms: int,
         ndim: int,
         batch_size: int,
         iterations: int,
         save_path: Optional[str],
         restore_path: Optional[str],
         save_frequency: float,
         structure: jnp.array,):
