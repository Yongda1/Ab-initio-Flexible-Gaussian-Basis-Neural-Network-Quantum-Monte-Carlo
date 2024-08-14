from typing import Tuple
import chex
from AIQMC import constants
from AIQMC import hamiltonian
from AIQMC import nn
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol

"""Before we go into the loss function, we have to finish the hamiltonian first."""