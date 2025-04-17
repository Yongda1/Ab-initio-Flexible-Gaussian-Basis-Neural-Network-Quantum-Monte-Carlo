import jax
import chex
import optax
import logging
import time
import jax.numpy as jnp
import numpy as np
import kfac_jax
from typing_extensions import Protocol
from typing import Optional, Tuple, Union
from AIQMCpretrain1 import checkpoint
from jax.experimental import multihost_utils
from AIQMCpretrain1.VMC import VMCmcstep
from AIQMCpretrain1.wavefunction_Ynlm import nn
#from AIQMCrelease3.Energy import pphamiltonian
#from AIQMCrelease3.Loss import pploss as qmc_loss_functions
from AIQMCpretrain1.Loss import loss as qmc_loss_functions
from AIQMCpretrain1 import constants
from AIQMCpretrain1.Energy import hamiltonian
from AIQMCpretrain1.Optimizer import adam
from AIQMCpretrain1.utils import writers
from AIQMCpretrain1.initial_electrons_positions.init import init_electrons
from AIQMCpretrain1.spin_indices import jastrow_indices_ee
from AIQMCpretrain1.spin_indices import spin_indices_h
import functools
#logging.basicConfig(level = logging.INFO)
from jax.config import config
config.update("jax_debug_nans", True)
config.parse_flags_with_absl()