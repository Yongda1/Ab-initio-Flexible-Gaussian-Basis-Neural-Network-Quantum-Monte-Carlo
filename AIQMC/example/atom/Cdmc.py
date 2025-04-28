import sys
import jax.numpy as jnp
from absl import logging
from AIQMC.tools.utils import system
from AIQMC import base_config
from AIQMC.DMC.main_dmc import main as train

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (3, 3)  # (alpha electrons, beta electrons)
cfg.system.nelectrons = 6
cfg.single_move = False
cfg.pp_use = False
cfg.network.complex = True
cfg.system.molecule = [system.Atom('C', (0, 0, 0))]
# Set training parameters
cfg.batch_size = 100
cfg.optim.iterations = 1000
cfg.pretrain.iterations = 10
cfg.mcmc.steps = 20
cfg.network.hidden_dims = ((32, 16), (32, 16), (32, 16), (32, 16))
"""I dont understand that why the optimization with Jastrow Factors is not stable. """
#cfg.network.jastrow = 'simple_ee'
cfg.optim.optimizer = 'adam'
cfg.network.determinants = 1
cfg.network.full_det = True
cfg.log.save_path = 'save'

atoms = jnp.array([[0.0, 0.0, 0.0]])
charges = jnp.array([6.0])

train(cfg,
      charges=charges,
      nelectrons=6,
      natoms=1,
      ndim=3,
      batch_size=100,
      iterations=20,
      tstep=0.01,
      nspins=(3, 3),
      nsteps=10,
      nblocks=1,
      feedback=1.0,
      save_path='save',
      restore_path='restore_DMC')