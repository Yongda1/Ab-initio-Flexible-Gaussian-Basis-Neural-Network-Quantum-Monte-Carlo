import sys
from absl import logging
from GaussianNet.tools.utils import system
from GaussianNet import base_config
from GaussianNet.main_train import train_vmcstep
import jax
jax.config.update("jax_debug_nans", True)
#jax.config.update("jax_traceback_filtering", "off")
# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (3, 3)  # (alpha electrons, beta electrons)
cfg.system.nelectrons = 6

cfg.single_move = True
cfg.pp_use = False

cfg.network.complex = True
cfg.system.molecule = [system.Atom('C', (0, 0, 0))]
# Set training parameters
cfg.batch_size = 10
cfg.optim.iterations = 1001
cfg.pretrain.iterations = 10
cfg.mcmc.steps = 10
#cfg.network.hidden_dims = ((32, 16), (32, 16), (32, 16), (32, 16))
"""I dont understand that why the optimization with Jastrow Factors is not stable. """
#cfg.network.jastrow = 'simple_ee'
cfg.optim.optimizer = 'adam'
cfg.network.determinants = 1
cfg.network.full_det = True
cfg.log.save_path = 'save'
train_vmcstep.train(cfg)