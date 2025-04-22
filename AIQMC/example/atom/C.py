import sys
from absl import logging
from AIQMC.tools.utils import system
from AIQMC import base_config
from AIQMC.main_train import train

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (2, 2)  # (alpha electrons, beta electrons)
cfg.system.nelectrons = 4
cfg.single_move = False
cfg.pp_use = True
cfg.network.complex = True
cfg.system.molecule = [system.Atom('C', (0, 0, 0))]
# Set training parameters
cfg.batch_size = 100
cfg.pretrain.iterations = 10
cfg.optim.optimizer = 'adam'
cfg.network.determinants = 1
cfg.network.full_det = True
cfg.log.save_path = 'save'
train.train(cfg)