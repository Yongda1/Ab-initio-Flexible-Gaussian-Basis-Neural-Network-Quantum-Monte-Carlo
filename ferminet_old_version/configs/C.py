import sys

from absl import logging
from ferminet.utils import system
from ferminet import base_config
from ferminet import train

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (3, 3)  # (alpha electrons, beta electrons)
cfg.network.complex = True

cfg.system.molecule = [system.Atom('C', (0, 0, 0))]

# Set training parameters
cfg.batch_size = 100
cfg.pretrain.iterations = 0

train.train(cfg)