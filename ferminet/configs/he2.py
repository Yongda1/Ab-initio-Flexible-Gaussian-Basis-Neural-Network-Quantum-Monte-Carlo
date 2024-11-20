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
cfg.system.electrons = (2, 2)  # (alpha electrons, beta electrons)
cfg.network.complex = True
cfg.system.molecule = [system.Atom('He', (0, 0, -1)), system.Atom('He', (0, 0, 1))]

# Set training parameters
cfg.batch_size = 4
cfg.pretrain.iterations = 0

train.train(cfg)