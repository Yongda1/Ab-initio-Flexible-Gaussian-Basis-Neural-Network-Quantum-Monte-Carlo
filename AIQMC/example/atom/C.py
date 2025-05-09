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
cfg.system.electrons = (3, 3)  # (alpha electrons, beta electrons)
cfg.system.nelectrons = 6
cfg.single_move = False
cfg.pp_use = False
cfg.network.complex = True
cfg.system.molecule = [system.Atom('C', (0, 0, 0))]
# Set training parameters
cfg.batch_size = 100
cfg.optim.iterations = 201
cfg.pretrain.iterations = 100
cfg.mcmc.steps = 10
cfg.network.hidden_dims = ((32, 16), (32, 16), (32, 16), (32, 16))
"""I dont understand that why the optimization with Jastrow Factors is not stable. """
#cfg.network.jastrow = 'simple_ee'
cfg.optim.optimizer = 'adam'
cfg.network.determinants = 1
cfg.network.full_det = True
cfg.log.save_path = 'save'
train.train(cfg)
"""the vmc step is not working well. Probably is just becasue we made some mistakes. 27.4.2025.
Maybe we should go back to the single electron version. Then check how to solve it. 27.4.2025.
From, next week, we focus on the paper that we need, so the problem will solve maybe August.
 This problem is also could be from the wrong understand of the gradients of neural network. 27.4.2025."""