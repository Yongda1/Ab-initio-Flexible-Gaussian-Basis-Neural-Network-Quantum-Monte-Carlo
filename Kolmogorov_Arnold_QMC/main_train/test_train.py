import base_config
import train
import jax
from GaussianNet.tools.utils import system

cfg = base_config.default()

cfg.system.electrons = (3, 3)  # (alpha electrons, beta electrons)
cfg.system.nelectrons = 6
cfg.system.molecule = [system.Atom('C', (0, 0, 0))]

"""the optimization is not stable. But why ? is the residual connect necessary? 11.11.2025.
what does grid range really mean? can it be minus?"""
cfg.batch_size = 10
cfg.layer_dims = [4, 10, 10, 10]
cfg.g = [50, 50, 50,]
cfg.k = [5, 5, 5,]
cfg.grid_range = [[0, 5], [0, 5], [0, 5]]
cfg.envelope.g_envelope = 20
cfg.envelope.k_envelope = 5
cfg.envelope.grid_range_envelope = [-10, 10]
cfg.iterations = 100
train.train(cfg)

