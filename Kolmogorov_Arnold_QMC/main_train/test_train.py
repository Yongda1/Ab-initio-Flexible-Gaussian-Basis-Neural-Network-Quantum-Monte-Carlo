import base_config
import train
import jax
from GaussianNet.tools.utils import system
jax.config.update("jax_traceback_filtering", "off")
cfg = base_config.default()

cfg.system.electrons = (3, 3)  # (alpha electrons, beta electrons)
cfg.system.nelectrons = 6
cfg.system.molecule = [system.Atom('C', (0, 0, 0))]

"""the optimization is not stable. But why ? Is the residual connect necessary? 11.11.2025.
what does grid range really mean? can it be minus?"""
cfg.batch_size = 10
cfg.layer_dims = [4, 20, 20, 20, 20]
cfg.g = [10, 10, 10, 10]
cfg.k = [3, 3, 3, 3]
cfg.grid_range = [[0, 2], [0, 2], [0, 2], [0, 2]]
cfg.envelope.g_envelope = 10
cfg.envelope.k_envelope = 5
cfg.envelope.grid_range_envelope = [0, 2]
cfg.iterations = 100
cfg.preiterations = 2000
cfg.chebyshev = True
cfg.spline = False
cfg.add_bias = True
cfg.external_weights = True
cfg.add_residual = True
cfg.envelope_chebyshev = False
cfg.envelope_spline = False
cfg.envelope_simple = True
train.train(cfg)

