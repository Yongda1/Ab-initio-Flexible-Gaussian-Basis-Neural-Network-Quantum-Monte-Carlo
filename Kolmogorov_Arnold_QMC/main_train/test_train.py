import base_config
import train
import jax

cfg = base_config.default()
cfg.batch_size = 10000 # currently, useless input.
cfg.layer_dims = [4, 4, 4, 6]
cfg.g = [100, 100, 100,]
cfg.k = [3, 3, 3,]
cfg.grid_range = [[0, 5], [0, 5], [0, 5]]
cfg.iterations = 100
train.train(cfg)

