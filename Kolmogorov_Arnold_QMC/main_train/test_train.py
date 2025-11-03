import base_config
import train
import jax

cfg = base_config.default()
cfg.batch_size = 100
cfg.layer_dims = [4, 4, 4, 6]
cfg.g = [3, 3, 3,]
cfg.k = [3, 3, 3,]
cfg.grid_range = [[0, 1], [0, 1], [0, 1]]
cfg.iterations = 100
train.train(cfg)

