import base_config
import train
import jax

cfg = base_config.default()
cfg.batch_size = 100

train.train(cfg)

