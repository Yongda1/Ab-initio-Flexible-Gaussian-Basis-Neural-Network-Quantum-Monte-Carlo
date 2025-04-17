from jax import random
import jax
import numpy as np
import emlp.nn.flax as nn
from emlp.reps import T, V
from emlp.groups import SO

repin = 4*V
repout = V
G = SO(3)
x = np.random.randn(5, repin(G).size())
jax.debug.print("x:{}", x)
model = nn.EMLP(repin, repout, G)
key = random.PRNGKey(0)
emlp_params = model.init(random.PRNGKey(42), x)
y = model.apply(emlp_params, x)
jax.debug.print("emlp_params:{}", emlp_params)