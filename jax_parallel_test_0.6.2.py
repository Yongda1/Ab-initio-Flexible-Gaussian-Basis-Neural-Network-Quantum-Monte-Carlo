import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

"""communication collectives."""

devices = jax.devices()[:4]
print(devices)

mesh1d = Mesh(jax.devices()[:4], ('i',))
print(mesh1d)


mesh2d = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('i', 'j'))

@partial(jax.shard_map, mesh=mesh2d, in_specs=P('i', 'j'), out_specs=P(None, 'j'))
def f2(x_block):
  print('BEFORE:\n', x_block)
  y_block = jax.lax.psum(x_block, 'i')
  print('AFTER:\n', y_block)
  return y_block

x = jnp.arange(16).reshape(4, 4)
print('x', x)
y = f2(x)
print('FINAL RESULT:\n', y)