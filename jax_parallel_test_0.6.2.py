import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

"""communication collectives."""
'''
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
print('FINAL RESULT:\n', y)'''

n_atoms = 72
n_up = n_atoms // 2
n_down = n_atoms - n_up

initial_magmom = [1.0] * n_up + [-1.0] * n_down
np.random.shuffle(initial_magmom)
print(initial_magmom)

for i in range(len(initial_magmom)):
    print('Fe ' + str(initial_magmom[i]))

'''
for i in range(len(initial_magmom)):
    print('Fe' + str(i+1) + ' 55.845  Fe.pbesol-spn-rrkjus_psl.1.0.0.UPF')'''