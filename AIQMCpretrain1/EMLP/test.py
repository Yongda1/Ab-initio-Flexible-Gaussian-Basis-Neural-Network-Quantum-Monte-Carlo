from emlp.reps import V,sparsify_basis
import emlp
from emlp.groups import Z,S,SO,O,O13,SO13,RubiksCube
import jax.numpy as jnp
import numpy as np

G = Z(4)
rep = V(G)
v = np.random.randn(rep.size())
g = G.sample()
print(f"{rep.rho(g)}")
print(f"{v}")
print(f"{rep.rho(g)@v:}")
basis = V(Z(5)).equivariant_basis()
print("basis", basis)