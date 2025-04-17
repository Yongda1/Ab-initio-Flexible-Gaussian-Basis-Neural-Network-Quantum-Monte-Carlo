import emlp
from emlp.reps import T
from emlp.groups import Lorentz, SO
import numpy as np

repin = 3*T(0)
repout = 1*T(0)
group = SO(3)
model = emlp.nn.EMLP(repin, repout, group=group, num_layers=3, ch=128)
x = np.random.randn(3, repin(group).size())
y = model(x)
