"""convert the pseudopotential file to the proper format."""
import numpy as np
import jax.numpy as jnp

"""here, you should input the atomic symbol in the correct order."""
symbol = ['C', 'C']
number_C = symbol.count('C')
print("number_C", number_C)
atom_pp = open("./C.ccECP.nwchem", "r")
temp0 = []
for line in atom_pp.readlines():
    temp0.append(line.split())

print("temp0", temp0)
local_order1 = 0
local_order2 = 0
local_order3 = 0
local_order4 = 0
for i in range(len(temp0)):
    for j in range(len(temp0[i])):
        if temp0[i][j] == 'ul':
            print(temp0[i][j])
            local_order1 = i
        if temp0[i][j] == 'S':
            print(temp0[i][j])
            local_order2 = i
        if temp0[i][j] == 'P':
            local_order3 = i
        else:
            local_order3 = len(temp0)


lines_number = local_order2 - local_order1
local_parameters = np.array(temp0[(local_order1 + 1):local_order2])
Rn_local = np.array([np.array(local_parameters[:, 0], dtype=float)])
Rn_local = np.repeat(Rn_local, axis=0, repeats=number_C)
Local_exps = np.array([np.array(local_parameters[:, 1], dtype=float)])
Local_exps = np.repeat(Local_exps, axis=0, repeats=number_C)
Local_coes = np.array([np.array(local_parameters[:, 2], dtype=float)])
Local_coes = np.repeat(Local_coes, axis=0, repeats=number_C)

print('local_order3', local_order3)
lines_number_nonlocal = local_order3 - local_order2
"""to be continued ... I am not sure if I should spend time on this thing. 11.3.2025."""



