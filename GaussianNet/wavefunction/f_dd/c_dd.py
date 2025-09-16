import jax
import chex
import jax.numpy as jnp
import math
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from GaussianNet.wavefunction import network_blocks

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]


def coefficients_layer(hidden_dims_coe):
    """Here, as said in f_uu, we suppose that coefficient c is a function of two electrons position c = f(r1, r2).
    we can construct the simple neural network as the dense layer.
    The input vector is a one dimensional array which contains r1 and r2, i.e., the coordinates of electrons.
    """

    def init(key: chex.PRNGKey) -> Tuple[int, ParamTree]:
        params = {}
        key, coe_key = jax.random.split(key, num=2)
        dims_one_in = 3 #we always have two input variables.
        layers = []
        #jax.debug.print("hidden_dims_coe:{}", hidden_dims_coe)
        for i in range(len(hidden_dims_coe)):
            layer_params = {}
            dims_one_out= hidden_dims_coe[i]
            #jax.debug.print("dims_one_in:{}", dims_one_in)
            #jax.debug.print("dims_one_out:{}", dims_one_out)
            layer_params['coe'] = network_blocks.init_linear_layer(
                coe_key,
                in_dim=dims_one_in,
                out_dim=dims_one_out,
                include_bias=True,
            )
            layers.append(layer_params)
            dims_one_in = dims_one_out
        output_dims = dims_one_in
        params['coe_layers'] = layers
        return output_dims, params

    def apply_layer(params: Mapping[str, ParamTree],
                    coe_in: jnp.ndarray):
        coe_in_next = jnp.tanh(network_blocks.linear_layer(coe_in, **params['coe']))
        return coe_in_next

    def apply(params: Mapping[str, ParamTree],
              coe_in: jnp.ndarray):
        for i in range(len(hidden_dims_coe)):
            coe_in = apply_layer(params['coe_layers'][i], coe_in,)
        coe_to_pf_orbitals = coe_in
        return coe_to_pf_orbitals

    return init, apply

'''
seed = 23
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
n=3 # n is the number of spin-up electrons.
number_coe = int(math.factorial(n) / (math.factorial(2) * math.factorial(n-2)))
jax.debug.print("number_coe:{}", number_coe)
hidden_dims_coe = (16, 16, 16, number_coe)
c_uu_init, c_uu_apply = coefficients_layer(hidden_dims_coe=hidden_dims_coe)
params = {}
n_uu_dims, params['c_uu'] = c_uu_init(subkey)
coe_in = jnp.array([0, 0.1, 0.1])
jax.debug.print("n_uu_dims:{}", n_uu_dims)
jax.debug.print("params['c_uu']:{}", params['c_uu'])
output_uu = c_uu_apply(params['c_uu'], coe_in)
jax.debug.print("output_uu:{}", output_uu)'''