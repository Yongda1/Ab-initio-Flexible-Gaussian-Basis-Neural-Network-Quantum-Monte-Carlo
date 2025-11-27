import jax.numpy as jnp
import jax
import chex

"""we implement the associated Legendre polynomials in this module."""


def p00(x):
    """p_l_m"""
    return jnp.ones_like(x)

def p10(x):
    return x


def init_associated_legendre(key: chex.PRNGKey,
                             n_in: int,
                             n_out: int,
                             d: int,
                             add_residual: bool = True,
                             add_bias: bool = True,
                             external_weights: bool = True,
                             ):
    """initialize the parameters for chebyshev polynomial basis functions.
    we only consider the situation that all bool types are true.17.11.2025."""
    key_basis, key_residual, key_external_weights, key_bias = jax.random.split(key, 4)
    ext_dim = d if add_bias else d+1
    std = 1.0/jnp.sqrt(n_in * ext_dim)
    c_basis = jax.nn.initializers.truncated_normal(stddev=std,)(key_basis, (n_out, n_in, ext_dim))
    #jax.debug.print("c_basis:{}", c_basis)
    c_res = jax.nn.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(key_residual, (n_out, n_in))
    #jax.debug.print("c_res:{}", c_res)
    bias = jnp.zeros(n_out)
    #jax.debug.print("bias:{}", bias)
    c_ext = jnp.ones((n_out, n_in))
    return {'c_basis': c_basis, 'c_res': c_res, 'c_ext': c_ext, 'bias': bias, }


def generate_high_l_legendre_polynomial_m_0(input, l: jnp.ndarray):
    p_init_1, p_init_2, x = input
    p_init_3 = (2 * l + 1) * x * p_init_2 - (l + 0) * p_init_1
    return (p_init_2, p_init_3, x), p_init_3


def associated_legendre_polynomial_each_layer(x: jnp.ndarray,
                                              n_in: int,
                                              n_out: int,
                                              l: int, ):
    """l: the l in QM."""
    batch = x.shape[0]
    x = jnp.tanh(x) # this line is important to make the net stable.I think it is also import for a_legendre polynomials.
    x = jnp.expand_dims(x, axis=-1)
    jax.debug.print("x:{}",x)
    x = jnp.tile(x, (1, 1, n_out))
    #jax.debug.print("x:{}",x)
    l_range = jnp.arange(1, l)
    jax.debug.print("l_range:{}",l_range)
    p_init_1 = p00(x)
    p_init_2 = p10(x)
    output1, p_l_high_order = jax.lax.scan(f=generate_high_l_legendre_polynomial_m_0, init=(p_init_1, p_init_2, x),  xs=l_range)
    p_l_total = jnp.concatenate((p_init_1[None,...], p_init_2[None, ...], p_l_high_order), axis=0)
    jax.debug.print("p_l_total:{}",p_l_total.shape)
    p_l_total = jnp.transpose(p_l_total, (1, 0, 2, 3)) # 3 is the number of features
    jax.debug.print("p_l_total:{}",p_l_total)

    for l in l_range:
        m = jnp.arange(-l, l+1)
        jax.debug.print("m:{}",m)

    """to be continued... 27.11.2025. I dont have time to solve it."""


    '''
    p_init_1 = p00(x)
    p_init_2 = p10(x)
    """in the following lines, we generate the different l with m=0."""
    for i in l_range:
        p_temp = p_init_2
        p_init_2 = (2*i+1)*x*p_init_2 - (i+0)*p_init_1
        p_init_1 = p_temp
        jax.debug.print("i:{}",i)
        jax.debug.print("p_init_2:{}",p_init_2)
    '''

    return None


def residual(x: jnp.ndarray,):
    return x/(1+jnp.exp(-x))




def forward_each_layer(x: jnp.ndarray,
                       n_in: int,
                       n_out: int,
                       d: int,
                       c_basis: jnp.ndarray,
                       c_ext: jnp.ndarray,
                       bias: jnp.ndarray,
                       c_res: jnp.ndarray,):
    batch = x.shape[0]
    Bi = associated_legendre_polynomial_each_layer(x, n_in, n_out, d)
    '''
    act = Bi.reshape(batch, -1)
    #jax.debug.print("act:{}", act)
    act_w = c_basis * c_ext[..., None]
    act_w = act_w.reshape(n_out, -1)
    #jax.debug.print("act_w:{}", act_w)
    y = jnp.matmul(act, act_w.T)
    if c_res is not None:
        res = residual(x)
        res_w = c_res
        full_res = jnp.matmul(res, res_w.T) # (batch, n_out)
        """consider to change + to *, also change the residual function."""
        y += full_res

    if bias is not None:
       y += bias

    return y
    '''


'''this part is for debugging the chebyshev polynomial basis functions.'''
seed = 23
key = jax.random.PRNGKey(seed)
input = jnp.array([[0.1, 0.2, 0.1,], [0.2, 0.2, 0.2,]])
params = init_associated_legendre(key, 3, 4, 3,)
output = forward_each_layer(x=input, n_in=3, n_out=4, d=3,
                            c_basis = params['c_basis'],
                            c_ext = params['c_ext'],
                            bias = params['bias'],
                            c_res = params['c_res'])
jax.debug.print("output:{}", output)
