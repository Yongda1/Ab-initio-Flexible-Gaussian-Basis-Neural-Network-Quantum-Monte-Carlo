import jax
import jax.numpy as jnp
from flax import nnx
from typing import Union
from BasedGrid import BaseGrid
from general import solve_full_lstsq



class BaseLayer(nnx.Module):
    def __init__(self,
                 n_in : int = 2,
                 n_out: int = 5,
                 k: int = 3,
                 G: int = 3,
                 grid_range: tuple = (0, 1),
                 grid_e: float = 0.05,
                 residual: Union[nnx.Module, None] = nnx.silu,
                 external_weights: bool = True,
                 init_scheme: Union[dict, None]=None,
                 add_bias: bool = True,
                 seed: int = 42):
        """
        #\phi(x) = w_b * b(x) + w_s * spline(x) (1)
        :param n_in:
        :param n_out:
        :param k:
        :param G:
        :param grid_range:
        :param grid_e:

        :param residual: b(x) = x/(1 + e ^(-x))
        :param external_weights: w_s
        :param init_scheme:
        :param add_bias:
        :param seed:
        """
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.grid_range = grid_range
        self.residual = residual
        self.Rng = nnx.Rngs(seed)
        self.grid = BaseGrid(n_in=n_in, n_out=n_out, k=k, G=G, grid_range=grid_range, grid_e=grid_e)
        c_res, c_basis = self._initialize_params(init_scheme, seed) #c_res is a n_in * n_in matrix.18.09.2025.
        #jax.debug.print("c_res:{}", c_res)
        jax.debug.print("c_basis:{}", c_basis)
        if external_weights:
            self.c_spl = nnx.Param(
                nnx.initializers.ones(
                    self.Rng.params(), (self.n_out, self.n_in), jnp.float32)
            )
            jax.debug.print("c_spl:{}", self.c_spl)
        else:
            self.c_spl = None

        self.c_basis = nnx.Param(c_basis)
        self.c_res = nnx.Param(c_res)

    def basis(self, x):
        jax.debug.print("x:{}", x)
        batch = x.shape[0]
        jax.debug.print("batch:{}", batch)
        # Extend to shape (batch, n_in*n_out)
        """x is batched input. For example x is [[1, 1, 1], [2, 2, 2]], But the information need to be transferred into each edge."""
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out, )).reshape((batch, self.n_in * self.n_out))
        #jax.debug.print("x_ext:{}", x_ext)
        # Transpose to shape (n_in*n_out, batch)
        x_ext = jnp.transpose(x_ext, (1, 0))
        # Broadcasting for vectorized operations
        #jax.debug.print("x_ext:{}", x_ext)
        """grid的最外层是12个，因为我们由12条edge"""
        grid = jnp.expand_dims(self.grid.item, axis=2)
        """here, the input x will be transferred into each edge. 每一列是输入节点的vector， Thus, in our example, 
        the matrix has two cols, i.e., two batches, but 12 rows, corresponding to two edges."""
        x = jnp.expand_dims(x_ext, axis=1)
        #jax.debug.print("grid:{}", grid)
        jax.debug.print("x:{}", x)
        # k = 0 case
        #temp1 = grid[:, :-1]
        #temp2 = grid[:, 1:]
        #jax.debug.print("temp1:{}", temp1)
        #jax.debug.print("temp2:{}", temp2)
        #temp3 = (x >= grid[:, :-1]).astype(float)
        #jax.debug.print("temp3:{}", temp3)

        """the following part is the construction of basis_spline functions.26.09.2025
        first, we need get the position of input coordinates. """
        basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
        #jax.debug.print("basis_splines:{}", basis_splines)

        for K in range(1, self.k+1):
            #jax.debug.print("K:{}", K)
            #temp4 = grid[:, :-(K+1)]
            #jax.debug.print("temp4:{}", temp4)
            #temp5 = grid[:, K:-1]
            #jax.debug.print("temp5:{}", temp5)
            """construct the basis functions,
            B_i,p (x) = (x - grid_i)/(grid_i+p - grid_i) * B_i,p-1(x) + (grid_i+p+1 - x)/(grid_i+p+1 - grid_i+1) * B_i+1,p-1(t)
            exactly this equation.
            """
            left_term = (x - grid[:, :-(K+1)]) / (grid[:, K:-1] - grid[:, :-(K+1)])
            right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
            basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

        jax.debug.print("basis_splines:{}", basis_splines)
        return basis_splines



    def _initialize_params(self, init_scheme, seed):
        """c_res is the w_b in equation."""
        c_res = nnx.initializers.glorot_uniform(in_axis=-1, out_axis=-2)(self.Rng.params(), (self.n_out, self.n_in), jnp.float32)
        #jax.debug.print("c_res:{}", c_res)
        """notice that the number of c_basis on one edge is 6 now. Because K=3 and G=3, the number of basis function is 6. 
        Thus, the number of the coe of basis function must be 6. The total number is 12, is n_in * n_out. 24.09.2025."""
        c_basis = nnx.initializers.normal(stddev=0.1)(self.Rng.params(), (self.n_in * self.n_out, self.grid.G + self.k), jnp.float32)
        #jax.debug.print("c_basis:{}", c_basis)
        return c_res, c_basis

    def update_grid(self, x, G_new):
        Bi = self.basis(x)
        ci = self.c_basis.value
        #jax.debug.print("Bi:{}", Bi)
        #jax.debug.print("ci:{}", ci)
        ciBi = jnp.einsum('ij, ijk -> ik', ci, Bi)
        """to be continued for the update function. 26.09.2025"""
        self.grid.update(x, G_new)
        A = self.basis(x)
        Bj = jnp.transpose(A, (0, 2, 1))
        ciBi = jnp.expand_dims(ciBi, axis=-1)
        cj = solve_full_lstsq(Bj, ciBi)
        cj = jnp.squeeze(cj, axis=-1)
        self.c_basis = nnx.Param(cj)
        #jax.debug.print("c_basis_value:{}", self.c_basis.value)

    def __call__(self, x):
        """The layer's forward pass."""
        batch = x.shape[0]

        Bi = self.basis(x)
        ci = self.c_basis.value
        spl = jnp.einsum('ij,ijk->ik', ci, Bi)
        spl = jnp.transpose(spl, (1, 0))


        jax.debug.print("c_basis_value:{}", self.c_basis.value)
        jax.debug.print("c_spl:{}", self.c_spl)



    






x_batch = jnp.array([[1.5, 1.5, 1.6],
                     [1, 1, 1]])

input = jnp.array([[0.1, 0.1, 0.1],
               [0.2, 0.2, 0.2],
               [0.3, 0.3, 0.3],
               [0.4, 0.4, 0.4],
               [0.5, 0.5, 0.5],
               [0.6, 0.6, 0.6]])

layer = BaseLayer(n_in = 3, n_out = 4, k=3, G=3,external_weights=True)
outut = layer.update_grid(x=input, G_new=5)