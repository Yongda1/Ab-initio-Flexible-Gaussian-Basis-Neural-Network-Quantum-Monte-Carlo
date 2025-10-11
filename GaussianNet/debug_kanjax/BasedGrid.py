import jax.numpy as jnp
import jax

class BaseGrid:
    def __init__(self, n_in, n_out, k, G, grid_range: tuple = (-1, 1), grid_e: float = 0.05):
        self.n_in = n_in
        self.n_out = n_out
        self.k = k
        self.G = G
        self.grid_range = grid_range
        self.grid_e = grid_e
        # Initialize the grid, which is henceforth callable as .item
        self.item = self._initialize()

    def _initialize(self):
        h = (self.grid_range[1] - self.grid_range[0]) / self.G
        #jax.debug.print('h:{}', h)
        """here, it remains unclear. How to calculate the number of points. 23.09.2025."""
        grid = jnp.arange(-self.k, self.G+self.k+1, dtype=jnp.float32) * h + self.grid_range[0]
        #jax.debug.print('grid:{}', grid)
        grid = jnp.expand_dims(grid, 0)
        grid = jnp.tile(grid, (self.n_in * self.n_out, 1))
        jax.debug.print('grid_base_grid:{}', grid)
        return grid

    def update(self, x, G_new):
        """to be continued 26.09.2025."""
        """Let us try to finish it. 28.09.2025."""
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k -> ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in*self.n_out))
        x_ext = jnp.transpose(x_ext, (1, 0))
        """the above part is still the construction of input layer. nothing more."""
        jax.debug.print("x_ext:{}", x_ext)
        """dont understand why the input data need to be sorted."""
        x_sorted = jnp.sort(x_ext, axis=1)
        jax.debug.print("x_sorted:{}", x_sorted)
        jax.debug.print("G_new:{}", G_new)
        jax.debug.print("batch:{}", batch)
        ids = jnp.concatenate((jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1])))
        jax.debug.print("ids:{}", ids)
        grid_adaptive = x_sorted[:, ids]
        jax.debug.print("grid_adaptive:{}", grid_adaptive)

        margin = 0.01
        """calculate the length between points for new points."""
        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / G_new
        grid_uniform = (jnp.arange(G_new + 1, dtype=jnp.float32) * uniform_step[:, None] + x_sorted[:, 0][:, None] - margin)
        jax.debug.print("grid_uniform:{}", grid_uniform)
        grid = self.grid_e * grid_uniform + (1.0 -self.grid_e) * grid_adaptive
        jax.debug.print("grid:{}", grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / G_new
        jax.debug.print("h:{}", h)
        left = jnp.squeeze((jnp.arange(self.k, 0, -1) * h[:, None]), axis=1)
        right = jnp.squeeze((jnp.arange(1, self.k+1) * h[:, None]), axis=1)
        grid = jnp.concatenate([grid[:, [0]] - left, grid, grid[:, [-1]] + right], axis=1)
        #jax.debug.print('grid_update:{}', grid)
        self.item = grid
        self.G = G_new