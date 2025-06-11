"""KAN need a new version of JAX above 0.6.0 version."""
from jaxkan.KAN import KAN
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def f(x, y):
    return x**2 + 2*jnp.exp(y)


def generate_data(minval=-1, maxval=1, num_samples=1000, seed=42):
    key = jax.random.PRNGKey(seed)
    x_key, y_key = jax.random.split(key)

    x1 = jax.random.uniform(x_key, shape=(num_samples,), minval=minval, maxval=maxval)
    x2 = jax.random.uniform(y_key, shape=(num_samples,), minval=minval, maxval=maxval)

    y = f(x1, x2).reshape(-1, 1)
    X = jnp.stack([x1, x2], axis=1)

    return X, y

seed = 42

X, y = generate_data(minval=-1, maxval=1, num_samples=1000, seed=seed)