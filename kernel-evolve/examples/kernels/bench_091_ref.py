"""Auto-generated Glaucis reference for L1 problem 91: cumsum_reverse."""

import jax
import jax.numpy as jnp


batch_size = 32768
input_shape = (32768,)
dim = 1


class Model:
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return jnp.flip(jnp.cumsum(jnp.flip(x, axis=self.dim), axis=self.dim), axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, *input_shape))]


def get_init_inputs():
    return [dim]


def simple_compute(_id=91):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
