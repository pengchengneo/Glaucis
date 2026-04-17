"""Auto-generated Glaucis reference for L1 problem 53: Min_reduction_over_a_dimension."""

import jax
import jax.numpy as jnp


batch_size = 128
dim1 = 4096
dim2 = 4095


class Model:
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x):
        return jnp.min(x, axis=self.dim)


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, dim1, dim2))
    return [x]


def get_init_inputs():
    return [1]


def simple_compute(_id=53):
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    return model(*inputs)


def reference_fn(**kwargs):
    return simple_compute(**kwargs)
